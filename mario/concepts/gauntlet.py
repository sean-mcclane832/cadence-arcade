"""Gauntlet (Sean's rule, updated 10/31): real start of 1-1, FIVE lives. A flag moves on to the next level
(1-2, 1-3, 1-4, 2-1, ...); a death restarts the CURRENT level from its start with the same brain (no
learning between lives, no extra lives). Game over on the 5th death. Every attempt is clipped whole,
deaths included.

    python gauntlet.py --name U3ck83 --brain ../brains/U3ck83.npz --out runs/gauntlet/g01_U3ck83
    python gauntlet.py --name relay --relay 1-1=../brains/1-1_U3ck3.npz,1-2=...,default=../brains/U3ck83.npz --out ...

Notes: each level starts as a fresh level with small Mario (a mushroom does not carry over). The first try
of a level uses 0 idle frames ("start 0"); retries use 37, 83, 110, 131 idle frames, because the emulator is
deterministic and an identical restart would replay the identical death. The 40-decision
stuck limit counts as losing a life (the brain stood still); the 900-decision cap also.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cplay as C  # noqa: E402
import dagger as D  # noqa: E402

RETRY_STARTS = (0, 37, 83, 110, 131)  # idle frames for try 1..5 of a level (the emulator is deterministic:
# an identical restart replays the identical death, so each retry starts a few frames later, like a real restart)
ORDER = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]


def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--brain", default=None)
    ap.add_argument("--relay", default=None); ap.add_argument("--senses", default="body")
    ap.add_argument("--lives", type=int, default=5); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="text on the clip, e.g. 'one brain' or 'relay of specialists'")
    a = ap.parse_args()
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from game import ACTIONS
    out = (HERE / a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    if a.relay:
        rel = dict(kv.split("=") for kv in a.relay.split(","))
        pols = {k: str((HERE / v).resolve()) for k, v in rel.items()}
    else:
        pols = {"default": str((HERE / a.brain).resolve())}
    label = a.label or ("relay of specialists" if a.relay else "one brain")
    w = imageio_ffmpeg.write_frames(str(out) + ".mp4", (512, 480 + 40), fps=30, codec="libx264", pix_fmt_out="yuv420p",
                                    output_params=["-crf", "30", "-preset", "medium", "-movflags", "+faststart"],
                                    macro_block_size=8)
    w.send(None)
    f16 = font(18)
    nf = [0]

    def send(screen, text, colour=(30, 30, 30)):
        nf[0] += 1
        if nf[0] % 2:
            return
        img = Image.new("RGB", (512, 520), colour)
        img.paste(Image.fromarray(screen).resize((512, 480), Image.NEAREST), (0, 40))
        d = ImageDraw.Draw(img); d.text((8, 10), text, fill=(255, 255, 255), font=f16)
        w.send(np.ascontiguousarray(np.asarray(img)))

    lives = a.lives; li = 0; log = []; cleared = []; t0 = time.time(); poster = None
    while lives > 0 and li < len(ORDER):
        L = ORDER[li]; lvl = tuple(int(v) for v in L.split("-"))
        pol_path = pols.get(L, pols.get("default"))
        pol = D._policy(pol_path, "cadence")
        who = Path(pol_path).stem if a.relay else a.name
        tries = sum(1 for e in log if e["level"] == L)
        st = RETRY_STARTS[min(tries, len(RETRY_STARTS) - 1)]
        b = C.CBody(st, a.senses, "lean", lvl); g = b.g
        text = f"{label}: {who} | {L} try {tries + 1} | lives {lives}/{a.lives} | cleared {len(cleared)}"
        while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
            o = b.obs()
            act = int(np.asarray(pol.predict(o[None])).reshape(-1)[0])
            if a.senses in ("ram", "ramscreen"):
                b.eprev = C.concept_map(g.env.ram)[1]
            fr = []
            g.step(act, keep_frames=fr); b.held = ACTIONS[act]
            for f in fr:
                send(f, text)
            if poster is None and g.decisions == 40:
                poster = g.screen.copy()
        end = "flag" if g.flag else ("died" if g.done else "stuck")
        tail = f"{L}: FLAG" if g.flag else f"{L}: {'died' if g.done else 'stuck'} at x {g.max_x}"
        for _ in range(45 if g.flag else 30):
            try:
                s_, _, _, _ = g.env.step(0)
            except Exception:  # noqa: BLE001
                break
            send(s_, text + "   " + tail, (20, 90, 20) if g.flag else (120, 20, 20))
        g.close()
        log.append(dict(level=L, brain=who, start=st, end=end, x=int(g.max_x), decisions=g.decisions, lives_before=lives))
        print(f"{L} {who}: {end} x {g.max_x} (lives {lives})", flush=True)
        if g.flag:
            cleared.append(L); li += 1
        else:
            lives -= 1
    w.close()
    if poster is not None:
        Image.fromarray(poster).resize((512, 480), Image.NEAREST).save(str(out) + "_poster.png")
    deaths = {}
    for e in log:
        if e["end"] != "flag":
            deaths.setdefault(e["level"], []).append(e["x"])
    rep = dict(name=a.name, label=label, pols=pols, lives=a.lives, cleared=cleared, furthest=log[-1]["level"] if log else None,
               deaths=deaths, lives_left=lives, attempts=log, seconds=round(time.time() - t0),
               when=time.strftime("%Y-%m-%d %H:%M"), clip=out.name + ".mp4", clip_seconds=round(nf[0] / 60, 1))
    Path(str(out) + ".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("name", "cleared", "deaths", "lives_left", "clip_seconds")}), flush=True)


if __name__ == "__main__":
    main()
