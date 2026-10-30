"""Record a stored solo attempt from a dagger.py run (history.json) as an MP4 of real NES frames.
Replays the brain's stored buttons from the same start jitter (deterministic emulator) and checks
the replay ends at the same x and flag before writing.

    python record_fix.py --run runs/fix_cad_body3 --round 2 --start 0 --out ../docs/mario/levels/fix-1-1.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from game import Game  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--start", type=int, required=True); ap.add_argument("--level", default="1-1")
    ap.add_argument("--file", default="history.json", help="history.json, or first_sight.json (eval-only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads((HERE / a.run / a.file).read_text())
    evs = data[a.level] if a.file != "history.json" else next(h for h in data if h["round"] == a.round)["eval"]
    pick = next(e for e in evs if e["start"] == a.start)
    lv = tuple(int(v) for v in a.level.split("-"))
    g = Game(noop_start=pick["start"], level=lv, menu="lean", bg_per_frame=True)
    frames = [g.screen.copy()]
    for act in pick["actions"]:
        if g.done:
            break
        g.step(act, keep_frames=frames)
    ok = g.max_x == pick["max_x"] and g.flag == pick["flag"]
    print(f"replay max_x {g.max_x} flag {g.flag} (stored {pick['max_x']} {pick['flag']}) -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)
    for _ in range(90 if g.flag else 30):
        try:
            s, _, _, _ = g.env.step(0); frames.append(s.copy())
        except Exception:  # noqa: BLE001
            break
    frames = frames[::2]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(out), (512, 480), fps=30, codec="libx264", pix_fmt_out="yuv420p",
                                    output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"],
                                    macro_block_size=16)
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f.repeat(2, axis=0).repeat(2, axis=1)))
    w.close()
    from PIL import Image
    Image.fromarray(frames[min(len(frames) - 1, 45)].repeat(2, 0).repeat(2, 1)).save(out.with_name(out.stem + "_poster.png"))
    meta = dict(run=a.run, level=a.level, round=a.round, start=pick["start"], max_x=int(g.max_x), flag=bool(g.flag),
                decisions=len(pick["actions"]), seconds=round(len(frames) / 30, 1), bytes=out.stat().st_size)
    out.with_suffix(".json").write_text(json.dumps(meta)); print(json.dumps(meta))
    g.close()


if __name__ == "__main__":
    main()
