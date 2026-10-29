"""Record a stored solo attempt (levels.py state.json) as an MP4 of real NES frames.

The emulator is deterministic: replaying the brain's stored buttons from the same start
reproduces the attempt frame for frame. The script checks that the replay ends at the same x and
flag before writing anything.

    python record_level.py --level 1-1 --round 7 --start 0 --out ../docs/mario/levels/1-1.mp4
    python record_level.py --level 1-2 --best --out ../docs/mario/levels/1-2-furthest.mp4
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
from levels import MENU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/levels/state.json")
    ap.add_argument("--level", required=True)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--best", action="store_true", help="the furthest solo attempt on this level")
    ap.add_argument("--retention", default=None, help="take the attempt from the retention test after LEVEL")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    st = json.loads((HERE / args.state).read_text())
    if args.retention:
        r = next(r for r in st["retention"] if r["after"] == args.retention)
        evs = [(-1, e) for e in r["results"][args.level]["eval"]]
    else:
        evs = [(h["round"], e) for h in st["history"] if h["level"] == args.level for e in h["eval"]
               if args.round is None or h["round"] == args.round]
    if args.start is not None:
        evs = [p for p in evs if p[1]["start"] == args.start]
    rnd, pick = max(evs, key=lambda p: (p[1]["flag"], p[1]["max_x"], -p[0]))
    lv = tuple(int(v) for v in args.level.split("-"))
    g = Game(noop_start=pick["start"], level=lv, menu=MENU)
    frames = [g.screen.copy()]
    for a in pick["actions"]:
        if g.done:
            break
        g.step(a, keep_frames=frames)
    # a few frames after the end, so the finish is visible
    tail = 60 if g.flag else 30
    ok = (g.max_x == pick["max_x"] and g.flag == pick["flag"])
    print(f"level {args.level} round {rnd} start {pick['start']}: replay max_x {g.max_x} flag {g.flag} "
          f"(stored {pick['max_x']} {pick['flag']}) -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        sys.exit(1)
    for _ in range(tail):
        try:
            s, _, _, _ = g.env.step(0)
            frames.append(s.copy())
        except Exception:  # noqa: BLE001
            break
    frames = frames[::2]  # 30 fps
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(out), (512, 480), fps=30, codec="libx264", pix_fmt_out="yuv420p",
                                    output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"],
                                    macro_block_size=16)
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f.repeat(2, axis=0).repeat(2, axis=1)))
    w.close()
    poster = out.with_name(out.stem + "_poster.png")
    from PIL import Image
    Image.fromarray(frames[min(len(frames) - 1, 45)].repeat(2, 0).repeat(2, 1)).save(poster)
    meta = dict(level=args.level, round=rnd, start=pick["start"], max_x=int(g.max_x), flag=bool(g.flag),
                decisions=len(pick["actions"]), seconds=round(len(frames) / 30, 1),
                bytes=out.stat().st_size, retention=args.retention)
    out.with_suffix(".json").write_text(json.dumps(meta))
    print(json.dumps(meta))
    g.close()


if __name__ == "__main__":
    main()
