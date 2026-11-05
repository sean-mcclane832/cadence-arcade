"""Trace one coach run: x, y, button, and the plan scores at every decision; screenshots near the end.
    python trace.py --level 2-1 --start 0 --coach base --out runs/trace_2-1_0
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--coach", default="base"); ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--prefix", default="", help="trace.json whose actions to replay first")
    ap.add_argument("--prefix-len", type=int, default=0)
    ap.add_argument("--opts", default="{}")
    a = ap.parse_args()
    harness.below_normal()
    out = (HERE / a.out); out.mkdir(parents=True, exist_ok=True)
    rows, shots = [], []

    def cb(g, lab, wt, dt):
        ram = g.env.ram
        rows.append(dict(i=g.decisions, x=int(g.x), y=int(ram[0x00CE]), vx=int(np.int8(np.uint8(ram[0x0057]))),
                         a=int(lab), w=wt, dt=round(dt, 4)))
        shots.append(g.screen.copy()); del shots[:-a.shots]

    task = dict(level=a.level, start=a.start, coach=a.coach, opts=json.loads(a.opts))
    if a.prefix:
        task["prefix"] = [q["a"] for q in json.loads((HERE / a.prefix).read_text())["rows"]][:a.prefix_len]
    r = harness.run_one(task, cb=cb)
    r.pop("xs", None)
    (out / "trace.json").write_text(json.dumps(dict(result=r, rows=rows)))
    from PIL import Image
    for k, s in enumerate(shots):
        Image.fromarray(s).save(out / f"end_{k:02d}.png")
    print(json.dumps(r))
    xs = [q["x"] for q in rows]
    for q in rows[::10] + rows[-15:]:
        print(q)


if __name__ == "__main__":
    main()
