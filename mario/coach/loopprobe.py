"""Replay a recorded coach run (runs.json) and log the loop-castle RAM per decision: Mario's page and x, y,
state, the level loader's page/column, LoopCommand and the World 7 multi-loop counters. Saves screens at chosen
decisions. Addresses are from the SMB disassembly (smbdis).
    python loopprobe.py --runs runs/cov_w4_7 --level 7-4 --start 0 --at 100,120,140 --out runs/loop_7-4
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import game  # noqa: E402

RAM = dict(page=0x6D, px=0x86, y=0xCE, state=0x1D, ldpage=0x0725, ldcol=0x0726, loop=0x0745,
           mlc=0x06D9, mlp=0x06DA, area=0x0760, world=0x075F)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs/cov_w4_7")
    ap.add_argument("--level", required=True); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--at", default=""); ap.add_argument("--out", required=True); ap.add_argument("--upto", type=int, default=10**9)
    a = ap.parse_args()
    r = next(q for q in json.loads((HERE / a.runs / "runs.json").read_text()) if q["level"] == a.level and q["start"] == a.start)
    at = {int(v) for v in a.at.split(",") if v}
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    g = game.Game(noop_start=a.start, level=tuple(int(v) for v in a.level.split("-")), menu="lean", bg_per_frame=True)
    e = g.env; rows = []
    from PIL import Image
    for k, act in enumerate(r["acts"][:a.upto]):
        if g.done:
            break
        sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
        e._time_last, e._x_position_last, e.done = sv
        g.step(int(act))
        ram = e.ram
        row = dict(i=k + 1, a=int(act), x=int(g.x), raw=int(e._x_position), **{n: int(ram[ad]) for n, ad in RAM.items()})
        rows.append(row)
        if k + 1 in at:
            Image.fromarray(g.screen).save(out / f"d{k + 1:04d}.png")
    (out / "rows.json").write_text(json.dumps(rows))
    prev = None
    for row in rows:  # print the rows where anything loop-related changes
        key = (row["loop"], row["mlc"], row["mlp"], row["ldpage"], row["area"], row["page"])
        if key != prev:
            print(row)
        prev = key


if __name__ == "__main__":
    main()
