"""Which button patterns get over 2-1's tall wall via the springboard (x ~3000)? Structured enumeration from a
coached state near the spring: [RUN]*a + [NOOP]*b + [RUNJUMP]*k + [RUN]*rest (and LEFT back-up variants).
    python springprobe.py --prefix-len 290
"""
from __future__ import annotations
import argparse, json, sys, itertools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=290); ap.add_argument("--L", type=int, default=40)
    a = ap.parse_args()
    harness.below_normal()
    import game, smartcoach as S
    from game import RUN, RUNJUMP, NOOP, LEFT
    rows = json.loads((HERE / "runs/trace_2-1_3/trace.json").read_text())["rows"][:280]
    rows += json.loads((HERE / "runs/t_smart_2-1_3p/trace.json").read_text())["rows"]
    g = game.Game(noop_start=3, level=(2, 1), menu="lean", bg_per_frame=True); e = g.env
    for q in rows[:a.prefix_len]:
        sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
        e._time_last, e._x_position_last, e.done = sv
        g.step(q["a"])
    r = e.ram
    print("root x", g.x, "y", int(r[0x00CE]), "enemies", [(hex(int(r[0x16 + i])), int(r[0x6E + i]) * 256 + int(r[0x87 + i]), int(r[0xCF + i]))
                                                   for i in range(5) if r[0x0F + i]])
    seqs = []
    for pre in (RUN, NOOP, LEFT):
        for a_, b, k in itertools.product(range(0, 14), range(0, 9, 2), (1, 2, 4, 8, 16)):
            s = [pre] * a_ + [NOOP] * b + [RUNJUMP] * k
            seqs.append((pre, a_, b, k, (s + [RUN] * a.L)[:a.L]))
    ok = []
    with S.Sim(g) as sim:
        for pre, a_, b, k, s in seqs:
            st, x, _ = sim.run(s)
            if x > 3100 or st == S.FLAG:
                ok.append((pre, a_, b, k, st, x))
    print(f"{len(ok)}/{len(seqs)} get past x 3100")
    for o in ok[:40]:
        print(o)


if __name__ == "__main__":
    main()
