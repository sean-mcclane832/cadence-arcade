"""2-1 springboard: how often do random sticky button sequences clear the wall from a coached state? (python springrate.py 310; needs runs/trace_2-1_3 and runs/t_smart_2-1_3p from trace.py)"""
import json, sys, numpy as np
sys.path.insert(0, r"P:/projects/cadence-arcade/mario/coach"); sys.path.insert(0, r"P:/projects/cadence-arcade/mario")
import harness; harness.below_normal()
import game, smartcoach as S
from game import RUN, RUNJUMP, NOOP, LEFT
H = r"P:/projects/cadence-arcade/mario/coach/runs/"
rows = json.load(open(H + "trace_2-1_3/trace.json"))["rows"][:280] + json.load(open(H + "t_smart_2-1_3p/trace.json"))["rows"]
n0 = int(sys.argv[1])
g = game.Game(noop_start=3, level=(2, 1), menu="lean", bg_per_frame=True); e = g.env
for q in rows[:n0]:
    sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
    e._time_last, e._x_position_last, e.done = sv; g.step(q["a"])
cont = [q["a"] for q in rows[n0:n0 + 40]]
rng = np.random.default_rng(0)
def sticky(maxd):
    s = []
    while len(s) < 40: s += [int(rng.choice(S.SHOOT_BUTTONS, p=S.SHOOT_P))] * int(rng.integers(1, maxd + 1))
    return s[:40]
with S.Sim(g) as sim:
    print("actual continuation:", sim.run(cont))
    for name, gen in (("sticky10", lambda: sticky(10)), ("sticky4", lambda: sticky(4))):
        ok = 0
        for _ in range(600):
            st, x, _ = sim.run(gen()); ok += (x > 3100 or st == S.FLAG)
        print(name, ok, "/ 600")
