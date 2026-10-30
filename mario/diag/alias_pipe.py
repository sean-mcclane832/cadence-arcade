"""E2: at a stuck point, does the coach's label depend on the held button, with the screen unchanged?"""
import sys, json, numpy as np
sys.path.insert(0, "..")
from game import Game, ACTION_NAMES, RUN, RUNJUMP
h = json.load(open("../runs/main/history.json"))
for rnd, si in ((18, 0), (19, 0), (17, 5)):
    e = [r for r in h if r["round"] == rnd][0]["eval"][si]
    g = Game(noop_start=e["start"])
    for a in e["actions"][:-5]:
        g.step(a)
    last = e["actions"][-6]
    o0 = g.observation(); lab0 = g.coach_weighted()
    seq = []
    for a in (RUN, RUN, RUNJUMP, RUNJUMP, RUN):
        o1 = g.observation(); lab = g.coach_weighted()
        seq.append((ACTION_NAMES[a], "coach->", ACTION_NAMES[lab[0]], lab[1], "x", g.x, "screen diff vs stuck", round(float(np.abs(o1 - o0).max()), 3), "A held", int(g.env.ram[0x0A] & 0x80 > 0)))
        g.step(a)
    print(f"round {rnd} start {e['start']} stuck at {e['max_x']} holding {ACTION_NAMES[last]}: coach says {ACTION_NAMES[lab0[0]]} w{lab0[1]}")
    for s in seq: print("   then pressed", s)
    g.close()
