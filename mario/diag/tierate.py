"""How many of the brain's disagreements with the coach are ties (<= 2 px), small (<= 50 px) or losing (> 50 px)?
python tierate.py BRAIN LEVEL STARTS"""
import sys
sys.path.insert(0, "..")
import numpy as np
import dagger as D
from game import MENUS, Game
brain, level, starts = sys.argv[1], tuple(int(v) for v in sys.argv[2].split("-")), [int(s) for s in sys.argv[3].split(",")]
mb = sorted({a for p in MENUS["lean"] for a in p})
tot = dict(rows=0, agree=0, tie=0, small=0, lose=0, crit_rows=0, crit_wrong=0)
for s in starts:
    r = D.episode(dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb, level=level, mode="eval", start=s, policy=brain))
    g = Game(noop_start=s, level=level, menu="lean", bg_per_frame=True)
    for a in r["actions"]:
        sc = g.coach_scores(); first = np.array([p[0] for p in g.plans]); best = sc.max()
        lab, w = g.coach_weighted(); tot["rows"] += 1
        mine = sc[first == a]; gap = best - (mine.max() if len(mine) else -1e9)
        if w > 1: tot["crit_rows"] += 1; tot["crit_wrong"] += int(a != lab)
        if a == lab: tot["agree"] += 1
        elif gap <= 2: tot["tie"] += 1
        elif gap <= 50: tot["small"] += 1
        else: tot["lose"] += 1
        g.step(a)
        if g.done: break
    print(s, r["end"], r["max_x"], dict(tot), flush=True)
