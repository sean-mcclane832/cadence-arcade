import os, sys
from pathlib import Path
MARIO = Path(__file__).resolve().parents[1]
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
import numpy as np
import cs, mem, plug
plug.worker_init(cs.deep_levels(), "smart", {})
C = mem.C
N = ["noop", "right", "r+j", "run", "run+j", "jump", "left"]
for L, s, hold in (("4-2", 48, 0), ("4-2", 48, 150), ("4-4", 0, 0), ("4-4", 0, 150), ("1-1", 0, 0)):
    body = C.CBody(s, "fabius", "lean", tuple(int(v) for v in L.split("-"))); g = body.g
    for _ in range(hold):
        body.step(0)
    sc = np.asarray(g.coach_scores()); first = np.array([p[0] for p in g.plans])
    best = {N[k]: round(float(sc[first == k].max()), 1) for k in sorted(set(first))}
    lab, w = g.coach_weighted()
    tie = sc.max() - sc[first == 0].max() <= 2.0
    print(L, s, "after", hold, "noop decisions: best score by first button", best, "| coach label", N[lab], "| noop within 2 px (tie rule keeps noop):", tie, flush=True)
    g.close()
