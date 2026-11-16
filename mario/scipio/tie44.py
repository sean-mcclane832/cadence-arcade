"""4-4 start stalls: which starts make the tie rule fire (noop within 2 px of the best plan)? Holding noop as the brain does."""
import sys
from pathlib import Path
MARIO = Path(__file__).resolve().parents[1]
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
import numpy as np
import cs, mem, plug
plug.worker_init(cs.deep_levels(), "smart", {})
C = mem.C
N = ["noop", "right", "r+j", "run", "run+j", "jump", "left"]
for L in ("4-4", "4-2"):
    for s in cs.SEL8:
        body = C.CBody(s, "fabius", "lean", tuple(int(v) for v in L.split("-"))); g = body.g; ties = []; labs = []
        for t in range(0, 130):
            if t in (0, 40, 80, 125, 126, 127, 128, 129):
                sc = np.asarray(g.coach_scores()); first = np.array([p[0] for p in g.plans])
                lab, _ = g.coach_weighted()
                tie = lab != 0 and sc[first == 0].max() >= np.nanmax(sc) - 2.0
                labs.append(N[0] if (lab == 0 or tie) else N[lab]); ties.append(int(tie))
            body.step(0)
        print(L, "start", s, "x", g.x, "label as written at t 0/40/80/125-129:", labs, "tie fired:", ties, flush=True)
        g.close()
