"""Where do the 4-4 noop labels come from? A 225-decision noop stall at 4-4 labelled exactly as mem.episode does
(cplay._play, tiew=True, last_all 100 + every 3rd), from DAgger's start pool (cs.POOL), under the old rule (-1).
    label44.py start [start ...]
"""
import os, sys, collections
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
MARIO = Path(__file__).resolve().parents[1]
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
import numpy as np
import cs, mem, plug
plug.worker_init(cs.deep_levels(), "smart", {})
C = mem.C
N = ["noop", "right", "r+j", "run", "run+j", "jump", "left"]
n = 225
idx = {i for i in range(n) if i >= n - 100 or i % 3 == 0}
for s in map(int, sys.argv[1:]):
    out = C._play(dict(C.task("none", "body", "4-4", s, mode="dagger"), script=[0] * n, tiew=True, keep_obs=False),
                  None, idx, np.random.default_rng(0))
    lab, w = out["labels"], out["weights"]
    c = collections.Counter((N[a], float(b)) for a, b in zip(lab, w))
    order = sorted(idx)[:len(lab)]
    first_noop = next((t for t, a in zip(order, lab) if a == 0), None)
    print("4-4 start", s, "end", out["end"], "decisions", out["decisions"], "labels (button, weight):", dict(c),
          "first noop label at t", first_noop, flush=True)
