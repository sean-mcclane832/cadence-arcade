"""Phase 0: do brains saved under Cadence 0.91.1 load and act the same under 0.91.3?
Runs predict() on a fixed set of stored screens (first 3,000 rows of each level's pile) and saves the answers.
    <python> equiv.py OUT.npz"""
import sys, json
from pathlib import Path
import numpy as np
import cadence
from cadence import Brain
HERE = Path(__file__).resolve().parent; M = HERE.parent
brains = ["U3ck83", "one_11_13", "1-1_U3ck3", "1-2_12n59", "U3ck28"]
extra = {"st_U3_ck0044": M / "runs/st_U3/ck_0044.npz"}
piles = {"1-1": "runs/dV_1-1", "1-3": "runs/dV_1-3", "1-4": "runs/dV_1-4", "1-2": "runs/d12f"}
X = []
for L, d in piles.items():
    p = M / d / "data.npz"
    if not p.exists():
        p = M / d / f"data_{L}.npz"
    z = np.load(p); X.append(z["X"][:750])
X = np.concatenate(X).astype(np.float64) / 255.0
out = {"version": cadence.__version__}
paths = {b: M / "brains" / f"{b}.npz" for b in brains}; paths.update(extra)
for name, p in paths.items():
    b = Brain.load(p)
    out[name] = np.concatenate([np.asarray(b.predict(X[i:i + 500])).reshape(-1) for i in range(0, len(X), 500)])
    print(name, np.bincount(out[name], minlength=7).tolist(), flush=True)
np.savez(sys.argv[1], **out)
