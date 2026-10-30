"""E19: labelled rows with both senses (old 2,000 + fovea 1,152) on one level.
python collect_fov.py OUTDIR LEVEL BRAIN EPISODES WORKERS"""
import sys, json, time, os
sys.path.insert(0, "..")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import dagger as D
from game import MENUS
if __name__ == "__main__":
    out, level, brain, n, w = sys.argv[1], tuple(int(v) for v in sys.argv[2].split("-")), os.path.abspath(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(out, exist_ok=True)
    mb = sorted({a for p in MENUS["lean"] for a in p})
    base = dict(senses="fovea", menu="lean", kind="cadence", menu_buttons=mb, level=level, pol_inputs=2000, policy=brain)
    rng = np.random.default_rng(3)
    paths = [tuple(c) for c in json.loads(open(sys.argv[6]).read())] if len(sys.argv) > 6 else []
    ts = [dict(base, mode="coach", start=s, eps=e, seed=s) for s, e in ((2, 0.0), (19, 0.0), (45, 0.0), (60, 0.12), (70, 0.12), (77, 0.25))]
    pool = [s for s in range(80) if not 32 <= s <= 41]
    for i in range(n):
        if paths and i % 3:
            s0, p = paths[int(rng.integers(len(paths)))]; k = int(rng.integers(0, max(1, len(p) - 20)))
            ts.append(dict(base, mode="dagger", start=s0, prefix=p[:k], seed=500 + i))
        else:
            ts.append(dict(base, mode="dagger", start=int(rng.choice(pool)), seed=500 + i))
    ex = ProcessPoolExecutor(w, mp_context=get_context("spawn"), initializer=D._worker_init)
    t = time.time(); res = list(ex.map(D.episode, ts))
    X = np.concatenate([r["obs"] for r in res if len(r["labels"])]); y = np.concatenate([r["labels"] for r in res if len(r["labels"])])
    W = np.concatenate([r["weights"] for r in res if len(r["labels"])])
    np.savez(f"{out}/data.npz", X=X, y=y, W=W)
    print(f"{len(y)} rows, {X.shape[1]} cols, {time.time() - t:.0f}s, ends {[r['end'][0] for r in res]}")
