"""E9: is the 1-1 loss after 1-2 practice capacity or drift? Teach a brain on BOTH levels' coach
labels (share of 1-1 rows per lesson), then test solo on both and the label fit on each.
  python consolidate.py BRAIN OUT LESSONS SHARE_1_1"""
import os, sys, json, time, dataclasses
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, "..")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import dagger as D
from game import MENUS

def fit(b, X, y, W, n=2000, seed=0):
    i = np.random.default_rng(seed).choice(len(y), min(n, len(y)), replace=False)
    pr = np.concatenate([np.asarray(b.predict(X[i[k:k + 500]] / 255.0)).reshape(-1) for k in range(0, len(i), 500)])
    c = W[i] > 1
    return round(float((pr == y[i]).mean()), 3), round(float((pr[c] == y[i][c]).mean()), 3)

if __name__ == "__main__":
    import cadence as cd
    brain_p, out, L, share = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 2000; eta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.001
    os.makedirs(out, exist_ok=True)
    b = cd.Brain.load(brain_p); c = b.learner.config
    b.learner.config = dataclasses.replace(c, eta=eta, eta_bias=eta / 10)
    A = np.load("../runs/fix_cad_body3/data.npz"); B = np.load("../runs/fix_cad_1-2/data.npz")
    A = (A["X"], A["y"], A["W"]); B = (B["X"], B["y"], B["W"])
    ex = ProcessPoolExecutor(2, mp_context=get_context("spawn"), initializer=D._worker_init)
    mb = sorted({a for p in MENUS["lean"] for a in p})
    def solo(lv):
        p = os.path.abspath(f"{out}/policy.npz"); b.save(p)
        ts = [dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb, level=lv, mode="eval", start=s, policy=p)
              for s in (0, 5, 11, 17, 23)]
        return [r["max_x"] for r in ex.map(D.episode, ts)]
    rng = np.random.default_rng(5); recs = []
    for i in range(0, L + 1, chunk):
        if i:
            D.teach(b, "cadence", B[0], B[1], B[2], chunk, rng, lambda m: None, old=A, share=share)
        rec = dict(lesson=i, fit11=fit(b, *A), fit12=fit(b, *B), solo11=solo((1, 1)), solo12=solo((1, 2)))
        print(json.dumps(rec), flush=True); recs.append(rec); b.save(f"{out}/ck_{i:05d}.npz")
    b.save(f"{out}/brain.npz"); ex.shutdown()
