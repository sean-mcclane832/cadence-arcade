"""E6: why does Cadence fit the body-sense coach labels worse and less steadily than the MLP?
Offline, rand 80/20 split of a dagger.py data file. Usage:
  python fit_body.py ARM LESSONS [data_dir]
ARM tokens (joined by _): e1 / e03 / e3 (teaching steps), nosurp, fs4k (free_steps 4096),
c1 (one body copy instead of 8), b16 (16-row lessons), h512, cont (continue: report every 1k)."""
import os, sys, json, time, dataclasses
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
import cadence as cd
arm, L = sys.argv[1], int(sys.argv[2])
src = sys.argv[3] if len(sys.argv) > 3 else "../runs/fix_mlp_body2"
t = set(arm.split("_"))
d = np.load(f"{src}/data.npz"); X, y, W = d["X"], d["y"], d["W"]
if "c1" in t:
    X = X[:, :1664 + 42]
n = len(y); rng = np.random.default_rng(0); tr = rng.random(n) < 0.8; te = ~tr
kw = dict(teaching="normalized", learning_surprise="nosurp" not in t)
if "fs4k" in t: kw["learning_free_steps"] = 4096
b = cd.Brain.compose(X.shape[1], 7, modules=(512 if "h512" in t else 256,), seed=0, **kw)
eta = 0.001 if "e1" in t else 0.0003 if "e03" in t else None
if eta:
    c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=eta, eta_bias=eta / 10)
B = 16 if "b16" in t else 32
p = W[tr] / W[tr].sum(); Xtr, ytr, Wtr = X[tr], y[tr], W[tr]
def acc(Xs, ys, ws):
    pr = np.concatenate([np.asarray(b.predict(Xs[i:i + 500] / 255.0)).reshape(-1) for i in range(0, len(ys), 500)])
    c = ws > 1
    return round(float((pr == ys).mean()), 3), round(float((pr[c] == ys[c]).mean()), 3)
r = np.random.default_rng(2); t0 = time.time(); unsettled = []; recs = []
import warnings; warnings.simplefilter("ignore")
for i in range(L + 1):
    if i % 1000 == 0:
        rec = dict(arm=arm, lesson=i, s=round(time.time() - t0), train=acc(Xtr, ytr, Wtr), test=acc(X[te], y[te], W[te]),
                   unsettled=round(float(np.mean(unsettled)), 3) if unsettled else None)
        print(json.dumps(rec), flush=True); unsettled = []
    if i == L: break
    idx = r.choice(len(ytr), B, p=p)
    _, rep = b.learner.step(b.stimulus(Xtr[idx] / 255.0, memory=False), ytr[idx])
    fs = rep.get("free_steps", 0); unsettled.append(float(np.mean(np.asarray(fs) >= b.learner.config.free_steps)))
