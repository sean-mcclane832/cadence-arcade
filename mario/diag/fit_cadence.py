"""D1: can a Cadence 0.91.1 brain fit the coach labels? Same rand20 split as fit_torch.py
(rng 0). Usage: python fit_cadence.py ARM [data.npz] [lessons]"""
import os, sys, json, time, dataclasses
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import cadence as cd
arm = sys.argv[1]
p = sys.argv[2] if len(sys.argv) > 2 else "../runs/main/data.npz"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
d = np.load(p); X = d["X"]; y = d["y"]; W = d["W"]
if "coachonly" in arm:
    k = W >= 1; X, y, W = X[k], y[k], W[k]
n = len(y); rng = np.random.default_rng(0)
tr = rng.random(n) < 0.8; te = ~tr
Xtr, ytr, Wtr = X[tr], y[tr], W[tr]; Xte, yte, Wte = X[te], y[te], W[te]
scale = 2.0 if "x2" in arm else 1.0
def prep(a): return np.clip(a.astype(np.float64) / 255.0 * scale, 0, 1)
kw = dict(teaching="normalized", learning_surprise=True)
if "nosurp" in arm: kw["learning_surprise"] = False
if "founder" in arm: kw = dict(learning_surprise="surp" in arm and "nosurp" not in arm)
if "rel" in arm: kw.update(learning_normalize_relative=0.1, learning_nudged_steps=32, learning_nudged_fixed=True)
hidden = 512 if "h512" in arm else 256
b = cd.Brain.compose(X.shape[1], 7, modules=(hidden,), seed=0, **kw)
if "e1" in arm:
    c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=0.001, eta_bias=0.0001)
if "e03" in arm:
    c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=0.0003, eta_bias=0.00003)
w = None if "unif" in arm else Wtr / Wtr.sum()
ev_tr = np.random.default_rng(1).choice(len(ytr), 2000, replace=False)
ev_te = np.arange(min(2000, len(yte)))
def acc(Xs, ys, ws):
    pr = np.concatenate([np.asarray(b.predict(prep(Xs[i:i + 500]))).reshape(-1) for i in range(0, len(ys), 500)])
    c = ws > 1
    return dict(all=round(float((pr == ys).mean()), 3), crit=round(float((pr[c] == ys[c]).mean()), 3) if c.any() else None,
                pred_counts=np.bincount(pr, minlength=7).tolist())
log = open(f"fit_{arm}.log", "w")
t0 = time.time(); r = np.random.default_rng(2)
for i in range(L + 1):
    if i in (0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000) or i == L:
        rec = dict(lesson=i, s=round(time.time() - t0), train=acc(Xtr[ev_tr], ytr[ev_tr], Wtr[ev_tr]), test=acc(Xte[ev_te], yte[ev_te], Wte[ev_te]))
        log.write(json.dumps(rec) + "\n"); log.flush()
    if i == L: break
    idx = r.choice(len(ytr), 32, p=w)
    b.learner.step(b.stimulus(prep(Xtr[idx]), memory=False), ytr[idx])
b.save(f"fit_{arm}.npz")
