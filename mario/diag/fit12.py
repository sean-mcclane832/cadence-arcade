"""E12: offline Cadence fit on 1-1 / 1-2 coach labels. python fit12.py OUT UNITS SHARE_11 LESSONS [INIT_BRAIN]"""
import os, sys, json, time, dataclasses
sys.path.insert(0, "..")
import numpy as np
import cadence as cd
out, units, share, L = sys.argv[1], tuple(int(u) for u in sys.argv[2].split("x")), float(sys.argv[3]), int(sys.argv[4])
init = sys.argv[5] if len(sys.argv) > 5 else None
A = np.load("../runs/fix_cad_body3/data.npz"); B = np.load("../runs/d12/data.npz")
rng = np.random.default_rng(0)
def split(d):
    n = len(d["y"]); tr = rng.random(n) < 0.8
    return [(d["X"][m], d["y"][m], d["W"][m].astype(float)) for m in (tr, ~tr)]
(A_tr, A_te), (B_tr, B_te) = split(A), split(B)
if init:
    b = cd.Brain.load(init)
else:
    b = cd.Brain.compose(2000, 7, modules=units, seed=0, teaching="normalized", learning_surprise=False)
c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=0.001, eta_bias=0.0001)
def fit(D, n=1500):
    X, y, W = D; i = np.random.default_rng(1).choice(len(y), min(n, len(y)), replace=False)
    pr = np.concatenate([np.asarray(b.predict(X[i[k:k + 500]] / 255.0)).reshape(-1) for k in range(0, len(i), 500)])
    return round(float((pr == y[i]).mean()), 3)
pa = A_tr[2] / A_tr[2].sum(); pb = B_tr[2] / B_tr[2].sum(); t = time.time()
for k in range(1, L + 1):
    na = rng.binomial(32, share)
    ia = rng.choice(len(A_tr[1]), na, p=pa); ib = rng.choice(len(B_tr[1]), 32 - na, p=pb)
    x = np.concatenate([A_tr[0][ia], B_tr[0][ib]]) / 255.0; y = np.concatenate([A_tr[1][ia], B_tr[1][ib]])
    b.learner.step(b.stimulus(x, memory=False), y)
    if k % 1000 == 0:
        r = dict(lessons=k, s=round(time.time() - t), f11=(fit(A_tr), fit(A_te)), f12=(fit(B_tr), fit(B_te)))
        print(json.dumps(r), flush=True)
b.save(out + ".npz")
