"""E19: label consistency / learnability with and without the fovea (cadence venv, torch, GPU).
python fit_fov.py data.npz"""
import sys, numpy as np, torch
d = np.load(sys.argv[1]); X = d["X"].astype(np.float32) / 255.0; y = d["y"]; W = d["W"]
n = len(y); rng = np.random.default_rng(0); tr = rng.random(n) < 0.8; te = ~tr
sets = {"old (eyes+body)": np.r_[0:2000], "old+fovea": np.r_[0:X.shape[1]], "body+fovea": np.r_[1664:X.shape[1]], "fovea only": np.r_[2000:X.shape[1]]}
for name, cols in sets.items():
    Xs = torch.tensor(X[:, cols], device="cuda"); yt = torch.tensor(y, device="cuda")
    a, b = Xs[tr], Xs[te]
    dmat = torch.cdist(b, a); nn = yt[tr][dmat.argmin(1)]
    nn_te = float((nn == yt[te]).float().mean())
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(len(cols), 256), torch.nn.ReLU(), torch.nn.Linear(256, 7)).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3); w = torch.tensor(W[tr], device="cuda", dtype=torch.float32)
    idx = np.flatnonzero(tr)
    for ep in range(60):
        p = rng.permutation(len(idx))
        for k in range(0, len(p), 32):
            j = torch.tensor(p[k:k + 32], device="cuda")
            loss = (torch.nn.functional.cross_entropy(m(a[j]), yt[tr][j], reduction="none") * w[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ptr = m(a).argmax(1); pte = m(b).argmax(1)
    c = torch.tensor(W[te] > 1, device="cuda")
    print(f"{name:16s} 1-NN held-out {nn_te:.3f} | MLP train {float((ptr == yt[tr]).float().mean()):.3f} held-out "
          f"{float((pte == yt[te]).float().mean()):.3f} crit {float((pte[c] == yt[te][c]).float().mean()):.3f}", flush=True)
