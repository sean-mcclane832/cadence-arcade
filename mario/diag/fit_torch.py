"""D1/D2: can a backprop learner fit the coach labels, and how consistent are they (1-NN)?
Runs in the cadence venv (torch). Usage: python fit_torch.py data.npz [coach_only]"""
import sys, json, numpy as np, torch
p = sys.argv[1]; coach_only = len(sys.argv) > 2
d = np.load(p); X = d["X"].astype(np.float32) / 255.0; y = d["y"]; W = d["W"]
if coach_only:
    k = W >= 1; X, y, W = X[k], y[k], W[k]
n = len(y); dev = "cuda"
rng = np.random.default_rng(0)
splits = {"late20": (np.arange(n) < int(0.8 * n)), "rand20": rng.random(n) < 0.8}
out = {"file": p, "n": n, "labels": np.bincount(y, minlength=7).tolist()}
def acc(pred, yy, ww):
    return dict(all=float((pred == yy).mean()), crit=float((pred[ww > 1] == yy[ww > 1]).mean()) if (ww > 1).any() else None)
for name, tr in splits.items():
    te = ~tr
    Xt, Xe = torch.tensor(X[tr], device=dev), torch.tensor(X[te], device=dev)
    yt, ye = torch.tensor(y[tr], device=dev), torch.tensor(y[te], device=dev)
    wt = torch.tensor(W[tr], device=dev, dtype=torch.float32)
    res = {"majority_te": float((y[te] == np.bincount(y[tr]).argmax()).mean())}
    # 1-NN
    pr = []
    for i in range(0, len(Xe), 1024):
        dist = torch.cdist(Xe[i:i + 1024], Xt); pr.append(yt[dist.argmin(1)])
    pr = torch.cat(pr).cpu().numpy(); res["1nn_te"] = acc(pr, y[te], W[te])
    # train-set 1NN excluding self = label consistency
    pr = []
    for i in range(0, len(Xt), 1024):
        dist = torch.cdist(Xt[i:i + 1024], Xt); dist[torch.arange(dist.shape[0]), torch.arange(i, i + dist.shape[0])] = 1e9
        pr.append(yt[dist.argmin(1)])
    pr = torch.cat(pr).cpu().numpy(); res["1nn_train_loo"] = acc(pr, y[tr], W[tr])
    for arch in ("logreg", "mlp256"):
        torch.manual_seed(0)
        D = X.shape[1]
        net = torch.nn.Linear(D, 7) if arch == "logreg" else torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7))
        net = net.to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
        for weighted in (False, True):
            pass
        for ep in range(60):
            perm = torch.randperm(len(yt), device=dev)
            for i in range(0, len(yt), 32):
                b = perm[i:i + 32]
                loss = (torch.nn.functional.cross_entropy(net(Xt[b]), yt[b], reduction="none") * wt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep in (4, 19, 59):
                with torch.no_grad():
                    a_tr = acc(net(Xt).argmax(1).cpu().numpy(), y[tr], W[tr]); a_te = acc(net(Xe).argmax(1).cpu().numpy(), y[te], W[te])
                res[f"{arch}_ep{ep+1}"] = dict(train=a_tr, test=a_te)
    out[name] = res
print(json.dumps(out, indent=1))
