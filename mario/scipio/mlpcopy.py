"""Leg 2a: a plain backprop MLP on the same piles, same split and same two numbers as tinycopy.py.
    idle.sh 098 mlpcopy.py --arm plain --seed 0 [--epochs 2]
3,536 -> 256 (ReLU) -> 7, Adam 1e-3, batch 256, rows weighted like the lessons (critical x3 via the pile weight).
The MLP has no memory: each frame is answered alone (the held buttons and the previous frame are in the input).
Arms: plain | change (change rows x4) | noheld (held-button inputs blanked in training and scoring) | drop (blanked on half the rows).
Writes scratch/tiny/mlp_<arm>_<seed>.json."""
import argparse, json, os, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np, torch
from pathlib import Path
from tinycopy import load_split, LEVELS, T_EVAL
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="plain"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--sub", type=float, default=1.0)
a = ap.parse_args()
train, held = load_split(); rs = np.random.default_rng(5)
X, y, w, ch = [], [], [], []
for L in LEVELS:
    for t in train[L]:
        m = t["y"] >= 0
        if a.sub < 1: m = m & (rs.random(len(m)) < a.sub)
        X.append(t["O"][m]); y.append(t["y"][m]); ww = t["w"][m]; w.append(np.where(ww > 1, 3.0, 1.0)); ch.append((t["y"] != t["prev"])[m])
X = np.concatenate(X); y = np.concatenate(y); w = np.concatenate(w); ch = np.concatenate(ch)
if a.arm.startswith("change"): w = w * np.where(ch, float(a.arm[6:] or 4), 1.0)
def prep(o, drop=None):
    x = torch.from_numpy(o.astype(np.float32) / 255.0)
    if a.arm in ("noheld", "aonly") or drop is not None:
        v = x[:, 1664:2000].reshape(len(x), 8, 42)
        if a.arm == "noheld": v[:, :, 38:42] = 0
        elif a.arm == "aonly": keep = v[:, 0, 38].clone(); v[:, :, 38:42] = 0; v[:, 0, 38] = keep
        else: v[drop, :, 38:42] = 0
    return x
torch.manual_seed(a.seed); r = np.random.default_rng(a.seed)
net = torch.nn.Sequential(torch.nn.Linear(3536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7)); opt = torch.optim.Adam(net.parameters(), 1e-3)
yt, wt = torch.from_numpy(y.astype(np.int64)), torch.from_numpy(w.astype(np.float32)); steps = int(a.epochs * len(X) / 256); t0 = time.time()
for s in range(steps):
    j = r.integers(0, len(X), 256)
    x = prep(X[j], drop=(r.random(256) < 0.5) if a.arm == "drop" else None)
    loss = (torch.nn.functional.cross_entropy(net(x), yt[j], reduction="none") * wt[j]).sum() / wt[j].sum()
    opt.zero_grad(); loss.backward(); opt.step()
def score(atts, T):
    ok, c, hb = [], [], []
    with torch.no_grad():
        for q in atts:
            n = min(T, len(q["y"])); ans = net(prep(q["O"][:n])).argmax(1).numpy()
            ok.append(ans == q["y"][:n]); c.append(q["y"][:n] != q["prev"][:n]); hb.append(ans == q["prev"][:n])
    ok, c, hb = np.concatenate(ok), np.concatenate(c), np.concatenate(hb)
    return dict(all=round(float(ok.mean()), 4), repeat=round(float(ok[~c].mean()), 4), change=round(float(ok[c].mean()), 4), n=int(ok.size), n_change=int(c.sum()), answers_held_button=round(float(hb.mean()), 4))
out = dict(arm=a.arm, seed=a.seed, epochs=a.epochs, rows=int(len(X)), steps=steps, s=round(time.time() - t0), held=score([held[L] for L in LEVELS], T_EVAL), seen=score([train[L][0] for L in LEVELS], 100))
Path("scratch/tiny").mkdir(exist_ok=True); torch.save(net.state_dict(), f"scratch/tiny/mlp_{a.arm}_{a.seed}_E{int(a.epochs)}.pt"); json.dump(out, open(f"scratch/tiny/mlp_{a.arm}_{a.seed}" + ("" if a.epochs == 2 else f"_E{int(a.epochs)}") + ".json", "w"), indent=1); print(json.dumps(out))
