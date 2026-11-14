"""Leg 2c run B: the leg 2a MLP on equal data with Cato2A (all 11 coach runs per level, plus Cato2A's own DAgger rows).
    idle.sh 098 mlpdag.py --arm all11|all11dag --seed 0 [--epochs 20]
Same net and training as mlpcopy.py: 3,536 -> 256 (ReLU) -> 7, Adam 1e-3, batch 256, rows with pile weight W > 1 weigh 3,
plain inputs. Rows: every labelled row (y >= 0) of all 11 coach runs of the 24 levels (nothing held out: the brains were
taught all 11); arm all11dag adds every labelled row of scratch/dag/Cato2A/seq_<L>_d*.npz (25 rounds, copied finished files).
These DAgger rows lie on Cato2A's path, not the MLP's: "equal data", not DAgger for the MLP.
Writes scratch/tiny/mlp_<arm>_<seed>_E<epochs>.pt and .json."""
import argparse, json, os, time, glob
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np, torch
from pathlib import Path
FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
LEVELS = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=("all11", "all11dag")); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=float, default=20.0)
a = ap.parse_args()


def rows(f):
    z = np.load(f); li = z["li"]; y = z["y"]; m = y >= 0
    return z["O"][li[m]], y[m], np.where(z["W"][m] > 1, 3.0, 1.0)


X, y, w = [], [], []; n_coach = n_dag = n_files = 0
for L in LEVELS:
    o, yy, ww = rows(FAB / L / f"seq_{L}_coach.npz"); X.append(o); y.append(yy); w.append(ww); n_coach += len(yy)
    if a.arm == "all11dag":
        for f in sorted(glob.glob(f"scratch/dag/Cato2A/seq_{L}_d*.npz")):
            o, yy, ww = rows(f); X.append(o); y.append(yy); w.append(ww); n_dag += len(yy); n_files += 1
X = np.concatenate(X); y = np.concatenate(y); w = np.concatenate(w)
if a.arm == "all11dag": assert n_files == 600, n_files
torch.manual_seed(a.seed); r = np.random.default_rng(a.seed)
net = torch.nn.Sequential(torch.nn.Linear(3536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7)); opt = torch.optim.Adam(net.parameters(), 1e-3)
yt, wt = torch.from_numpy(y.astype(np.int64)), torch.from_numpy(w.astype(np.float32)); steps = int(a.epochs * len(X) / 256); t0 = time.time()
for s in range(steps):
    j = r.integers(0, len(X), 256)
    x = torch.from_numpy(X[j].astype(np.float32) / 255.0)
    loss = (torch.nn.functional.cross_entropy(net(x), yt[j], reduction="none") * wt[j]).sum() / wt[j].sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 2000 == 0: print("step", s, "of", steps, "loss", round(float(loss), 4), round(time.time() - t0), "s", flush=True)
with torch.no_grad():
    fit = sum(int((net(torch.from_numpy(X[i:i + 4096].astype(np.float32) / 255.0)).argmax(1).numpy() == y[i:i + 4096]).sum()) for i in range(0, len(X), 4096)) / len(X)
tag = f"mlp_{a.arm}_{a.seed}_E{int(a.epochs)}"
Path("scratch/tiny").mkdir(exist_ok=True); torch.save(net.state_dict(), f"scratch/tiny/{tag}.pt")
out = dict(arm=a.arm, seed=a.seed, epochs=a.epochs, rows=int(len(X)), coach_rows=n_coach, dagger_rows=n_dag, dagger_files=n_files,
           critical_rows=int((w > 1).sum()), steps=steps, s=round(time.time() - t0), fit_all_rows=round(fit, 4))
json.dump(out, open(f"scratch/tiny/{tag}.json", "w"), indent=1); print(json.dumps(out))
