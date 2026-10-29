"""Offline sweep of teaching settings on the round-0 coach screens (held-out = last 20% of rows,
i.e. whole later coach runs). One process per setting. Picks the setting for the real run.

    python sweep.py NAME   (NAME in CONFIGS)
"""
import json, os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent

CONFIGS = {
    "norm_surprise": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=False),
    "norm": dict(kw=dict(teaching="normalized"), weighted=False),
    "norm_surprise_eta1e-3": dict(kw=dict(teaching="normalized", learning_surprise=True, learning_eta=0.001, learning_eta_bias=0.0001), weighted=False),
    "founder": dict(kw=dict(), weighted=False),
    "norm_surprise_weighted": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=True),
    "founder_surprise": dict(kw=dict(learning_surprise=True), weighted=False),
    "norm_surprise_512": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=False, hidden=512),
    "sky_norm_surprise": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=False, tf="sky"),
    "sky_norm": dict(kw=dict(teaching="normalized"), weighted=False, tf="sky"),
    "sky_founder": dict(kw=dict(), weighted=False, tf="sky"),
    "z_norm_surprise": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=False, tf="z"),
    "sky_norm_surprise_weighted": dict(kw=dict(teaching="normalized", learning_surprise=True), weighted=True, tf="sky"),
}
for _k in ("sky_norm_surprise", "z_norm_surprise", "norm_surprise", "sky_founder"):
    CONFIGS["small_" + _k] = dict(CONFIGS[_k], small=True)
SKY_GREY = (0.299 * 92 + 0.587 * 148 + 0.114 * 252) / 255.0

EVERY = 250

def main(name, lessons=3000, data="runs/main/data.npz"):
    from cadence import Brain
    cfg = CONFIGS[name]
    d = np.load(HERE / data); X, y = d["X"], d["y"]
    X = X / 255.0
    if cfg.get("small"):  # 52 x 64 grey -> 26 x 32 (8 x 8 px blocks), both frames
        X = X.reshape(len(X), 2, 26, 2, 32, 2).mean(axis=(3, 5)).reshape(len(X), -1)
    if cfg.get("tf") == "sky":
        X = np.abs(X - SKY_GREY)
    elif cfg.get("tf") == "z":
        m, sd = X[: int(0.8 * len(y))].mean(0), X[: int(0.8 * len(y))].std(0) + 0.05
        X = (X - m) / sd / np.sqrt(X.shape[1])
    X = X * 255.0
    n = len(y); cut = int(0.8 * n)
    tr, te = np.arange(cut), np.arange(cut, n)
    b = Brain.compose(X.shape[1], 7, modules=(cfg.get("hidden", 256),), seed=0, **cfg["kw"])
    rng = np.random.default_rng(0)
    freq = np.bincount(y[tr], minlength=7).astype(float)
    w = 1 / np.sqrt(np.maximum(freq[y[tr]], 1)); w /= w.sum()
    out = []; t = time.time()
    for i in range(lessons + 1):
        if i % EVERY == 0:
            p = np.asarray(b.predict(X[te] / 255.0)).reshape(-1)
            ptr = np.asarray(b.predict(X[tr[::4]] / 255.0)).reshape(-1)
            row = dict(lesson=i, test=float((p == y[te]).mean()), train=float((ptr == y[tr[::4]]).mean()),
                       pred_counts=np.bincount(p, minlength=7).tolist(), sec=round(time.time() - t))
            out.append(row); print(name, row, flush=True)
        if i == lessons: break
        idx = rng.choice(tr, 32, p=w) if cfg["weighted"] else rng.choice(tr, 32)
        b.learner.step(b.stimulus(X[idx] / 255.0, memory=False), y[idx])
    (HERE / "runs/sweep" / f"{name}.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main(sys.argv[1])
