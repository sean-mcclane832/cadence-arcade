"""Cheap reference players on the same 5 eval starts as the brain.

- random: a uniformly random button every decision (5 different random streams per start)
- run: hold run (right + B) forever
- run+jump rhythm: run+jump for 8 decisions, run for 4, repeat (a "spam jump" heuristic)
- mlp (optional, --mlp runs/main): a plain 1-hidden-layer MLP (256 units, Adam, backprop) trained
  on the SAME final labelled screens the brain learned from, same passes; it plays greedily.
  It gets the brain's data for free (no DAgger rounds of its own), so it is a cheap reference,
  not a matched rival.

    python baselines.py --out runs/baselines.json [--mlp runs/main]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from game import Game, RUN, RUNJUMP  # noqa: E402
from train import EVAL_STARTS, MAX_DECISIONS, STUCK  # noqa: E402


def play(policy, start: int) -> dict:
    g = Game(noop_start=start)
    acts = []
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        a = policy(g)
        acts.append(a)
        g.step(a)
    out = dict(start=start, max_x=g.max_x, flag=g.flag, decisions=g.decisions)
    g.close()
    return out


class MLP:
    def __init__(self, n_in, hidden=256, n_out=7, seed=0, lr=1e-3):
        r = np.random.default_rng(seed)
        self.p = [r.normal(0, 1 / np.sqrt(n_in), (n_in, hidden)), np.zeros(hidden),
                  r.normal(0, 1 / np.sqrt(hidden), (hidden, n_out)), np.zeros(n_out)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]
        self.t, self.lr = 0, lr

    def logits(self, x):
        h = np.maximum(0, x @ self.p[0] + self.p[1])
        return h, h @ self.p[2] + self.p[3]

    def step(self, x, y):
        h, z = self.logits(x)
        pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        g = pr; g[np.arange(len(y)), y] -= 1; g /= len(y)
        gh = (g @ self.p[2].T) * (h > 0)
        grads = [x.T @ gh, gh.sum(0), h.T @ g, g.sum(0)]
        self.t += 1
        for i, gr in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr; self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def act(self, x):
        return int(np.argmax(self.logits(x[None])[1][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/baselines.json")
    ap.add_argument("--mlp", default=None)
    ap.add_argument("--passes", type=float, default=6.0)
    args = ap.parse_args()
    res = {}
    rows = []
    for s in EVAL_STARTS:
        for k in range(5):
            rng = np.random.default_rng(1000 * s + k)
            rows.append(play(lambda g, r=rng: int(r.integers(7)), s))
    res["random"] = rows
    res["run"] = [play(lambda g: RUN, s) for s in EVAL_STARTS]
    res["run+jump rhythm"] = [play(lambda g: RUNJUMP if g.decisions % 12 < 8 else RUN, s) for s in EVAL_STARTS]
    if args.mlp:
        d = np.load(HERE / args.mlp / "data.npz")
        X, y = d["X"], d["y"]
        freq = np.bincount(y, minlength=7).astype(float)
        w = 1.0 / np.sqrt(np.maximum(freq[y], 1.0)); w /= w.sum()
        rng = np.random.default_rng(0)
        net = MLP(X.shape[1])
        for i in range(int(args.passes * len(y) / 32)):
            idx = rng.choice(len(y), 32, p=w)
            net.step(X[idx] / 255.0, y[idx])
        agree = float(np.mean([net.act(X[i] / 255.0) == y[i] for i in rng.choice(len(y), 2000)]))
        res["mlp"] = [play(lambda g: net.act(g.observation()), s) for s in EVAL_STARTS]
        res["mlp_info"] = dict(samples=int(len(y)), passes=args.passes, train_agree=agree)
    for k, v in res.items():
        if isinstance(v, list):
            xs = [r["max_x"] for r in v]
            print(f"{k:18s} median x {np.median(xs):6.0f}  max {max(xs)}  flags {sum(r['flag'] for r in v)}/{len(v)}")
    (HERE / args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
