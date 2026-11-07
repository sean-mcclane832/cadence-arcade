"""The registered test (plan: vault Plans/Flappy Brain 2026-11-06): 20 fresh games, seed 5000, cap 100 pipes,
brain greedy with no learning, vs the teacher and a random flapper on the same games. Run once.
    python test.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from cadence import Brain
import game as G

HERE = Path(__file__).resolve().parent
N, SEED, CAP = 20, 5000, 100


def run(policy):
    g = G.Flappy(N, seed=SEED, cap=CAP)
    while not g.dead.all():
        g.step(policy(g))
    return g.score.copy()


def main():
    b = Brain.load(HERE / "brains" / "corvus.npz"); b.reset()
    rng = np.random.default_rng(SEED)
    res = {"brain": run(lambda g: np.asarray(b.act(g.senses(), greedy=True)).reshape(-1)),
           "teacher": run(G.teacher),
           "random (flaps 8% of frames)": run(lambda g: (rng.random(g.n) < 0.08).astype(int))}
    rep = {k: dict(median=float(np.median(v)), mean=float(v.mean()), min=int(v.min()), max=int(v.max()), scores=v.tolist())
           for k, v in res.items()}
    rep["rule"] = "plays = brain median >= 20 pipes (cap 100), 20 games, seed 5000, greedy, no learning"
    rep["passed"] = rep["brain"]["median"] >= 20
    (HERE / "test_result.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(k, v if not isinstance(v, dict) else {x: v[x] for x in ("median", "mean", "min", "max")})


if __name__ == "__main__":
    main()
