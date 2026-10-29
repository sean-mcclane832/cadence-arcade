"""First-sight baselines per level, same protocol as levels.py first sight (starts 0, 6, 12 idle
frames; same stop rules): always-run, always run+jump (held), random buttons (10 attempts, fixed
seeds, starts cycling 0/6/12), and a fresh untrained Cadence brain (same input as the blind run).

    python baselines_levels.py --levels 1-1,1-2,... --out runs/baselines_levels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"

from game import Game, N_IN, RUN, RUNJUMP  # noqa: E402

STARTS = (0, 6, 12)
MAX_DECISIONS, STUCK = 1500, 60


def play(level, start, policy):
    g = Game(noop_start=start, level=level, menu="lean", bg_per_frame=True)
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        g.step(policy(g))
    r = dict(start=start, max_x=int(g.max_x), flag=bool(g.flag))
    g.close()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out", default="runs/baselines_levels.json")
    ap.add_argument("--fresh", default="1")
    args = ap.parse_args()
    out = HERE / args.out
    res = json.loads(out.read_text()) if out.exists() else {}
    fresh = None
    if args.fresh == "1":
        from cadence import Brain
        fresh = Brain.compose(N_IN, 7, modules=(256,), seed=0, teaching="normalized", learning_surprise=True)
    for name in args.levels.split(","):
        if name in res:
            continue
        lv = tuple(int(v) for v in name.split("-"))
        r = {}
        r["run"] = [play(lv, s, lambda g: RUN) for s in STARTS]
        r["runjump"] = [play(lv, s, lambda g: RUNJUMP) for s in STARTS]
        rnd = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            rnd.append(play(lv, STARTS[seed % 3], lambda g: int(rng.integers(7))))
        r["random"] = rnd
        if fresh is not None:
            r["fresh"] = [play(lv, s, lambda g: int(np.asarray(fresh.predict(g.observation()[None])).reshape(-1)[0]))
                          for s in STARTS]
        r["median"] = {k: float(np.median([e["max_x"] for e in v])) for k, v in r.items() if isinstance(v, list)}
        res[name] = r
        out.write_text(json.dumps(res))
        print(name, r["median"], flush=True)


if __name__ == "__main__":
    main()
