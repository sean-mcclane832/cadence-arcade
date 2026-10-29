"""Re-measure saved checkpoints: each start played by a freshly loaded copy of the brain.

    python evalckpt.py runs/main/brain_r03.npz runs/main/brain_r06.npz --out runs/main/ckpt_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from train import EVAL_STARTS, MAX_DECISIONS, STUCK  # noqa: E402


def play(path, start):
    from cadence import Brain
    from game import Game
    brain = Brain.load(path)
    g = Game(noop_start=start)
    acts = []
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        a = int(np.asarray(brain.predict(g.observation()[None])).reshape(-1)[0])
        acts.append(a); g.step(a)
    out = dict(start=start, max_x=g.max_x, flag=g.flag, decisions=g.decisions, actions=acts)
    g.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for c in args.ckpts:
        rows = [play(HERE / c, s) for s in EVAL_STARTS]
        again = play(HERE / c, EVAL_STARTS[0])
        res[c] = dict(eval=rows, repeat_start0_same=again["actions"] == rows[0]["actions"])
        print(c, [r["max_x"] for r in rows], "flags", sum(r["flag"] for r in rows),
              "deterministic:", res[c]["repeat_start0_same"], flush=True)
    (HERE / args.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
