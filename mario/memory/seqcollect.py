"""Coach demonstrations WITH their order (burn-in needs the moments before each taught row): replay the clean coach
paths stored with the F2 piles, keep every decision's screen, coach labels (tie-aware) on every 3rd decision + the
last 60, as DAgger labels. One file per level: seq/seq_<L>_coach.npz.
    python seqcollect.py --out runs/seq
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mem  # noqa: E402
C = mem.C
PILES = {"1-1": "pile_1-1", "1-2": "pile_1-2", "1-3": "pile_1-3", "1-4": "pile_1-4"}


def replay(task):
    L, s, path = task["level"], task["start"], task["path"]
    n = len(path)
    idx = {i for i in range(n) if i >= n - 60 or i % 3 == 0}
    r = C._play(dict(C.task("none", "body", L, s, mode="dagger"), script=path, tiew=True, keep_obs=True), None, idx,
                np.random.default_rng(0))
    return dict(obs=r["obs"], labels=r["labels"], weights=r["weights"], actions=np.array(r["actions"]), flag=r["flag"])


def save_attempts(path, atts):
    """atts: list of dict(obs (n,2000) uint8, actions (n,), lab_idx, labels, weights) -> one npz."""
    O = np.concatenate([a["obs"] for a in atts]); A = np.concatenate([np.asarray(a["actions"], np.int64) for a in atts])
    off = np.cumsum([0] + [len(a["obs"]) for a in atts])
    li, y, W = [], [], []
    for k, a in enumerate(atts):
        li.append(off[k] + np.asarray(a["lab_idx"], np.int64)); y.append(np.asarray(a["labels"], np.int64))
        W.append(np.asarray(a["weights"], np.float64))
    np.savez(path, O=O, A=A, off=off, li=np.concatenate(li), y=np.concatenate(y), W=np.concatenate(W))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = (HERE / a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for L, d in PILES.items():
        for s, path in json.loads((HERE.parent / "concepts" / "runs" / "p4" / d / "coach_paths.json").read_text()):
            tasks.append(dict(level=L, start=int(s), path=list(path)))
    t = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=mem.worker_init) as ex:
        res = list(ex.map(replay, tasks, chunksize=1))
    for L in PILES:
        atts = []
        for tk, r in zip(tasks, res):
            if tk["level"] != L:
                continue
            m = r["labels"] >= 0
            atts.append(dict(obs=r["obs"], actions=r["actions"], lab_idx=np.flatnonzero(m), labels=r["labels"][m],
                             weights=r["weights"][m]))
            assert len(r["obs"]) == len(r["actions"]), (len(r["obs"]), len(r["actions"]))
        save_attempts(out / f"seq_{L}_coach.npz", atts)
        print(L, len(atts), "attempts", sum(len(x["obs"]) for x in atts), "decisions", sum(len(x["labels"]) for x in atts),
              "labelled", "flags", sum(r["flag"] for tk, r in zip(tasks, res) if tk["level"] == L), flush=True)
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
