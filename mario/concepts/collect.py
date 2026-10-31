"""Coach demonstrations for the concepts run, stored with the FULL sense set (ramscreen: eyes | body | RAM map),
so any brain (body, ram, ramscreen) can be taught from the same rows. Same recipe as dagger.py round 0:
6 clean coach runs + 3 with 12% random buttons + 2 with 25%.
    python collect.py --levels 1-1,1-2,1-3,1-4 --out runs/p1
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cplay as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = (HERE / a.out).resolve()
    tasks, keys = [], []
    for L in a.levels.split(","):
        for s, eps, seed in [(s, 0.0, s) for s in (0, 3, 8, 13, 20, 27)] + [(s, 0.12, 100 + s) for s in (1, 15, 24)] + \
                [(s, 0.25, 200 + s) for s in (10, 22)]:
            tasks.append(C.task("none", "ramscreen", L, s, mode="coach", eps=eps, seed=seed)); keys.append((L, s, eps))
    t = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=C.worker_init) as ex:
        res = list(ex.map(C.episode, tasks, chunksize=1))
    for L in a.levels.split(","):
        rs = [(k, r) for k, r in zip(keys, res) if k[0] == L]
        X = np.concatenate([r["obs"] for _, r in rs]); y = np.concatenate([r["labels"] for _, r in rs])
        W = np.concatenate([r["weights"] for _, r in rs])
        paths = [(k[1], r["actions"]) for k, r in rs if r["flag"] and k[2] == 0.0]
        d = out / f"coach_{L}"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "data.npz", X=X, y=y, W=W); (d / "coach_paths.json").write_text(json.dumps(paths))
        print(f"{L}: {len(y)} rows, {int((W > 1).sum())} critical, flags {sum(r['flag'] for _, r in rs)}/{len(rs)}, "
              f"clean paths {len(paths)}, ends {[r['end'][0] for _, r in rs]} x {[r['max_x'] for _, r in rs]}", flush=True)
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
