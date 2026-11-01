"""Retry curves (place memory): n lives in a row from each held start, the place records kept between lives,
vs the reload control (the same brain loaded fresh before every life: records wiped), same idle-frame offsets,
same sampling seeds. Max x per life (not the running best) and clears.
    python retry.py --brain brains0/Agrippa_ck0.npz --levels 1-2,2-1 --out runs/p0_retry.json [--blame 5,30,-1]
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


def summarize(res, lives):
    out = {}
    for key in sorted({(r["level"], r["control"]) for r in res}):
        rs = [r for r in res if (r["level"], r["control"]) == key]
        xs = np.array([[l["x"] for l in r["lives"]] for r in rs]); fl = np.array([[l["flag"] for l in r["lives"]] for r in rs])
        out[f"{key[0]}|{'control' if key[1] else 'keep'}"] = dict(
            median_x=[int(v) for v in np.median(xs, 0)], mean_x=[int(v) for v in xs.mean(0)], clears=[int(v) for v in fl.sum(0)],
            starts_cleared_any=int(fl.any(1).sum()), n=len(rs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True); ap.add_argument("--levels", default="1-2,2-1")
    ap.add_argument("--starts", default="32,35,38,41,80,85,90,95"); ap.add_argument("--lives", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.05); ap.add_argument("--blame", default=None)
    ap.add_argument("--arms", default="keep,control"); ap.add_argument("--off", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    blame = None if not a.blame else tuple(float(v) for v in a.blame.split(","))
    blame = None if blame is None else (int(blame[0]), int(blame[1]), blame[2])
    tasks = [dict(policy=str((HERE / a.brain).resolve()), level=L, start=int(s), lives=a.lives, control=(arm == "control"),
                  temp=a.temp, blame=blame, off=tuple(p for p in a.off.split("+") if p))
             for L in a.levels.split(",") for s in a.starts.split(",") for arm in a.arms.split(",")]
    t = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=mem.worker_init) as ex:
        res = list(ex.map(mem.retry_series, tasks, chunksize=1))
    summ = summarize(res, a.lives)
    for k, v in summ.items():
        print(k, v, flush=True)
    mem.jdump((HERE / a.out).resolve(), dict(summary=summ, results=res, args=vars(a), seconds=round(time.time() - t)))
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
