"""Own-path critical error (LEARNING_FIX E20's number): the brain plays alone; along ITS path the coach scores
every decision; of the coach's critical calls (weight 10), how many did the brain get wrong?
    python critpath.py --out runs/p0/crit_12n59.json --brain ../brains/1-2_12n59.npz --senses body --level 1-2
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cplay as C  # noqa: E402

STARTS = (32, 35, 38, 41, 80, 85, 90, 95, 104, 116, 128, 136)


def one(t):
    r = C.episode(t)
    from game import Game
    g = Game(noop_start=t["start"], level=t["level"], menu="lean", bg_per_frame=True)
    tot = dict(start=t["start"], end=r["end"], max_x=r["max_x"], rows=0, agree=0, tie=0, small=0, lose=0, crit_rows=0, crit_wrong=0,
               crit_wrong_x=[])
    for a in r["actions"]:
        sc = g.coach_scores(); first = np.array([p[0] for p in g.plans]); best = sc.max()
        lab, w = g.coach_weighted(); tot["rows"] += 1
        mine = sc[first == a]; gap = best - (mine.max() if len(mine) else -1e9)
        if w > 1:
            tot["crit_rows"] += 1
            if a != lab:
                tot["crit_wrong"] += 1; tot["crit_wrong_x"].append(int(g.x))
        if a == lab: tot["agree"] += 1
        elif gap <= 2: tot["tie"] += 1
        elif gap <= 50: tot["small"] += 1
        else: tot["lose"] += 1
        g.step(a)
        if g.done:
            break
    g.close()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--brain", required=True)
    ap.add_argument("--senses", default="body"); ap.add_argument("--level", default="1-2")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    ts = [C.task(str((HERE / a.brain).resolve()), a.senses, a.level, s) for s in STARTS]
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=C.worker_init) as ex:
        res = list(ex.map(one, ts))
    cr = sum(r["crit_rows"] for r in res); cw = sum(r["crit_wrong"] for r in res)
    rows = sum(r["rows"] for r in res)
    summ = dict(brain=a.brain, level=a.level, crit_rows=cr, crit_wrong=cw, crit_err=round(cw / max(1, cr), 3), rows=rows,
                agree=round(sum(r["agree"] for r in res) / rows, 3), lose=sum(r["lose"] for r in res),
                flags=sum(r["end"] == "flag" for r in res))
    for r in res:
        print(r["start"], r["end"], r["max_x"], f"crit {r['crit_wrong']}/{r['crit_rows']}", r["crit_wrong_x"], flush=True)
    print(json.dumps(summ), flush=True)
    out = (HERE / a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summ, per_start=res), indent=1))


if __name__ == "__main__":
    main()
