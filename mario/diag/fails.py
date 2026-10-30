"""How solo attempts end: play a brain from starts, then have the coach label the last N decisions.
python fails.py BRAIN LEVEL STARTS N   (STARTS like 0,3,6)"""
import sys, json
sys.path.insert(0, "..")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import dagger as D
from game import MENUS, ACTION_NAMES
if __name__ == "__main__":
    brain, level, starts, N = sys.argv[1], tuple(int(v) for v in sys.argv[2].split("-")), [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4])
    mb = sorted({a for p in MENUS["lean"] for a in p})
    ex = ProcessPoolExecutor(min(4, len(starts)), mp_context=get_context("spawn"), initializer=D._worker_init)
    ts = [dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb, level=level, mode="dagger", start=s,
               policy=brain, last_all=N, every=10 ** 6, seed=s) for s in starts]
    for s, r in zip(starts, ex.map(D.episode, ts)):
        lab, own, w = r["labels"], r["own"], r["weights"]
        n = len(r["actions"]); keep = [i for i in range(n) if i >= n - N or i % 10 ** 6 == 0][-len(lab):] if len(lab) else []
        k = slice(len(lab) - min(N, len(lab)), None)
        crit_wrong = [(int(i), ACTION_NAMES[o], ACTION_NAMES[l]) for i, o, l, ww in zip(keep[k], own[k], lab[k], w[k]) if o != l and ww > 1]
        wrong = int(sum(o != l for o, l in zip(own[k], lab[k])))
        print(f"start {s:2d} end {r['end']:5s} x {r['max_x']:5d} dec {n:4d} | last {N}: wrong {wrong}, critical wrong {len(crit_wrong)} {crit_wrong[-2:]}", flush=True)
