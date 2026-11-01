"""Score brains in act mode (memory on, reset per attempt, greedy) or predict mode, on start sets and levels.
    python evalm.py --out runs/x.json --brains Cicero=brains0/Cicero_ck0.npz,W1=../brains/W1_F2ck110.npz@predict \
        --levels 1-1,1-2 --sets held [--off Cicero=thought+button]
NAME=PATH[@predict]. --off NAME=part+part switches memory parts off at play (ablation)."""
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
SETS = dict(sel=C.SEL, test=C.TEST, held=C.HELD, h8=(32, 35, 38, 41, 80, 85, 90, 95))


def run(brains, levels, sets, workers, off=None, pairs=None):
    tasks, keys = [], []
    for n, (p, kind) in brains.items():
        for L in levels:
            if pairs and (n, L) not in pairs:
                continue
            starts = sorted({st for k in sets for st in SETS[k]})
            for st in starts:
                tasks.append(dict(policy=p, kind=kind, level=L, start=st, off=tuple((off or {}).get(n, ()))))
                keys.append((n, L, st))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=mem.worker_init) as ex:
        res = list(ex.map(mem.episode, tasks, chunksize=1))
    rep = {}
    for (n, L, st), r in zip(keys, res):
        rep.setdefault(f"{n}|{L}", {})[str(st)] = dict(x=r["max_x"], flag=r["flag"], end=r["end"], dec=r["decisions"],
                                                       actions=r["actions"])
    summ = {}
    for k, d in rep.items():
        row = {}
        for sname in sets:
            ss = [str(v) for v in SETS[sname]]
            row[sname] = dict(flags=sum(d[s]["flag"] for s in ss), n=len(ss), mean=round(float(np.mean([d[s]["x"] for s in ss]))),
                              max=max(d[s]["x"] for s in ss))
        summ[k] = row
    return summ, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--brains", required=True)
    ap.add_argument("--levels", required=True); ap.add_argument("--sets", default="held")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--off", default=None)
    ap.add_argument("--pairs", default=None)
    a = ap.parse_args()
    brains = {}
    for spec in a.brains.split(","):
        n, p = spec.split("="); kind = "act"
        if "@" in p:
            p, kind = p.split("@")
        brains[n] = (str((HERE / p).resolve()), kind)
    off = {k: v.split("+") for k, v in (s.split("=") for s in a.off.split(","))} if a.off else None
    pairs = None if not a.pairs else {tuple(x.split(":")) for x in a.pairs.split(",")}
    t = time.time()
    summ, rep = run(brains, a.levels.split(","), a.sets.split(","), a.workers, off, pairs)
    for k, row in summ.items():
        print(k, " ".join(f"{s} {v['flags']}/{v['n']} mean {v['mean']} max {v['max']}" for s, v in row.items()), flush=True)
    out = (HERE / a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summ, detail=rep, off=a.off, seconds=round(time.time() - t))))
    print(f"done {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
