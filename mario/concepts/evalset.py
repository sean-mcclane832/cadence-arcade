"""Score saved brains on start sets (SEL / TEST / HELD) and levels, one pool for everything.
    python evalset.py --out runs/p0/held.json --brains U3ck83=../brains/U3ck83.npz,... --levels 1-1,1-2 --sets held
A brain spec NAME=PATH[@senses]; senses default body (dagger.py), or ram / ramscreen (cplay.py, CHEAT)."""
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

SETS = dict(sel=C.SEL, test=C.TEST, held=C.HELD, wide=C.WIDE, h80=tuple(range(80, 100)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--brains", required=True)
    ap.add_argument("--levels", required=True); ap.add_argument("--sets", default="held")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--pairs", default=None, help="optional NAME:LEVEL list restricting which brain-level pairs run")
    a = ap.parse_args()
    out = (HERE / a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    brains = {}
    for spec in a.brains.split(","):
        n, p = spec.split("="); s = "body"
        if "@" in p:
            p, s = p.split("@")
        brains[n] = (str((HERE / p).resolve()), s)
    pairs = None if not a.pairs else {tuple(x.split(":")) for x in a.pairs.split(",")}
    tasks, keys = [], []
    for n, (p, s) in brains.items():
        for L in a.levels.split(","):
            if pairs and (n, L) not in pairs:
                continue
            starts = sorted({st for k in a.sets.split(",") for st in SETS[k]})
            for st in starts:
                tasks.append(C.task(p, s, L, st)); keys.append((n, L, st))
    t = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=C.worker_init) as ex:
        res = list(ex.map(C.episode, tasks, chunksize=1))
    rep = {}
    for (n, L, st), r in zip(keys, res):
        d = rep.setdefault(f"{n}|{L}", {})
        d[str(st)] = dict(x=r["max_x"], flag=r["flag"], end=r["end"], dec=r["decisions"])
    summ = {}
    for k, d in rep.items():
        row = {}
        for sname in a.sets.split(","):
            ss = [str(v) for v in SETS[sname]]
            row[sname] = dict(flags=sum(d[s]["flag"] for s in ss), n=len(ss), mean=round(float(np.mean([d[s]["x"] for s in ss]))),
                              max=max(d[s]["x"] for s in ss))
        summ[k] = row
        print(k, " ".join(f"{s} {v['flags']}/{v['n']} mean {v['mean']} max {v['max']}" for s, v in row.items()), flush=True)
    out.write_text(json.dumps(dict(summary=summ, detail=rep, seconds=round(time.time() - t)), indent=1))
    print(f"done {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
