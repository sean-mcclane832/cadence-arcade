"""Post-hoc SEL8 scoring of saved checkpoints (Romulus read amendment 11/22, vault F-101: score every checkpoint 82-100,
not every other one). Same tasks as ctrain.consider(): mem.episode, kind act, 24 levels x cs.SEL8, fabius senses,
cs.worker_init(deep) with deep = cs.deep_levels(). Needs flags.sh's env (MARIO_STUCK=225).
    python selpost.py <out_dir> <ckpt.npz> [<ckpt.npz> ...] [--workers 3]
Writes <out_dir>/<ckpt stem>.json (per level: x, flags, mean, died, stuck, nostart) and skips ones already done.
Parity: an even checkpoint must reproduce its log.txt SEL line (check before trusting odd ones).
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "coldstart", HERE.parent, HERE.parent / "memory"):
    sys.path.insert(0, str(p))
import cs  # noqa: E402
import mem  # noqa: E402

LEVELS = [f"{w}-{s}" for w in range(1, 7) for s in range(1, 5)]


def summ(ev):  # as ctrain.summ
    return dict(x=[r["max_x"] for r in ev], flags=sum(r["flag"] for r in ev), mean=round(float(np.mean([r["max_x"] for r in ev]))),
                died=sum(r["end"] == "died" for r in ev), stuck=sum(r["end"] == "stuck" for r in ev),
                nostart=sum(r["end"] == "stuck" and r["max_x"] <= 60 for r in ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    assert cs.C.D.STUCK == 225, cs.C.D.STUCK
    if sys.platform == "win32":  # BelowNormal, like training; the spawned workers inherit it
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    deep = cs.deep_levels(); sel = tuple(cs.SEL8)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=cs.worker_init, initargs=(deep,)) as ex:
        for c in a.ckpts:
            c = Path(c).resolve(); f = out / (c.stem + ".json")
            if f.exists():
                print(f"skip {c.stem}", flush=True); continue
            t0 = time.time()
            tasks = [dict(policy=str(c), kind="act", level=L, start=s, senses="fabius") for L in LEVELS for s in sel]
            res = list(ex.map(mem.episode, tasks, chunksize=1))
            rec = {L: summ(res[i * len(sel):(i + 1) * len(sel)]) for i, L in enumerate(LEVELS)}
            tot = dict(flags=sum(r["flags"] for r in rec.values()), died=sum(r["died"] for r in rec.values()),
                       stuck=sum(r["stuck"] for r in rec.values()), mean=round(float(np.mean([r["mean"] for r in rec.values()]))))
            f.write_text(json.dumps(dict(ckpt=str(c), seconds=round(time.time() - t0), total=tot, levels=rec), indent=1))
            print(f"{c.stem}: {tot} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
