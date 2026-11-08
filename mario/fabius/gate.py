"""Recall-off gate (vault Plans/Fabius Build 2026-11-08, deep review fix 2), pre-registered: at chunks 10 and 25 a
Fabius checkpoint plays SEL8 on 6 levels with its recall memory on and off (recall_amplitude 0). If off beats on by
more than the noise floor (the Cato2 seed gap on the same levels at the same chunk), stop and rethink recall.
    python gate.py --run FabiusA --chunk 10 [--workers 2]
Writes runs/gate_<run>_ck<chunk>.json (per-level mean best x for on / off).
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
COLD = HERE.parent / "coldstart"
sys.path.insert(0, str(COLD))
import cs  # noqa: E402
LEVELS = ("1-2", "2-1", "3-3", "4-1", "5-2", "6-4")  # fixed in advance: one per world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    ck = COLD / "runs" / a.run / f"{a.run}_ck{a.chunk:04d}.npz"
    assert ck.exists(), ck
    res = {}
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=cs.worker_init, initargs=({},)) as ex:
        for name, off in (("on", ()), ("off", ("recall",))):
            tasks = [dict(policy=str(ck), kind="act", level=L, start=s, senses="fabius", off=off) for L in LEVELS for s in cs.SEL8]
            out = list(ex.map(cs.mem.episode, tasks, chunksize=1))
            res[name] = {L: round(float(np.mean([r["max_x"] for r in out[i * 8:(i + 1) * 8]]))) for i, L in enumerate(LEVELS)}
            res[name]["mean"] = round(float(np.mean([res[name][L] for L in LEVELS])))
    print(json.dumps(res))
    (HERE / "runs" / f"gate_{a.run}_ck{a.chunk:04d}.json").write_text(json.dumps(dict(ckpt=ck.name, levels=LEVELS, **res), indent=1))


if __name__ == "__main__":
    main()
