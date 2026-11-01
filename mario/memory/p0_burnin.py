"""Phase 0.4: burn-in K. Play real attempts in act mode, record every memory's true state before each decision, then
rebuild it from K burn-in acts (+ exact history / efference) and compare (cosine). Smallest K with mean cosine >= 0.95
for every memory, take the max.
    python p0_burnin.py --out runs/p0_burnin.json
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mem  # noqa: E402
KS = (0, 1, 2, 3, 4, 6, 8, 12)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--brains", default="Cicero=brains0/Cicero_ck0.npz,Livy=brains0/Livy_ck0.npz")
    a = ap.parse_args()
    rep = {}
    for spec in a.brains.split(","):
        name, p = spec.split("="); p = str((HERE / p).resolve())
        res = {}
        for level, start in (("1-2", 32), ("1-3", 80), ("1-1", 90)):
            pl = mem.MemPlayer(p, "greedy")
            true = []

            def cb(g, o, act):
                pass
            # record states BEFORE each decision: wrap act
            b = pl.b; orig = pl.act

            def act_rec(o):
                wm = None if b.working_memory is None or len(b.working_memory.trace) == 0 else b.working_memory.trace[0].copy()
                ef = None if b.efference is None or len(b.efference.trace) == 0 else b.efference.trace[0].copy()
                hi = None if b.history is None or len(b.history.trace) == 0 else b.history.trace[0].copy()
                true.append((wm, ef, hi))
                return orig(o)
            pl.act = act_rec
            r = mem.play_attempt(pl, level, start, keep_obs=True)
            O, A = r["obs"], np.array(r["actions"])
            b2 = mem.MemPlayer(p, "greedy").b
            d_h, d_e = mem.mem_decays(b2)
            rows = list(range(5, len(A), 9))
            out = {}
            for K in KS:
                cw, ce, ch = [], [], []
                for i in rows:
                    mem.prime(b2, [(O, A, i)], K)
                    wm, ef, hi = true[i]
                    if wm is not None:
                        t0 = b2.working_memory.trace; cw.append(cos(t0[0] if len(t0) else np.zeros_like(wm), wm))
                    if ef is not None:
                        ce.append(cos(b2.efference.trace[0], ef))
                    if hi is not None:
                        ch.append(cos(b2.history.trace[0], hi))
                out[K] = dict(trace=round(float(np.mean(cw)), 4) if cw else None,
                              trace_min=round(float(np.min(cw)), 4) if cw else None,
                              efference=round(float(np.mean(ce)), 4) if ce else None,
                              history=round(float(np.mean(ch)), 4) if ch else None, n=len(rows))
                print(name, level, start, "K", K, out[K], flush=True)
            res[f"{level}/{start}"] = dict(end=r["end"], x=r["max_x"], decisions=len(A), by_k=out)
        rep[name] = res
    mem.jdump((HERE / a.out).resolve(), rep)


if __name__ == "__main__":
    main()
