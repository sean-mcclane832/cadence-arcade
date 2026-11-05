"""Plateau test scorer (COLDSTART.md, "Plateau test (11/5)", pre-registered before launch).

Curve: per chunk, the 12-level mean of SEL max x on the 8 fixed starts cs.SEL8. Chunks scored on all 24 starts (before
the 11/5 resume) are reduced to the same 8 starts from the per-start x saved in history.jsonl, so every chunk of every
run uses the same 8 starts.
Primary: mean of that curve over the last 10 chunks ending at the common chunk C (default 200; --at C to score at an
earlier chunk, e.g. where CiceroLow is). Arm = mean of seeds A and B. Floor = largest within-arm seed gap.
"Memory plateaus higher" holds for an arm only if BOTH its seeds beat the same-letter Cato seed by more than the floor.
Secondary: 60-chunk OLS slope ending at C (x per chunk, with SE).

    python plateau.py            # at chunk 200 (or the furthest chunk all runs reached, with a warning)
    python plateau.py --at 150   # every run that reached chunk 150
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402

RUNS = ["CatoA", "CatoB", "CiceroA", "CiceroB", "LivyA", "LivyB", "CiceroLowA", "CiceroLowB"]
IDX24 = [list(cs.SEL).index(s) for s in cs.SEL8]


def curve(run):
    c = {}
    p = HERE / "runs" / run / "history.jsonl"
    if not p.exists():
        return c
    for line in open(p):
        h = json.loads(line)
        per = []
        for L in cs.TRAIN:
            x = h["solo"][L]["x"]
            per.append(np.mean([x[i] for i in IDX24]) if len(x) == len(cs.SEL) else np.mean(x))
        c[int(h["chunk"])] = float(np.mean(per))
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--at", type=int, default=200); ap.add_argument("--w", type=int, default=10)
    args = ap.parse_args()
    cur = {r: curve(r) for r in RUNS}
    C = args.at
    rep = {"at": C}
    for r, c in cur.items():
        ks = list(range(C - args.w + 1, C + 1))
        if not all(k in c for k in ks):
            print(f"{r}: not at chunk {C} yet (last {max(c) if c else None})"); continue
        prim = float(np.mean([c[k] for k in ks]))
        ks60 = [k for k in range(C - 59, C + 1) if k in c]
        A = np.vstack([ks60, np.ones(len(ks60))]).T; y = np.array([c[k] for k in ks60])
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        se = float(np.sqrt(res[0] / (len(y) - 2) / np.sum((np.array(ks60) - np.mean(ks60)) ** 2))) if len(res) else float("nan")
        rep[r] = dict(primary=round(prim, 1), slope60=round(float(coef[0]), 2), slope60_se=round(se, 2))
        print(f"{r}: primary {prim:.1f}  slope60 {coef[0]:+.2f} +- {se:.2f} x/chunk")
    arms = {}
    for a in ("Cato", "Cicero", "Livy", "CiceroLow"):
        if f"{a}A" in rep and f"{a}B" in rep:
            pa, pb = rep[f"{a}A"]["primary"], rep[f"{a}B"]["primary"]
            arms[a] = dict(mean=round((pa + pb) / 2, 1), gap=round(abs(pa - pb), 1))
    if arms:
        floor = max(v["gap"] for v in arms.values()); rep["arms"] = arms; rep["floor"] = floor
        print("arms:", arms, "floor", floor)
        if "Cato" in arms:
            for a in ("Cicero", "Livy", "CiceroLow"):
                if a not in arms:
                    continue
                dA = rep[f"{a}A"]["primary"] - rep["CatoA"]["primary"]; dB = rep[f"{a}B"]["primary"] - rep["CatoB"]["primary"]
                v = ("PLATEAUS HIGHER" if min(dA, dB) > floor else "PLATEAUS LOWER" if max(dA, dB) < -floor
                     else "NOT DISTINGUISHABLE")
                rep[f"verdict_{a}"] = dict(dA=round(dA, 1), dB=round(dB, 1), verdict=v)
                print(f"{a} vs Cato: A {dA:+.1f}, B {dB:+.1f} (floor {floor}) -> {v}")
    (HERE / "runs" / f"plateau_{C}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
