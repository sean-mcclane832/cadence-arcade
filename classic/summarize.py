"""Summarize the runs: per-arm table, paired position-term comparisons, curves for the page.

    ../.venv-classic/Scripts/python.exe summarize.py   # writes runs/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ARMS = ("pos32", "pos0", "norecall", "mlp1", "mlp2", "random")


def load(run: Path) -> dict:
    lives: dict = {}
    for f in sorted(run.glob("*-*.json")):
        r = json.loads(f.read_text())
        lives.setdefault(r["arm"], {})[r["seed"]] = r
    return lives


def per_seed(r: dict, last: int) -> dict:
    return {"score": float(np.mean(r["score"][-last:])),
            "agree": float(np.mean(r["agree"][-last:])),
            "length": float(np.mean(r["length"][-last:]))}


def paired(a: dict, b: dict, last: int) -> dict:
    seeds = sorted(set(a) & set(b))
    d = np.array([per_seed(a[s], last)["score"] - per_seed(b[s], last)["score"] for s in seeds])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"seeds": seeds, "diff": d.round(3).tolist(), "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()), "median_diff": float(np.median(d)), "wilcoxon_p": p}


def table(lives: dict, last: int) -> dict:
    out = {}
    for arm, by in lives.items():
        rows = [per_seed(r, last) for r in by.values()]
        t = {}
        for k in ("score", "agree", "length"):
            v = np.array([x[k] for x in rows])
            t[k] = {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                    "q3": float(np.percentile(v, 75)), "mean": float(v.mean())}
        t["n"] = len(rows)
        t["position_maps"] = float(np.median([r["position"][-1] for r in by.values()]))
        t["warned"] = {k: int(sum(r.get("warned", {}).get(k, 0) for r in by.values()))
                       for k in ("unsettled", "same_features")}
        out[arm] = t
    return out


def curves(lives: dict) -> dict:
    out = {}
    for arm, by in lives.items():
        m = np.array([[np.mean(s) for s in r["score"]] for r in by.values()])
        g = np.array([r["agree"] for r in by.values()])
        out[arm] = {"med": np.median(m, 0).round(2).tolist(), "q1": np.percentile(m, 25, 0).round(2).tolist(),
                    "q3": np.percentile(m, 75, 0).round(2).tolist(),
                    "agree": np.median(g, 0).round(3).tolist()}
    return out


def main():
    main_run, rec = Path("runs/main"), Path("runs/recency0")
    meta = json.loads((main_run / "meta.json").read_text())
    last = meta["last"]
    L = load(main_run)
    S = {"meta": meta, "table": table(L, last), "curves": curves(L),
         "pos32_vs_pos0": paired(L["pos32"], L["pos0"], last),
         "pos0_vs_norecall": paired(L["pos0"], L["norecall"], last),
         "pos32_vs_norecall": paired(L["pos32"], L["norecall"], last),
         "mlp2_vs_mlp1": paired(L["mlp2"], L["mlp1"], last)}
    if (rec / "meta.json").exists():
        R = load(rec)
        if "pos32" in R and "pos0" in R:
            S["recency0"] = {"table": table(R, last), "curves": curves(R),
                             "pos32_vs_pos0": paired(R["pos32"], R["pos0"], last)}
    ident = Path("runs/identity.json")
    if ident.exists():
        S["identity"] = json.loads(ident.read_text())
    Path("runs/summary.json").write_text(json.dumps(S, indent=1) + "\n")
    for arm in ARMS:
        t = S["table"].get(arm)
        if t:
            print(f"{arm:9s} score {t['score']['median']:.2f} [{t['score']['q1']:.2f}-{t['score']['q3']:.2f}]"
                  f" agree {t['agree']['median']:.3f} length {t['length']['median']:.0f} n={t['n']} warned {t['warned']}")
    for k in ("pos32_vs_pos0", "pos0_vs_norecall", "pos32_vs_norecall", "mlp2_vs_mlp1"):
        p = S[k]; print(k, "wins", p["wins"], "losses", p["losses"], "median diff", round(p["median_diff"], 2), "p", round(p["wilcoxon_p"], 4))
    if "recency0" in S:
        for arm, t in S["recency0"]["table"].items():
            print("rec0", arm, round(t["score"]["median"], 2), round(t["agree"]["median"], 3))
        p = S["recency0"]["pos32_vs_pos0"]
        print("rec0 pos32_vs_pos0 wins", p["wins"], "losses", p["losses"], "median diff", round(p["median_diff"], 2), "p", round(p["wilcoxon_p"], 4))


if __name__ == "__main__":
    main()
