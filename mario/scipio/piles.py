"""Leg 1: what the coach piles hold (data/fab, read-only). Label mix per level, critical share, how the inputs
are distributed (always-on inputs, dead inputs). numpy only. Writes scratch/piles.json."""
import json
from pathlib import Path
import numpy as np
FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
HERE = Path(__file__).resolve().parent
ACTIONS = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
GROUPS = dict(periph_now=(0, 832), periph_prev=(832, 1664), body=(1664, 2000), fov_R=(2000, 2384), fov_G=(2384, 2768),
              fov_B=(2768, 3152), fov_prev=(3152, 3536))
levels = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
out = {"levels": {}}
tot = np.zeros(7); totc = np.zeros(7); s1 = np.zeros(3536); s2 = np.zeros(3536); n = 0; on = np.zeros(3536)
rows_all = 0
for L in levels:
    z = np.load(FAB / L / "data.npz")
    y, W = z["y"], z["W"]
    X = z["X"].astype(np.float32) / 255.0
    h = np.bincount(y, minlength=7); hc = np.bincount(y[W > 1], minlength=7)
    tot += h; totc += hc
    s1 += X.sum(0); s2 += (X ** 2).sum(0); on += (X > 0.02).sum(0); n += len(X)
    cov = json.loads((FAB / L / "coverage.json").read_text())
    out["levels"][L] = dict(rows=int(len(y)), critical_share=round(float((W > 1).mean()), 3), labels={ACTIONS[i]: int(h[i]) for i in range(7)},
                            clean_clears=cov.get("clean_clears"), clean_runs=cov.get("clean_runs"),
                            clean_decisions=[r["decisions"] for r in cov["runs"] if r["eps"] == 0.0],
                            weights=sorted(set(np.round(W, 2).tolist())))
    del X
mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0)); onshare = on / n
out["rows_total"] = int(n)
out["labels_total"] = {ACTIONS[i]: int(tot[i]) for i in range(7)}
out["labels_share"] = {ACTIONS[i]: round(float(tot[i] / tot.sum()), 4) for i in range(7)}
out["labels_critical_share"] = {ACTIONS[i]: round(float(totc[i] / max(1, totc.sum())), 4) for i in range(7)}
out["inputs"] = {}
for g, (a, b) in GROUPS.items():
    m, s, o = mean[a:b], std[a:b], onshare[a:b]
    out["inputs"][g] = dict(n=b - a, mean=round(float(m.mean()), 3), std=round(float(s.mean()), 3), on_share=round(float(o.mean()), 3),
                            dead=int((s < 1e-4).sum()), always_on=int(((m > 0.5) & (s < 0.15)).sum()),
                            mean_drive_sum=round(float(m.sum()), 1), varying_drive=round(float(s.sum()), 1))
out["inputs"]["all"] = dict(mean_drive_sum=round(float(mean.sum()), 1), std_sum=round(float(std.sum()), 1), dead=int((std < 1e-4).sum()))
body = mean[1664:1706]
out["body_mean_first_copy"] = [round(float(x), 2) for x in body]
(HERE / "scratch" / "piles.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(HERE / "scratch" / "pile_input_stats.npz", mean=mean, std=std, on=onshare)
print(json.dumps({k: out[k] for k in out if k != "levels"}, indent=0))
for L in levels:
    d = out["levels"][L]; print(L, d["rows"], d["critical_share"], d["labels"], d["clean_clears"], d["clean_decisions"][:3])
