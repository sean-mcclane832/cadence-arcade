"""Leg 1, questions 3 (dilution) and 5 (time budget). Text files only: history.jsonl, log.txt of the runs.
Reads P:/projects/cadence-arcade/mario/coldstart/runs (read-only). Writes scratch/budget.json and prints a summary."""
import json, re, sys, statistics as st
from pathlib import Path
import numpy as np
RUNS = Path("P:/projects/cadence-arcade/mario/coldstart/runs")
OUT = Path(__file__).resolve().parent / "scratch" / "budget.json"
SEL8_IN_24 = None
W13 = [f"{w}-{l}" for w in (1, 2, 3) for l in (1, 2, 3, 4)]
ALL24 = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]

def hist(name):
    rows = []
    for ln in (RUNS / name / "history.jsonl").read_text().splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows

def dagger_secs(name):
    out = []
    for ln in (RUNS / name / "log.txt").read_text(errors="replace").splitlines():
        m = re.search(r"dagger \((\d+)s\): (.*)", ln)
        if m:
            labs = [int(x) for x in re.findall(r"\+(\d+) x", m.group(2))]
            out.append((int(m.group(1)), sum(labs), len(labs)))
    return out

def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 1) if v else None

res = {}
for name in ("Cato2A", "Cato2B", "FabiusA", "FabiusB", "CatoA", "CatoB"):
    h = hist(name); d = dagger_secs(name)
    ch = [r for r in h if r["chunk"] > 0]
    teach = [r.get("teach_s") for r in ch]; sel = [r.get("sel_s") for r in ch]
    last = ch[-20:]
    r = dict(chunks=h[-1]["chunk"], teach_med=med(teach), sel_med=med(sel),
             teach_med_last20=med([x.get("teach_s") for x in last]), sel_med_last20=med([x.get("sel_s") for x in last]),
             teach_sum=sum(x for x in teach if x), sel_sum=sum(x for x in sel if x),
             dagger_rounds=len(d), dagger_sum=sum(x[0] for x in d), dagger_med=med([x[0] for x in d]),
             dagger_labels_med=med([x[1] for x in d]), dagger_s_per_label=round(sum(x[0] for x in d) / max(1, sum(x[1] for x in d)), 2),
             dagger_last5=[x[0] for x in d[-5:]], fit_last=ch[-1].get("fit"), sel_n=ch[-1].get("sel_n"))
    tot = r["teach_sum"] + r["sel_sum"] + r["dagger_sum"]
    r["share"] = {k: round(r[k + "_sum"] / tot, 3) for k in ("teach", "sel", "dagger")}
    r["work_hours"] = round(tot / 3600, 1)
    r["s_per_chunk_all_in"] = round(tot / max(1, len(ch)))
    # SEL attempt length proxy: share of per-start x that equal the level's start x (stood still)
    lastrec = ch[-1]["solo"]
    xs = [x for L in lastrec for x in lastrec[L]["x"]]
    r["last_chunk_stand_still_share"] = round(float(np.mean([x <= 60 for x in xs])), 3)
    res[name] = r

# Q3: dilution. Worlds 1-3 mean per chunk: Cato2 (24 levels) vs plateau Cato (12 levels).
def w13_curve(name):
    out = {}
    for r in hist(name):
        s = r["solo"]
        if all(L in s for L in W13):
            vals = []
            for L in W13:
                x = s[L]["x"]
                vals.append(float(np.mean(x)))  # 24-start chunks: mean over 24; 8-start chunks: mean over 8
            out[r["chunk"]] = float(np.mean(vals))
    return out
def all_curve(name):
    return {r["chunk"]: float(np.mean([r["solo"][L]["mean"] for L in r["solo"]])) for r in hist(name)}
def win(c, a, b):
    v = [c[k] for k in range(a, b + 1) if k in c]
    return round(float(np.mean(v)), 1) if v else None
cur = {n: w13_curve(n) for n in ("Cato2A", "Cato2B", "FabiusA", "FabiusB", "CatoA", "CatoB")}
allc = {n: all_curve(n) for n in cur}
tab = {}
for a, b in ((1, 10), (11, 20), (21, 30), (31, 40), (36, 42), (41, 50), (51, 60), (61, 70), (71, 80), (74, 83)):
    tab[f"{a}-{b}"] = {n: win(cur[n], a, b) for n in cur}
res["w13_mean_by_chunk_window"] = tab
res["all_levels_mean_by_window"] = {f"{a}-{b}": {n: win(allc[n], a, b) for n in allc} for a, b in ((31, 40), (36, 42), (71, 80), (74, 83))}
# equal lessons PER LEVEL: Cato2 chunk 2k has the lessons per level of plateau chunk k
res["equal_lessons_per_level"] = {
    "Cato2 chunks 61-80 (w1-3 mean)": {n: win(cur[n], 61, 80) for n in ("Cato2A", "Cato2B")},
    "plateau Cato chunks 31-40 (same lessons per level)": {n: win(cur[n], 31, 40) for n in ("CatoA", "CatoB")},
    "plateau Cato chunks 61-80 (same chunks)": {n: win(cur[n], 61, 80) for n in ("CatoA", "CatoB")},
    "Cato2 chunks 31-40": {n: win(cur[n], 31, 40) for n in ("Cato2A", "Cato2B")},
    "plateau Cato chunks 16-20 (same lessons per level as Cato2 31-40)": {n: win(cur[n], 16, 20) for n in ("CatoA", "CatoB")},
}
# fit by chunk
def fits(name):
    return {r["chunk"]: r["fit"]["all"] for r in hist(name) if r.get("fit")}
res["fit_all"] = {n: {str(k): fits(n).get(k) for k in (5, 10, 20, 40, 60, 80, 83)} for n in cur}
# per-level last-10 means for the 4 live runs
lv = {}
for n in ("Cato2A", "Cato2B", "FabiusA", "FabiusB"):
    h = [r for r in hist(n) if r["chunk"] > 0][-10:]
    lv[n] = {L: [round(float(np.mean([r["solo"][L]["mean"] for r in h]))), int(sum(r["solo"][L]["flags"] for r in h))] for L in ALL24}
    lv[n]["chunks"] = [h[0]["chunk"], h[-1]["chunk"]]
res["per_level_last10"] = lv
OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(res, indent=1))
for n in ("Cato2A", "Cato2B", "FabiusA", "FabiusB", "CatoA", "CatoB"):
    print(n, json.dumps(res[n]))
print(json.dumps(res["w13_mean_by_chunk_window"], indent=0))
print(json.dumps(res["all_levels_mean_by_window"]))
print(json.dumps(res["equal_lessons_per_level"], indent=0))
print(json.dumps(res["fit_all"]))
for n in lv:
    print(n, lv[n])
