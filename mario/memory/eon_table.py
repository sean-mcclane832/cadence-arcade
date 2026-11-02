"""Summarize end-of-night evalm jsons: HELD clears per level, World 1 total, World 2 first sight
(max x, mean x, flags, share of 2-1 starts past x 723 = the review's primary metric).
    python eon_table.py runs/eon/held_*.json"""
import json, sys
import numpy as np
rows = {}
for f in sys.argv[1:]:
    d = json.load(open(f))
    for k, v in d["detail"].items():
        n, L = k.split("|"); rows.setdefault(n, {})[L] = v
out = {}
for n, lv in rows.items():
    r = {}
    for L, st in lv.items():
        xs = np.array([s["x"] for s in st.values()]); fl = sum(s["flag"] for s in st.values())
        r[L] = dict(flags=int(fl), n=len(xs), mean=int(xs.mean()), max=int(xs.max()), past723=float((xs > 723).mean()))
    r["W1"] = sum(r[L]["flags"] for L in ("1-1", "1-2", "1-3", "1-4") if L in r)
    out[n] = r
    w2 = " | ".join(f"{L} {r[L]['flags']}/50 max {r[L]['max']} mean {r[L]['mean']}" + (f" past723 {r[L]['past723']:.0%}" if L == "2-1" else "")
                    for L in ("2-1", "2-2") if L in r)
    print(f"{n:16s} " + " ".join(f"{r[L]['flags']:2d}" for L in ("1-1", "1-2", "1-3", "1-4") if L in r) + f"  W1 {r['W1']:3d}/200  {w2}")
json.dump(out, open("runs/eon/table.json", "w"), indent=1)
