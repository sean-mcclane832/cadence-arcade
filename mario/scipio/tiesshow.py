import json, numpy as np
res = json.load(open("scratch/play/ties.json")); BTN = (0, 3, 4, 6)
ch = rp = 0; ch_tie = ch_lab_ok = rp_other = rp_lab_ok = alld = 0; per = {}
for r in res:
    c = t = 0
    for row in r["rows"]:
        sc = np.array(row["sc"]); best = sc.max(); y, prev = row["y"], row["prev"]
        if y not in BTN: continue
        lab_ok = sc[BTN.index(y)] >= best - 2
        if y != prev:
            ch += 1; c += 1; ch_lab_ok += lab_ok
            if prev in BTN and sc[BTN.index(prev)] >= best - 2: ch_tie += 1; t += 1
        else:
            rp += 1; rp_lab_ok += lab_ok; rp_other += (np.sum(sc >= best - 2) > 1)
    per[r["level"]] = (c, t, r["flag"], r["s"])
print("replays reaching flag", sum(r["flag"] for r in res), "of", len(res))
print(f"change frames {ch}: held button within 2 px of best (tie) {ch_tie} = {ch_tie/ch:.3f}; coach label within 2 px of best {ch_lab_ok/ch:.3f}")
print(f"repeat frames sampled {rp}: another button also within 2 px {rp_other/rp:.3f}; label within 2 px of best {rp_lab_ok/rp:.3f}")
print({L: v[:2] for L, v in per.items()}); print("secs", [v[3] for v in per.values()])
print("no flag:", [(r["level"], r["max_x"], r["n"]) for r in res if not r["flag"]])
for ok in (True, False):
    c = t = 0
    for r in res:
        if r["flag"] != ok: continue
        for row in r["rows"]:
            sc = np.array(row["sc"]); y, prev = row["y"], row["prev"]
            if y in BTN and y != prev:
                c += 1; t += prev in BTN and sc[BTN.index(prev)] >= sc.max() - 2
    print("flag replays" if ok else "non-flag replays", c, round(t / max(c, 1), 3))
