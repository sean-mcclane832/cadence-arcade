"""Question 2: on the brain's own path, was its button as good as the coach's best (within 2 px), near the end vs earlier?"""
import json, sys, numpy as np
BTN = (0, 3, 4, 6); DEAD = -1e5
for tag in sys.argv[1:]:
    res = json.load(open(f"scratch/play/label_{tag}.json"))
    drift = sum(r["replay_max_x"] != r["max_x"] for r in res)
    def stat(rows):
        ok, alive, hold, fix_change, n = 0, 0, 0, 0, 0
        for row in rows:
            sc = np.array(row["sc"]); best = sc.max(); a = row["a"]; n += 1
            good = a in BTN and sc[BTN.index(a)] >= best - 2
            ok += good; hold += (a == row["prev"])
            if not good: fix_change += (BTN[int(sc.argmax())] != row["prev"])
        return dict(n=n, as_good=round(ok / max(n, 1), 3), repeats_prev=round(hold / max(n, 1), 3), wrong_and_best_needed_a_change=round(fix_change / max(n - ok, 1), 3))
    for end in ("died", "stuck"):
        rs = [r for r in res if r["end"] == end]
        print(f"== {tag} {end}: attempts {len(rs)} (replay drift in {drift} of {len(res)})")
        for name, lo, hi in (("last 4 decisions", 0, 3), ("5-16 before the end", 4, 15), ("40-47 before the end", 40, 47)):
            print("  ", name, stat([row for r in rs for row in r["rows"] if lo <= row["back"] <= hi]))
    # score scale check
    allsc = np.array([row["sc"] for r in res for row in r["rows"]]); print("  score range", float(allsc.min()), float(np.median(allsc)), float(allsc.max()))
