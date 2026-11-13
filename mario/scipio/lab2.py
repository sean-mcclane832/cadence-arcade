import json, numpy as np
BTN = (0, 3, 4, 6)
for tag in ("Cato2A", "Cato2B", "FabiusA"):
    rows = [row for r in json.load(open(f"scratch/play/label_{tag}.json")) if r["end"] == "died" for row in r["rows"] if 4 <= row["back"] <= 15]
    w = [row for row in rows if not (row["a"] in BTN and row["sc"][BTN.index(row["a"])] >= max(row["sc"]) - 2)]
    alive = [row for row in w if max(row["sc"]) > -5e4]
    print(tag, "wrong", len(w), "of", len(rows), "| wrong while repeating the held button", round(np.mean([row["a"] == row["prev"] for row in w]), 3),
          "| wrong with a live plan still on the menu", len(alive))
