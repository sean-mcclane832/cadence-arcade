"""End-of-night pick (11/2): top-3 checkpoints per arm by mtrain's joint SEL key, pooled over leg 1 + r2.
Chunk 0 is skipped: leg-1 ck0 is the shared start W1_F2ck110 (identical in every arm, scored once as "Start"), r2 ck0 = leg-1 last. Ties: earlier (leg, chunk) first.
    python eon_select.py Cato Cicero Livy AgrippaX  -> runs/eon/picks.json (merged)"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
N2 = HERE / "runs" / "night2"


def picks(arm, n=3):
    rows = []
    for leg, d in ((1, N2 / arm), (2, N2 / f"{arm}_r2")):
        if not (d / "history.jsonl").exists():
            continue
        for line in open(d / "history.jsonl"):
            r = json.loads(line)
            if r["chunk"] == 0:  # leg-1 ck0 = W1_F2ck110 for every arm (scored once as the shared start); r2 ck0 = leg-1 last
                continue
            rows.append(dict(arm=arm, leg=leg, chunk=r["chunk"], key=r["key"],
                             sel=[r["solo"][L]["flags"] for L in ("1-1", "1-2", "1-3", "1-4")],
                             path=str((d / f"{arm}_ck{r['chunk']:04d}.npz").relative_to(HERE)).replace("\\", "/")))
    rows.sort(key=lambda r: (tuple(-v for v in r["key"]), r["leg"], r["chunk"]))
    return rows[:n], len(rows)


if __name__ == "__main__":
    f = HERE / "runs" / "eon" / "picks.json"
    allp = json.loads(f.read_text()) if f.exists() else {}
    for arm in sys.argv[1:]:
        top, n = picks(arm)
        allp[arm] = dict(n_candidates=n, top=top)
        print(arm, f"of {n}:", [(t["leg"], t["chunk"], t["key"], t["sel"]) for t in top])
    f.write_text(json.dumps(allp, indent=1))
