"""Build docs/mario/index.html from runs/main/history.json, baselines and replay.json.

    python build_page.py --run runs/main --baselines runs/baselines.json --out ../docs/mario
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAG_X = 3161


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--baselines", default="runs/baselines.json")
    ap.add_argument("--out", default="../docs/mario")
    args = ap.parse_args()
    text = json.loads((HERE / "page_text.json").read_text(encoding="utf-8"))
    run = HERE / args.run
    hist = json.loads((run / "history.json").read_text())
    base = json.loads((HERE / args.baselines).read_text())
    out = (HERE / args.out).resolve()
    replays = []
    for name, label in (("assist", "full level, assisted"), ("replay", "brain alone")):
        f = out / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["sheets"] = [str(x) for x in r["sheets"]]
            r["label"] = label
            r["note"] = text.get(f"note_{name}", "")
            replays.append(r)
    import cadence

    # Only honest measurements: evals made before the stale-worker fix are replaced by fresh
    # re-measurements of the checkpoints saved at those rounds (evalckpt.py), others are dropped.
    curve = []
    ck = json.loads((run / "ckpt_eval.json").read_text()) if (run / "ckpt_eval.json").exists() else {}
    for name, v in ck.items():
        rnd = int(name.split("_r")[-1].split(".")[0])
        xs = [e["max_x"] for e in v["eval"]]
        curve.append(dict(round=rnd, xs=xs, flags=sum(e["flag"] for e in v["eval"]), src="checkpoint"))
    for h in hist:
        if "eval" not in h or h.get("eval_stale"):
            continue
        xs = [e["max_x"] for e in h["eval"]]
        curve.append(dict(round=h["round"], samples=h["samples"], xs=xs,
                          flags=sum(e["flag"] for e in h["eval"]), own=h.get("own_agree"), src="live"))
    curve.sort(key=lambda c: c["round"])
    final = hist[-1] if "eval" in hist[-1] else hist[-2]
    assert not final.get("eval_stale")
    rows = []

    def row(name, recs, note):
        xs = [r["max_x"] for r in recs]
        rows.append(dict(name=name, median=float(np.median(xs)), best=int(max(xs)),
                         flags=f"{sum(r['flag'] for r in recs)}/{len(recs)}", note=note))

    row("Cadence brain (this one)", final["eval"], f"after {final['round']} DAgger rounds")
    if "mlp" in base:
        row("plain MLP, same screens", base["mlp"], "backprop + Adam on the brain's final data set")
    row("random buttons", base["random"], "5 random streams per start")
    row("always run", base["run"], "hold right + B")
    row("run + jump rhythm", base["run+jump rhythm"], "jump 8, run 4, repeat")
    data = dict(curve=curve, rows=rows, replays=replays, flag_x=FLAG_X, version=cadence.__version__,
                samples=final["samples"], rounds=final["round"], text=text,
                mlp_info=base.get("mlp_info"))
    tpl = (HERE / "page_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*DATA*/null", json.dumps(data))
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote", out / "index.html", len(html))


if __name__ == "__main__":
    main()
