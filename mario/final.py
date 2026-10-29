"""After training: pick the showcase round, run assisted attempts, record replays.

The brain is one continuing brain; its state after each round was snapshotted (brain_eNN.npz).
The showcase is the round with the best median eval x (ties: the later round), i.e. early
stopping on the eval, and the page says which round it is.

    python final.py --run runs/main --out ../docs/mario
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="../docs/mario")
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()
    run = HERE / args.run
    hist = json.loads((run / "history.json").read_text())
    rows = []
    for h in hist:
        if "eval" in h and not h.get("eval_stale") and (run / f"brain_e{h['round']:02d}.npz").exists():
            xs = [e["max_x"] for e in h["eval"]]
            rows.append((float(np.median(xs)), h["round"], xs))
    for r in rows:
        print("round", r[1], "median", r[0], r[2])
    pick = args.round if args.round is not None else max(rows, key=lambda r: (r[0], r[1]))[1]
    print("showcase round", pick)
    brain = run / f"brain_e{pick:02d}.npz"
    env = dict(os.environ, NUMBA_NUM_THREADS="1")
    procs = []
    for s in (0, 5, 11, 17, 23):
        d = run / f"assist_final"
        d.mkdir(exist_ok=True)
        (d / "brain.npz").write_bytes(brain.read_bytes()) if s == 0 else None
    for s in (0, 5, 11, 17, 23):
        procs.append(subprocess.Popen([PY, str(HERE / "assist.py"), "--run", f"{args.run}/assist_final",
                                       "--start", str(s), "--out", str(run / "assist_final" / f"s{s}.json")],
                                      env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    for p in procs:
        out, _ = p.communicate()
        print(out.strip().splitlines()[-1] if out.strip() else "assist failed")
    res = [json.loads((run / "assist_final" / f"s{s}.json").read_text()) for s in (0, 5, 11, 17, 23)
           if (run / "assist_final" / f"s{s}.json").exists()]
    summary = dict(showcase_round=pick, rounds=[dict(round=r[1], median=r[0], xs=r[2]) for r in rows],
                   assist=[dict(start=r["start"], flag=r["flag"], brain_share=r["brain_share"],
                                rescues=len(r["rescues"]), decisions=r["decisions"]) for r in res])
    (run / "final_summary.json").write_text(json.dumps(summary, indent=1))
    best = max(res, key=lambda r: (r["flag"], r["brain_share"]))
    best_path = run / "assist_final" / f"s{best['start']}.json"
    subprocess.run([PY, str(HERE / "record.py"), "--run", args.run, "--out", args.out, "--round",
                    str([i for i, h in enumerate(hist) if h["round"] == pick and "eval" in h][-1]),
                    "--name", "replay"], cwd=HERE, check=True)
    subprocess.run([PY, str(HERE / "record.py"), "--run", args.run, "--out", args.out,
                    "--assist", str(best_path.relative_to(HERE)), "--name", "assist"], cwd=HERE, check=True)
    print(json.dumps(summary["assist"]))


if __name__ == "__main__":
    main()
