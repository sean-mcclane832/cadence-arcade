"""Scipio chunk-50 registered read (vault Plans/Scipio Design 2026-11-12, 'Registered 11/14 before launch').

Written and committed before any time-only ablation result was looked at. Reads scratch/ablate50/*.json
(run_ablate50.py) and the runs' history.jsonl, prints the rule branch it lands on. Rule 1(ii) already failed on the
history (vault Scipio Progress, 2026-11-16), so rule 1 is out and (i) picks rule 2 / rule 3 / split / makes worse.
    python read50.py [--out scratch/ablate50/read50.json]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "coldstart" / "runs"
AB = HERE / "scratch" / "ablate50"
CKS = (46, 48, 50)
FLOOR = 10            # max(|A-B| last-10 = 7.1, 10)
SHOCK = 0.05          # coach agreement drop under time-only that voids rule 1(i)
GUARD_HOLDS = True    # rule 4 on the history: A 807 / B 837 vs Twin 827, not below by 150
STOP = {"rule 2": "rule 2 (memory in use, not on stalls): continue to 100, read again at 100",
        "rule 3": "rule 3 (memory not used) on both seeds: stop all four runs at 50",
        "split": "split on (i): continue to 100, report 'split'",
        "makes worse": "memory used, makes stalls worse: continue to 100 only if rule 4 holds"
                       + (" (it holds)" if GUARD_HOLDS else " (it FAILS: stop)"),
        "no verdict": "shock check failed: rule 1(i) gets no verdict; report as a shock"}


def history(run):
    return {r["chunk"]: r for r in map(json.loads, open(RUNS / run / "history.jsonl")) if r.get("scored", True)}


def check_none(run, ck, rep, hist):
    """none must replay the history row: every attempt's max x and the stuck count."""
    solo = hist[ck]["solo"]
    att = rep["modes"]["none"]["attempts"]
    bad = 0
    for L in rep["levels"]:
        xs = [a["max_x"] for a in att if a["level"] == L]
        if xs != solo[L]["x"]:
            bad += 1
            print(f"  MISMATCH {run} ck{ck} {L}: none {xs} history {solo[L]['x']}")
    hs = sum(v["stuck"] for v in solo.values())
    ns = rep["modes"]["none"]["summary"]["stuck"]
    return bad == 0 and hs == ns, hs, ns


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(AB / "read50.json")); a = ap.parse_args()
    out, ok_all, deltas = {}, True, {}
    for run in ("ScipioA", "ScipioB", "ScipioFast"):
        hist = history(run); rows = []
        for ck in CKS:
            p = AB / f"{run}_ck{ck:04d}.json"
            if not p.exists():
                print(f"{run} ck{ck}: missing"); continue
            rep = json.loads(p.read_text())
            if "none" not in rep["modes"]:
                print(f"{run} ck{ck}: no none mode yet"); continue
            ok, hs, ns = check_none(run, ck, rep, hist)
            ok_all &= ok
            row = dict(ck=ck, integrity=ok, history_stuck=hs)
            for m, v in rep["modes"].items():
                s = v["summary"]; ag = v.get("coach_agreement")
                row[m] = dict(stuck=s["stuck"], nostart=s["stuck_nostart"], mid=s["stuck_mid"], died=s["died"],
                              flags=s["flags"], mean=s["mean"], agree=ag["all"] if ag else None)
            rows.append(row)
            line = f"{run} ck{ck} integrity {'OK' if ok else 'FAIL'} (history {hs}, none {ns})"
            for m in ("none", "time", "amp0"):
                if m in row:
                    r = row[m]
                    line += f" | {m} stuck {r['stuck']} (start {r['nostart']} mid {r['mid']}) died {r['died']} mean {r['mean']} agree {r['agree']}"
            print(line)
        done = [r for r in rows if "time" in r]
        d = dict(rows=rows)
        if len(done) == len(CKS):
            d["d_time"] = round(float(np.mean([r["time"]["stuck"] - r["none"]["stuck"] for r in done])), 2)
            d["d_time_nostart"] = round(float(np.mean([r["time"]["nostart"] - r["none"]["nostart"] for r in done])), 2)
            d["d_time_mid"] = round(float(np.mean([r["time"]["mid"] - r["none"]["mid"] for r in done])), 2)
            d["d_time_died"] = round(float(np.mean([r["time"]["died"] - r["none"]["died"] for r in done])), 2)
            if all("amp0" in r for r in done):
                d["d_amp0"] = round(float(np.mean([r["amp0"]["stuck"] - r["none"]["stuck"] for r in done])), 2)
            ag = [(r["none"]["agree"], r["time"]["agree"]) for r in done]
            if all(x is not None and y is not None for x, y in ag):
                d["agree_drop"] = round(float(np.mean([x - y for x, y in ag])), 4)
            deltas[run] = d
            print(f"{run}: mean over ck46/48/50 time-none stuck {d['d_time']:+} (nostart {d['d_time_nostart']:+}, mid "
                  f"{d['d_time_mid']:+}), died {d['d_time_died']:+}, amp0 {d.get('d_amp0')}, agreement drop {d.get('agree_drop')}")
        out[run] = d
    if not ok_all:
        print("\nINTEGRITY FAILED: none does not replay the history. The read stops here.")
        out["branch"] = "integrity failed"
    elif "ScipioA" in deltas and "ScipioB" in deltas:
        a_, b_ = deltas["ScipioA"], deltas["ScipioB"]
        shock = any((x.get("agree_drop") or 0) > SHOCK for x in (a_, b_))
        da, db = a_["d_time"], b_["d_time"]
        if shock:
            br = "no verdict"
        elif da > FLOOR and db > FLOOR:
            br = "rule 2"
        elif da < -FLOOR and db < -FLOOR:
            br = "makes worse"
        elif abs(da) <= FLOOR and abs(db) <= FLOOR:
            br = "rule 3"
        elif (da > FLOOR) != (db > FLOOR):
            br = "split"
        else:  # one seed below -10, the other within the floor: not a registered branch
            br = "unregistered (one seed lowers by more than 10, the other within the floor)"
        out["branch"] = br
        print(f"\nTime-only stuck rise: A {da:+}, B {db:+} (floor {FLOOR}); shock {'YES' if shock else 'no'}")
        print("BRANCH:", br, "->", STOP.get(br, "not registered: report as is, Sean decides"))
        print("Prediction (vault 6117a3b): <=10 on at least one seed, point +6 each; Fast nostart +8 or more under time.")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
