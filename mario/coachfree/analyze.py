"""Score the pilot by the pre-registered rule (COACHFREE.md). python analyze.py [level ...] -> runs/pilot_<L>/score.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def hist(d):
    p = d / "history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def score(L):
    root = HERE / "runs" / f"pilot_{L}"; out = {}
    H = {a: hist(root / a) for a in ("b1", "anch", "rand", "reload", "b3")}
    for a, h in H.items():
        if not h:
            continue
        ev = [r for r in h if "held" in r]
        x0 = np.array(ev[0]["held"]["x"], float)
        last = [r for r in ev if r["round"] in (26, 28, 30)]
        o = dict(rounds=max(r["round"] for r in h), held0=float(x0.mean()),
                 held_curve=[(r["round"], r["held"]["mean"], r["held"]["flags"]) for r in ev])
        if len(last) == 3:
            dx = np.mean([np.array(r["held"]["x"], float) for r in last], axis=0) - x0
            o.update(D=round(float(dx.mean()), 1), split_half_gap=round(float(abs(dx[0::2].mean() - dx[1::2].mean())), 1),
                     held_last3=round(float(x0.mean() + dx.mean()), 1), per_start_up=int((dx > 0).sum()), per_start_down=int((dx < 0).sum()))
        pr = [r for r in h if "practice" in r]
        o["practice_curve"] = [r["practice"]["mean"] for r in pr]
        o["practice_flags"] = sum(r["practice"]["flags"] for r in pr)
        if len(pr) >= 30:
            o["practice_last10"] = round(float(np.mean([r["practice"]["mean"] for r in pr[20:30]])), 1)
            o["practice_first10"] = round(float(np.mean([r["practice"]["mean"] for r in pr[:10]])), 1)
            o["practice_halfgap_21_25_vs_26_30"] = round(abs(float(np.mean([r["practice"]["mean"] for r in pr[20:25]]))
                                                         - float(np.mean([r["practice"]["mean"] for r in pr[25:30]]))), 1)
        fg = root / a / "forget.json"
        if fg.exists():
            o["forget_sel_mean"] = round(json.loads(fg.read_text())["mean"], 1)
        out[a] = o
    verdict = {}
    if "anch" in out and "D" in out["anch"]:
        for a in ("b1", "rand", "b3"):
            if a in out and "D" in out[a]:
                floor = max(abs(out["anch"]["D"]), out[a]["split_half_gap"])
                diff = out[a]["D"] - out["anch"]["D"]
                verdict[a] = dict(diff_vs_anch=round(diff, 1), floor=round(floor, 1),
                                  result="MOVED (up)" if diff > floor else ("WORSE" if diff < -floor else "not distinguishable"))
    if "reload" in out and "practice_last10" in out["reload"]:
        for a in ("b1", "rand", "b3", "anch"):
            if a in out and "practice_last10" in out[a]:
                verdict.setdefault(a, {})["practice_vs_reload"] = round(out[a]["practice_last10"] - out["reload"]["practice_last10"], 1)
                verdict[a]["practice_floor"] = out["reload"]["practice_halfgap_21_25_vs_26_30"]
    out["verdict"] = verdict
    (root / "score.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for L in sys.argv[1:] or ["6-1", "6-2"]:
        s = score(L)
        print(L, json.dumps({a: {k: v for k, v in o.items() if not k.endswith("curve")} for a, o in s.items()}, indent=1))
