"""Morning step for the cold-start night (COLDSTART.md). Run AFTER all six runs have logged 'deadline' (08:00):
    ../../.venv-concepts/Scripts/python.exe morning.py            (all steps)
    ... morning.py --steps curve                                    (just the pre-registered metric, seconds)

Steps (results in runs/morning/):
  curve  Pre-registered primary metric from each run's curve.csv: common lesson range Kc = the smallest last chunk
         over the 6 runs; AUC = mean SEL mean-x (12 levels) over chunks 0..Kc (trapezoid / Kc); END = SEL mean x at Kc.
         Arm = mean of seeds A, B. Noise floor = the largest within-arm |A - B| (over the 3 arms), per metric.
         Verdict per memory arm (Cicero, Livy): HELPS only if A-vs-CatoA AND B-vs-CatoB both exceed the floor, on AUC
         AND on END; HURTS by the mirror rule; otherwise NOT DISTINGUISHABLE.            -> curve.json
  held   Fixed checkpoints, scored greedy (act mode, cold settles) on HELD 50 for all 16 levels: ck Kc (equal
         lessons, the primary checkpoint) and each run's last checkpoint. World 4 = first sight (never coached or
         trained); 4-4 (maze castle) reported separately.                                  -> held.json
  gaunt  5-life gauntlets on each run's last checkpoint, greedy AND sampled (temperature 0.05), from 1-1 through
         4-4 (through World 2 fully, then World 3, then World 4 as the test), plus world-start gauntlets from 2-1, 3-1
         and 4-1 (secondary, so every world gets played). Every attempt clipped (.mp4).      -> gauntlet/*.json/mp4
  report morning.md table from the above.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memory"))
import cs  # noqa: E402

RUNS = [f"{a}{s}" for a in ("Cato", "Cicero", "Livy") for s in ("A", "B")]
OUT = HERE / "runs" / "morning"
PY = sys.executable


def read_curve(name):
    rows = (HERE / "runs" / name / "curve.csv").read_text().strip().splitlines()
    hdr = rows[0].split(","); data = [dict(zip(hdr, r.split(","))) for r in rows[1:]]
    return {int(d["chunk"]): d for d in data}, hdr


def step_curve():
    curves = {n: read_curve(n)[0] for n in RUNS}
    kc = min(max(c) for c in curves.values())
    m = {}
    for n, c in curves.items():
        ys = np.array([float(c[k]["sel_mean_all"]) for k in range(kc + 1)])
        auc = float((getattr(np, "trapezoid", None) or np.trapz)(ys) / kc) if kc else float(ys[0])
        m[n] = dict(auc=round(auc, 1), end=float(c[kc]["sel_mean_all"]), last_chunk=max(c), flags_end=int(c[kc]["sel_flags_all"]),
                    per_level_end={k[1:]: int(v) for k, v in c[kc].items() if k.startswith("m") and "-" in k})
    arms = {}
    for a in ("Cato", "Cicero", "Livy"):
        arms[a] = {met: dict(A=m[a + "A"][met], B=m[a + "B"][met], mean=round((m[a + "A"][met] + m[a + "B"][met]) / 2, 1),
                             gap=round(abs(m[a + "A"][met] - m[a + "B"][met]), 1)) for met in ("auc", "end")}
    floor = {met: max(arms[a][met]["gap"] for a in arms) for met in ("auc", "end")}
    verdict = {}
    for a in ("Cicero", "Livy"):
        v = {}
        for met in ("auc", "end"):
            dA = m[a + "A"][met] - m["CatoA"][met]; dB = m[a + "B"][met] - m["CatoB"][met]
            v[met] = dict(dA=round(dA, 1), dB=round(dB, 1), floor=floor[met],
                          sign=1 if min(dA, dB) > floor[met] else (-1 if max(dA, dB) < -floor[met] else 0))
        s = {v["auc"]["sign"], v["end"]["sign"]}
        v["verdict"] = "HELPS" if s == {1} else ("HURTS" if s == {-1} else "NOT DISTINGUISHABLE")
        verdict[a] = v
    rep = dict(kc=kc, lessons_kc=kc * 500, runs=m, arms=arms, floor=floor, verdict=verdict)
    (OUT / "curve.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(kc=kc, floor=floor, arms={a: {k: v["mean"] for k, v in arms[a].items()} for a in arms},
                          verdict={a: verdict[a]["verdict"] for a in verdict})), flush=True)
    return rep


def ckpts(kc):
    out = {}
    for n in RUNS:
        last = int(max(read_curve(n)[0]))
        for tag, k in (("kc", kc), ("last", last)):
            p = HERE / "runs" / n / f"{n}_ck{k:04d}.npz"
            if p.exists():
                out.setdefault(n, {})[tag] = (k, str(p))
    return out


def step_held(kc, workers):
    import evalm
    cks = ckpts(kc); brains = {}
    for n, d in cks.items():
        brains[f"{n}@ck{d['kc'][0]}"] = (d["kc"][1], "act")
        if d["last"][0] != d["kc"][0]:
            brains[f"{n}@ck{d['last'][0]}"] = (d["last"][1], "act")
    t = time.time()
    summ, rep = evalm.run(brains, cs.ALL, ["held"], workers)
    tab = {}
    for k, row in summ.items():
        n, L = k.split("|"); tab.setdefault(n, {})[L] = row["held"]
    for n, d in tab.items():
        d["train12"] = dict(flags=sum(d[L]["flags"] for L in cs.TRAIN), mean=round(float(np.mean([d[L]["mean"] for L in cs.TRAIN]))))
        d["w4_first_sight_3"] = dict(flags=sum(d[L]["flags"] for L in cs.TEST[:3]), mean=round(float(np.mean([d[L]["mean"] for L in cs.TEST[:3]]))))
        d["4-4_maze"] = d["4-4"]
    (OUT / "held.json").write_text(json.dumps(dict(table=tab, seconds=round(time.time() - t))))
    for n, d in tab.items():
        print(n, "train12", d["train12"], "W4(4-1..4-3)", d["w4_first_sight_3"], "4-4", d["4-4"], flush=True)
    return tab


def step_gaunt(workers):
    g = OUT / "gauntlet"; g.mkdir(parents=True, exist_ok=True)
    jobs = []
    for n in RUNS:
        last = int(max(read_curve(n)[0])); p = HERE / "runs" / n / f"{n}_ck{last:04d}.npz"
        for mode in ("greedy", "sampled"):
            for first in ("1-1", "2-1", "3-1", "4-1"):
                tag = f"{n}_ck{last}_{mode}_from{first}"
                jobs.append([PY, str(HERE / "cgauntlet.py"), "--name", f"{n} ck{last}", "--brain", str(p), "--mode", mode,
                             "--first", first, "--last", "4-4", "--out", str(g / tag)])
    running = []
    while jobs or running:
        while jobs and len(running) < workers:
            j = jobs.pop(0); running.append(subprocess.Popen(j, stdout=open(str(j[-1]) + ".log", "w"), stderr=subprocess.STDOUT))
        time.sleep(5); running = [r for r in running if r.poll() is None]
    res = {}
    for f in sorted(g.glob("*.json")):
        d = json.loads(f.read_text()); res[f.stem] = dict(cleared=d["cleared"], deaths=d["deaths"], furthest=d["furthest"], clip=d["clip"])
    (OUT / "gauntlets.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, "cleared", v["cleared"], "furthest", v["furthest"], flush=True)
    return res


def step_report():
    c = json.loads((OUT / "curve.json").read_text())
    L = [f"# Cold-start morning ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"Common lesson range: chunks 0..{c['kc']} ({c['lessons_kc']} lesson batches of 32 rows each).", "",
         "| run | AUC (mean SEL x, 12 levels) | END (at Kc) | SEL flags at Kc | last chunk |", "|---|---|---|---|---|"]
    for n, r in c["runs"].items():
        L.append(f"| {n} | {r['auc']} | {r['end']} | {r['flags_end']} | {r['last_chunk']} |")
    L += ["", f"Noise floor (largest within-arm seed gap): AUC {c['floor']['auc']}, END {c['floor']['end']}", ""]
    for a, v in c["verdict"].items():
        L.append(f"- {a} vs Cato: AUC dA {v['auc']['dA']} dB {v['auc']['dB']}; END dA {v['end']['dA']} dB {v['end']['dB']} -> **{v['verdict']}**")
    if (OUT / "held.json").exists():
        t = json.loads((OUT / "held.json").read_text())["table"]
        L += ["", "| checkpoint | 12 training levels HELD flags / mean x | World 4 first sight 4-1..4-3 flags / mean x | 4-4 maze mean x |", "|---|---|---|---|"]
        for n, d in t.items():
            L.append(f"| {n} | {d['train12']['flags']}/600, {d['train12']['mean']} | {d['w4_first_sight_3']['flags']}/150, {d['w4_first_sight_3']['mean']} | {d['4-4']['mean']} |")
    if (OUT / "gauntlets.json").exists():
        gj = json.loads((OUT / "gauntlets.json").read_text())
        L += ["", "| gauntlet | cleared | furthest |", "|---|---|---|"]
        for k, v in gj.items():
            L.append(f"| {k} | {', '.join(v['cleared']) or '-'} | {v['furthest']} |")
    (OUT / "morning.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT / "morning.md")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", default="curve,held,gaunt,report")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for n in RUNS:
        log = (HERE / "runs" / n / "log.txt").read_text(encoding="utf-8")
        if "deadline" not in log and "done." not in log:
            print(f"WARNING: {n} has not logged 'deadline' yet (still running?)", flush=True)
    steps = a.steps.split(",")
    rep = step_curve()
    if "held" in steps:
        step_held(rep["kc"], a.workers)
    if "gaunt" in steps:
        step_gaunt(a.workers)
    step_report()


if __name__ == "__main__":
    main()
