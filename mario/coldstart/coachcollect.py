"""Coach demonstrations for the cold-start night, World 1's recipe (concepts/collect.py = dagger.py round 0) on any
level: 6 clean coach runs (starts 0, 3, 8, 13, 20, 27) + 3 with 12% random buttons + 2 with 25%. Body senses (2,000).
Every decision is labelled, so a level the coach cannot clear still gives labels up to where it reaches.

Per level writes <out>/<L>/: data.npz (pile: X, y, W), coach_paths.json (clean flag paths), seq_<L>_coach.npz
(the same 11 attempts in order, every row labelled; mtrain's ordered-row format), coverage.json (clear rate of the
clean runs, max x / end of every run). --horizon L=H,... gives the coach a deeper plan menu on those levels.
--clean-only runs just the 6 clean runs (horizon probes).
    python coachcollect.py --levels 1-1,...,3-4 --out data/h16 --workers 8
--senses fabius stores Fabius's eyes (3,536, fabius/eyes.py); --coach smart labels with ../coach's smart coach (plug.py).
    python coachcollect.py --levels 1-1,...,6-4 --out data/fab --senses fabius --coach smart --stuck 225 --workers 6
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402
import seqcollect  # noqa: E402  (memory/: save_attempts)

C = cs.C
STUCK_USED = [0]
META = {}  # senses / coach, written into coverage.json
RECIPE = [(s, 0.0, s) for s in (0, 3, 8, 13, 20, 27)] + [(s, 0.12, 100 + s) for s in (1, 15, 24)] + [(s, 0.25, 200 + s) for s in (10, 22)]


def run(task):
    if task.get("stuck"):  # coach demos only: let the coach wait longer than 40 decisions (piranha cycles)
        C.D.STUCK = int(task["stuck"])
    t = time.time(); r = C.episode(task); r["secs"] = round(time.time() - t); return r


def write_level(out, L, rs, horizon):
    d = out / L; d.mkdir(parents=True, exist_ok=True)
    X = np.concatenate([r["obs"] for _, r in rs]); y = np.concatenate([r["labels"] for _, r in rs])
    W = np.concatenate([r["weights"] for _, r in rs])
    np.savez(d / "data.npz", X=X, y=y, W=W)
    paths = [(k[1], r["actions"]) for k, r in rs if r["flag"] and k[2] == 0.0]
    (d / "coach_paths.json").write_text(json.dumps(paths))
    atts = [dict(obs=r["obs"], actions=r["actions"], lab_idx=np.arange(len(r["labels"])), labels=r["labels"],
                 weights=r["weights"]) for _, r in rs]
    seqcollect.save_attempts(d / f"seq_{L}_coach.npz", atts)
    runs = [dict(start=k[1], eps=k[2], flag=bool(r["flag"]), end=r["end"], max_x=int(r["max_x"]), decisions=int(r["decisions"]),
                 secs=r["secs"]) for k, r in rs]
    clean = [x for x in runs if x["eps"] == 0.0]
    cov = dict(level=L, horizon=horizon, stuck=STUCK_USED[0] or C.D.STUCK, **META, clean_clears=sum(x["flag"] for x in clean), clean_runs=len(clean),
               all_clears=sum(x["flag"] for x in runs), rows=int(len(y)), critical=int((W > 1).sum()),
               clean_max_x=[x["max_x"] for x in clean], clean_ends=[x["end"] for x in clean], runs=runs)
    (d / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(f"{time.strftime('%H:%M:%S')} {L} h{horizon}: clean {cov['clean_clears']}/{len(clean)} max x {cov['clean_max_x']} "
          f"ends {''.join(e[0] for e in cov['clean_ends'])} | all {cov['all_clears']}/{len(runs)} | rows {cov['rows']} "
          f"crit {cov['critical']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--horizon", default="")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--stuck", type=int, default=0, help="stuck limit for the coach runs (default dagger.STUCK)")
    ap.add_argument("--senses", default="body", help="dagger.Body senses mode of the stored obs (body: 2,000; fabius: 3,536)")
    ap.add_argument("--coach", default="base", help="base (game.py) or a ../coach smartcoach variant, e.g. smart")
    ap.add_argument("--only", default="", help="clean runs from these starts only, e.g. 0 (smoke tests)")
    a = ap.parse_args(); STUCK_USED[0] = a.stuck; META.update(senses=a.senses, coach=a.coach)
    out = (HERE / a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    deep = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.horizon.split(",") if kv}
    recipe = [r for r in RECIPE if r[1] == 0.0] if a.clean_only else RECIPE
    if a.only:
        recipe = [r for r in RECIPE if r[1] == 0.0 and r[0] in {int(v) for v in a.only.split(",")}]
    levels = a.levels.split(",")
    tasks, keys = [], []
    for s, eps, seed in recipe:  # interleave levels so every level finishes at about the same time
        for L in levels:
            tasks.append(C.task("none", a.senses, L, s, mode="coach", eps=eps, seed=seed, stuck=a.stuck)); keys.append((L, s, eps))
    t0 = time.time(); res = {}
    if a.coach == "base":
        init, initargs = cs.worker_init, (deep,)
    else:
        sys.path.insert(0, str(HERE.parent / "coach"))
        import plug  # noqa: E402  (refuses World 8)
        init, initargs = plug.worker_init, (deep, a.coach, {})
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=init, initargs=initargs) as ex:
        futs = {ex.submit(run, tk): i for i, tk in enumerate(tasks)}
        for f in as_completed(futs):
            i = futs[f]; res[i] = f.result(); L = keys[i][0]
            mine = [j for j, k in enumerate(keys) if k[0] == L]
            if all(j in res for j in mine):
                write_level(out, L, [(keys[j], res[j]) for j in mine], deep.get(L, 16))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
