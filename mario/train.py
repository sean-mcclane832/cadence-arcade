"""Train ONE Cadence brain to play Super Mario Bros. World 1-1 from the screen.

Recipe (the upstream teach guide's: imitation at teaching="normalized", then DAgger on the
brain's own path):

  round 0   the coach plays 1-1 from several start jitters, some with random wrong buttons
            (so it also shows how to recover). Every moment is a labelled screen.
  training  the brain learns from all labelled screens so far: 32-row lessons, shuffled,
            `brain.learner.step(brain.stimulus(x, memory=False), labels)`.
  round r   the brain plays (`brain.predict`, greedy, its own button is executed), from the
            start and from random points along the coach's path; the coach labels every screen
            the brain reached. Labels are added; the same brain keeps learning. Repeat.
  eval      after each round the brain plays alone from 5 fixed start jitters; we log how far
            it got and whether it reached the flag.

The brain is never reset or replaced: one founder (seed 0), checkpointed after every round.

    python train.py --out runs/main --hours 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EVAL_STARTS = (0, 5, 11, 17, 23)
MAX_DECISIONS = int(os.environ.get("MARIO_MAX_DECISIONS", 700))
STUCK = 40
MIN_LESSONS = int(os.environ.get("MARIO_MIN_LESSONS", 3000))
MAX_LESSONS = int(os.environ.get("MARIO_MAX_LESSONS", 5000))


def compose_brain(seed: int, hidden: int):
    from cadence import Brain
    return Brain.compose(N_IN_(), 7, modules=(hidden,), seed=seed, teaching="normalized",
                         learning_surprise=True)


def N_IN_():
    from game import N_IN
    return N_IN


# ----------------------------------------------------------------------------------- workers
_BRAIN = None
_BRAIN_PATH = None


def _worker_init():
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _brain(path):
    """The worker's copy of the brain, reloaded whenever the checkpoint file changes. (Until
    04:15 on 2026-10-29 this keyed on the path alone, so pool workers kept the first brain they
    loaded and every later eval/DAgger attempt played a stale brain; see PROGRESS.md.)"""
    global _BRAIN, _BRAIN_PATH
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key != _BRAIN_PATH:
        from cadence import Brain
        _BRAIN = Brain.load(path)
        _BRAIN_PATH = key
    return _BRAIN


def episode(task: dict) -> dict:
    """One attempt. task: mode ('coach'|'brain'|'eval'), start, prefix (coach actions to replay
    first), eps, seed, brain (checkpoint path), label (bool)."""
    from game import Game
    rng = np.random.default_rng(task.get("seed", 0))
    g = Game(noop_start=task["start"])
    for a in task.get("prefix", []):
        g.step(a)
        if g.done:
            break
    brain = _brain(task["brain"]) if task["mode"] != "coach" else None
    obs, labels, acts, weights = [], [], [], []
    label_on = task.get("label", True)
    t0 = time.time()
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        o = g.observation()
        lab, wt = g.coach_weighted() if label_on else (-1, 0.0)
        if task["mode"] == "coach":
            a = lab if rng.random() >= task.get("eps", 0.0) else int(rng.integers(7))
        else:
            a = int(np.asarray(brain.predict(o[None])).reshape(-1)[0])
        if label_on:
            obs.append(np.round(o * 255).astype(np.uint8))
            labels.append(lab)
            weights.append(wt)
        acts.append(a)
        g.step(a)
    out = dict(task={k: v for k, v in task.items() if k != "prefix"}, prefix_len=len(task.get("prefix", [])),
               max_x=g.max_x, flag=g.flag, died=g.done and not g.flag, decisions=g.decisions,
               actions=acts, seconds=time.time() - t0)
    if label_on:
        out["obs"] = np.array(obs, dtype=np.uint8).reshape(-1, N_IN_())
        out["labels"] = np.array(labels, dtype=np.int64)
        out["weights"] = np.array(weights, dtype=np.float64)
    g.close()
    return out


# ---------------------------------------------------------------------------------- training
def teach(brain, X, y, passes: float, rng, log, W=None) -> dict:
    """Shuffled 32-row lessons over the whole set, drawn in proportion to the coach's weights
    (critical moments x10, see Game.coach_weighted; uniform when W is None). (Up-weighting rare
    BUTTONS made the readout swing between whole classes in the first attempt.)"""
    n = len(y)
    w = None if W is None else W / W.sum()
    lessons = int(np.clip(np.ceil(passes * n / 32), MIN_LESSONS, MAX_LESSONS))
    t = time.time()
    agree = []
    for i in range(lessons):
        idx = rng.choice(n, 32, p=w)
        x = X[idx].astype(np.float64) / 255.0
        brain.learner.step(brain.stimulus(x, memory=False), y[idx])
        if i % 200 == 0:
            check = rng.choice(n, 256, replace=False) if n > 256 else np.arange(n)
            p = np.asarray(brain.predict(X[check].astype(np.float64) / 255.0)).reshape(-1)
            agree.append(float((p == y[check]).mean()))
            log(f"  lesson {i}/{lessons} agree(sample)={agree[-1]:.3f} {time.time() - t:.0f}s")
    return dict(lessons=lessons, seconds=time.time() - t, agree=agree)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra-teach", action="store_true", help="on resume, teach once more before the next eval")
    args = ap.parse_args()
    out = (HERE / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    os.environ["NUMBA_NUM_THREADS"] = str(args.workers)  # before cadence/numba import
    deadline = time.time() + args.hours * 3600
    rng = np.random.default_rng(args.seed + 1)
    import cadence
    from cadence import Brain
    log(f"cadence {cadence.__version__} hidden={args.hidden} seed={args.seed} workers={args.workers}")
    ctx = get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_worker_init)

    data_p = out / "data.npz"
    brain_p = out / "brain.npz"
    hist_p = out / "history.json"
    history = json.loads(hist_p.read_text()) if (args.resume and hist_p.exists()) else []
    coach_paths: dict[int, list[int]] = {}

    if args.resume and data_p.exists():
        d = np.load(data_p)
        X, y = d["X"], d["y"]
        W = d["W"] if "W" in d.files else np.ones(len(y))
        brain = Brain.load(brain_p)
        coach_paths = {int(k): v for k, v in json.loads((out / "coach_paths.json").read_text()).items()}
        rnd = history[-1]["round"]
        log(f"resumed: {len(y)} samples, round {rnd}")
        if args.extra_teach:
            rep = teach(brain, X, y, args.passes, rng, log, W)
            brain.save(brain_p)
            history[-1].setdefault("extra_teach", []).append(rep)
    else:
        # round 0: coach demonstrations
        tasks = [dict(mode="coach", start=s, eps=0.0, seed=s) for s in (0, 3, 8, 13, 20, 27)]
        tasks += [dict(mode="coach", start=s, eps=0.12, seed=100 + s) for s in (1, 6, 15, 24, 30, 2, 9, 18)]
        tasks += [dict(mode="coach", start=s, eps=0.25, seed=200 + s) for s in (4, 10, 16, 22, 28, 31)]
        t = time.time()
        res = pool.map(episode, tasks, chunksize=1)
        X = np.concatenate([r["obs"] for r in res]); y = np.concatenate([r["labels"] for r in res])
        W = np.concatenate([r["weights"] for r in res])
        for r in res:
            if r["task"]["eps"] == 0.0 and r["flag"]:
                coach_paths[r["task"]["start"]] = r["actions"]
        (out / "coach_paths.json").write_text(json.dumps(coach_paths))
        summ = [(r["task"]["start"], r["task"]["eps"], r["max_x"], r["flag"]) for r in res]
        log(f"round 0 coach: {len(y)} samples in {time.time() - t:.0f}s; runs {summ}")
        log(f"  label counts {np.bincount(y, minlength=7).tolist()}")
        np.savez(data_p, X=X, y=y, W=W)  # save before teaching, so a stop never loses coach labels
        os.environ["NUMBA_NUM_THREADS"] = str(args.workers)
        brain = compose_brain(args.seed, args.hidden)
        rep = teach(brain, X, y, args.passes, rng, log, W)
        brain.save(brain_p)
        rnd = 0
        history.append(dict(round=0, samples=int(len(y)), teach=rep))

    while True:
        # evaluate the brain alone
        ev = pool.map(episode, [dict(mode="eval", start=s, brain=str(brain_p), label=False)
                                for s in EVAL_STARTS], chunksize=1)
        xs = [r["max_x"] for r in ev]; flags = [r["flag"] for r in ev]
        log(f"round {rnd} EVAL max_x={xs} flags={sum(flags)}/{len(flags)} median={np.median(xs):.0f}")
        history[-1]["eval"] = [dict(start=r["task"]["start"], max_x=r["max_x"], flag=r["flag"],
                                    decisions=r["decisions"], actions=r["actions"]) for r in ev]
        hist_p.write_text(json.dumps(history))
        if time.time() > deadline:
            log("deadline reached"); break
        if all(flags) and len(history) >= 2 and all(e["flag"] for e in history[-2].get("eval", [])):
            log("flag on every eval start two rounds running; stopping"); break
        rnd += 1
        # DAgger: the brain plays, the coach labels what it saw
        tasks = [dict(mode="brain", start=int(s), brain=str(brain_p), seed=rnd * 1000 + i)
                 for i, s in enumerate(rng.integers(0, 32, 8))]
        starts = sorted(coach_paths)
        for i in range(7):
            s = starts[int(rng.integers(len(starts)))]
            path = coach_paths[s]
            k = int(rng.integers(0, max(1, len(path) - 10)))
            tasks.append(dict(mode="brain", start=s, prefix=path[:k], brain=str(brain_p), seed=rnd * 1000 + 10 + i))
        t = time.time()
        res = pool.map(episode, tasks, chunksize=1)
        Xn = np.concatenate([r["obs"] for r in res if len(r["labels"])])
        yn = np.concatenate([r["labels"] for r in res if len(r["labels"])])
        Wn = np.concatenate([r["weights"] for r in res if len(r["labels"])])
        own_agree = float(np.mean([np.mean(np.array(r["actions"]) == r["labels"]) for r in res if len(r["labels"])]))
        X = np.concatenate([X, Xn]); y = np.concatenate([y, yn]); W = np.concatenate([W, Wn])
        np.savez(data_p, X=X, y=y, W=W)
        log(f"round {rnd} dagger: +{len(yn)} samples ({len(y)} total) in {time.time() - t:.0f}s; "
            f"brain agreed with coach on its own path {own_agree:.3f}; reached "
            f"{[(r['prefix_len'], r['max_x'], r['flag']) for r in res]}; critical {int((Wn > 1).sum())}")
        rep = teach(brain, X, y, args.passes, rng, log, W)
        brain.save(brain_p)
        history.append(dict(round=rnd, samples=int(len(y)), new=int(len(yn)), own_agree=own_agree,
                            teach=rep, dagger=[dict(start=r["task"]["start"], prefix=r["prefix_len"],
                                                    max_x=r["max_x"], flag=r["flag"]) for r in res]))
        hist_p.write_text(json.dumps(history))
        (out / f"brain_r{rnd:02d}.npz").write_bytes(brain_p.read_bytes()) if rnd % 3 == 0 else None
    pool.close()
    log("done")


if __name__ == "__main__":
    main()
