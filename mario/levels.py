"""ONE Cadence brain plays Super Mario Bros. in order, like a person: 1-1, 1-2, 1-3, 1-4, 2-1, ...

Per level:
  first sight  3 solo attempts BEFORE any training on the level (the generalization test).
  practice     rounds of: (a) self-play in the main process with `brain.step` and a reward
               (x progress, a death penalty, a flag bonus), so the brain learns from its own
               attempts through Cadence's reward learner, part of them restarted a little before
               where its best solo attempt died (practising the hard part); (b) a few coached
               attempts: the brain plays greedily and the look-ahead coach writes the button it
               would have pressed on each screen the brain reached (DAgger corrections; no coach
               demonstrations ever); the number of coached attempts per round shrinks with every
               level; (c) lessons from those coach corrections plus the brain's own attempts that
               made progress (self-imitation), with REPLAY of the rows from earlier levels' labels.
  solo eval    3 greedy attempts (`predict`), different only in idle frames before the first move.
               Cleared = flag in 2 of 3. Best checkpoint per level kept; never overwritten by a
               worse round; ROLLBACK rounds below the best -> back to the best state.
  move on      when cleared, or after CAP_MIN minutes of practice (a player who eventually gets
               through); retention on every earlier level is re-tested solo at each change.

    python levels.py --until 20:00
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LEVELS = [(w, s) for w in range(1, 9) for s in range(1, 5)]
EVAL_STARTS = (0, 6, 12)
MAX_DECISIONS = int(os.environ.get("MARIO_MAX_DECISIONS", 1500))
STUCK = 60
MENU = os.environ.get("MARIO_MENU", "lean")
LESSONS = int(os.environ.get("MARIO_LESSONS", 1500))
CAP_MIN = float(os.environ.get("MARIO_CAP_MIN", 38))
SELF_DECISIONS = int(os.environ.get("MARIO_SELF", 3000))         # reward-learning decisions per round (main process, ~27 ms each)
COACHED = (6, 0.12, 1)         # coached attempts per round: 6 * (1 - 0.12 * level index), at least 1
RECENCY, RECENCY_FLOOR = 0.75, 0.25
SELF_WEIGHT = 0.5              # self-imitation rows weigh half a coach correction
LATE_ETA = (0.0015, 0.00015)   # teaching rates from round LATE_FROM on a level (normalized: 0.003)
LATE_FROM = 3
ACTOR = (0.05, 0.005)          # reward learner rates (composed 1.0; the reward guide advises a tenth)
REPLAY = 0.1
OLD_PLAY = 0.1               # share of self-play decisions spent PLAYING an earlier level again
NO_COACH = set(os.environ.get("MARIO_NO_COACH", "").split(",")) - {""}  # levels where the coach can't clear: self-play only
ROLLBACK = 2
GAMES = ("hoi4",)


def level_name(lv):
    return f"{lv[0]}-{lv[1]}"


# ----------------------------------------------------------------------------------- workers
_BRAIN = None
_BRAIN_KEY = None


def _worker_init():
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def _brain(path):
    global _BRAIN, _BRAIN_KEY
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key != _BRAIN_KEY:
        from cadence import Brain
        _BRAIN = Brain.load(path)
        _BRAIN_KEY = key
    return _BRAIN


def make_game(level, start):
    from game import Game
    return Game(noop_start=start, level=tuple(level), menu=MENU, bg_per_frame=True)


def episode(task: dict) -> dict:
    """mode 'eval' (greedy brain, no labels) or 'coached' (greedy brain, coach labels each screen)."""
    from game import N_IN
    g = make_game(task["level"], task["start"])
    for a in task.get("prefix", []):
        g.step(a)
        if g.done:
            break
    brain = _brain(task["brain"])
    obs, labels, acts, weights = [], [], [], []
    label_on = task["mode"] == "coached"
    t0 = time.time()
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        o = g.observation()
        a = int(np.asarray(brain.predict(o[None])).reshape(-1)[0])
        if label_on:
            lab, wt = g.coach_weighted()
            obs.append(np.round(o * 255).astype(np.uint8)); labels.append(lab); weights.append(wt)
        acts.append(a)
        g.step(a)
    out = dict(task={k: v for k, v in task.items() if k != "prefix"}, prefix_len=len(task.get("prefix", [])),
               max_x=int(g.max_x), flag=bool(g.flag), died=bool(g.done and not g.flag), decisions=g.decisions,
               actions=acts, seconds=time.time() - t0)
    if label_on:
        out["obs"] = np.array(obs, dtype=np.uint8).reshape(-1, N_IN)
        out["labels"] = np.array(labels, dtype=np.int64)
        out["weights"] = np.array(weights, dtype=np.float64)
    g.close()
    return out


class Workers:
    """Process pool that survives a dead worker. multiprocessing.Pool hangs forever on a worker
    that dies mid-task, and replaces dead workers by spawning new ones that unpickle the parent's
    pipe handles; if the parent is going away at that moment the children die with
    PermissionError [WinError 5] in DuplicateHandle (the smoke run's end: the job was stopped from
    outside, taskkill took the workers first, Pool respawned 5, the parent then vanished). Here a
    broken pool is shut down and rebuilt, and a task that fails twice runs in the main process."""

    def __init__(self, n, log):
        self.n, self.log, self.ex = n, log, None
        self._new()

    def _new(self):
        if self.ex is not None:
            try:
                self.ex.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
        self.ex = ProcessPoolExecutor(self.n, mp_context=get_context("spawn"), initializer=_worker_init)

    def submit_all(self, tasks):
        return [self.ex.submit(episode, t) for t in tasks]

    def gather(self, tasks, futs):
        out = []
        for t, f in zip(tasks, futs):
            try:
                out.append(f.result(timeout=3600))
            except (BrokenProcessPool, OSError, EOFError, TimeoutError) as e:
                self.log(f"  worker failure {e!r}: rebuilding the pool, retrying the task")
                self._new()
                try:
                    out.append(self.ex.submit(episode, t).result(timeout=3600))
                except Exception as e2:  # noqa: BLE001
                    self.log(f"  retry failed {e2!r}: running it in the main process")
                    self._new()
                    out.append(episode(t))
        return out

    def map(self, tasks):
        return self.gather(tasks, self.submit_all(tasks))

    def close(self):
        self.ex.shutdown(wait=True)


# ---------------------------------------------------------------------------------- helpers
def game_running() -> bool:
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30).stdout.lower()
    except Exception:  # noqa: BLE001
        return False
    return any(gm in out for gm in GAMES)


def score(ev):
    xs = [e["max_x"] for e in ev]
    return [sum(e["flag"] for e in ev), float(np.median(xs)), float(np.mean(xs))]


def load_data(path):
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in ("X", "y", "W", "R")}


def save_data(path, d):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, path)


def add(d, X, y, W, rnd):
    if len(y) == 0:
        return d
    new = dict(X=X, y=y.astype(np.int64), W=W.astype(np.float64), R=np.full(len(y), rnd, dtype=np.int64))
    if d is None:
        return new
    return {k: np.concatenate([d[k], new[k]]) for k in d}


def teach(brain, cur, old, lessons, rnd, rng, log):
    age = rnd - cur["R"]
    w = cur["W"] * np.maximum(RECENCY ** age, RECENCY_FLOOR)
    p = w / w.sum(); n = len(cur["y"])
    po = None if old is None or not len(old["y"]) else old["W"] / old["W"].sum()
    n_old = 3 if po is not None else 0  # 3 of 32 rows ~ 10 %
    t = time.time(); agree = []
    for i in range(lessons):
        idx = rng.choice(n, 32 - n_old, p=p)
        x = cur["X"][idx]; yy = cur["y"][idx]
        if n_old:
            jdx = rng.choice(len(old["y"]), n_old, p=po)
            x = np.concatenate([x, old["X"][jdx]]); yy = np.concatenate([yy, old["y"][jdx]])
        brain.learner.step(brain.stimulus(x.astype(np.float64) / 255.0, memory=False), yy)
        if i % 500 == 0 or i == lessons - 1:
            chk = rng.choice(n, min(n, 512), replace=False)
            pr = np.asarray(brain.predict(cur["X"][chk].astype(np.float64) / 255.0)).reshape(-1)
            agree.append(float((pr == cur["y"][chk]).mean()))
    log(f"  taught {lessons} lessons ({n} rows here, replay {0 if old is None else len(old['y'])}) "
        f"agree {['%.2f' % a for a in agree]} {time.time() - t:.0f}s")
    return agree


def set_rates(brain, late, log):
    # Lower late teaching rates were planned (LATE_ETA) but cadence 0.91.1 refuses learning_eta under
    # teaching="normalized", so that retune always failed in the blind run (from round 3 of each
    # level). The actor rates had already been set at rounds 1-2 and persist. Same behaviour here,
    # without the failing call: teaching stays "normalized", actor rates set every round.
    try:
        brain.retune(actor_eta=ACTOR[0], actor_eta_bias=ACTOR[1])
    except Exception as e:  # noqa: BLE001
        log(f"  retune failed: {e!r}")


def self_play(brain, lv, budget, prefixes, rng, log):
    """Reward learning from the brain's own attempts (sampled actions through brain.step).
    Returns self-imitation rows: screens and own actions on stretches that set a new best x for
    the attempt, minus the last 15 decisions before a death."""
    from game import N_IN
    used = 0; runs = []; Xs, ys = [], []
    while used < budget:
        start = int(rng.integers(0, 32))
        pre = prefixes[int(rng.integers(len(prefixes)))] if prefixes and rng.random() < 0.5 else []
        g = make_game(lv, start)
        for a in pre:
            g.step(a)
            if g.done:
                break
        if g.done:
            g.close(); continue
        a = int(brain.step(g.observation()[None])[0])
        obs, acts, best_at = [], [], []
        while True:
            o = g.observation(); x0 = g.x; prev_best = g.max_x
            obs.append(o); acts.append(a)
            g.step(a)
            best_at.append(g.max_x > prev_best)
            r = float(np.clip((g.x - x0) / 8.0, -1, 1))
            end = g.done or g.decisions >= MAX_DECISIONS or g.stuck >= STUCK
            if g.done and not g.flag:
                r -= 2.0
            if g.flag:
                r += 5.0
            if end:
                brain.step(g.observation()[None], reward=np.array([r]), done=np.array([True]))
                break
            a = int(brain.step(g.observation()[None], reward=np.array([r]), done=np.array([False]))[0])
        used += len(acts)
        keep = np.zeros(len(acts), bool)
        last_best = np.flatnonzero(best_at)
        if len(last_best):
            hi = last_best[-1] - (15 if g.done and not g.flag else 0)
            keep[:max(0, hi)] = True
        if keep.any():
            Xs.append(np.round(np.array(obs)[keep] * 255).astype(np.uint8)); ys.append(np.array(acts)[keep])
        runs.append((len(pre), int(g.max_x), bool(g.flag)))
        g.close()
    X = np.concatenate(Xs) if Xs else np.zeros((0, N_IN), np.uint8)
    y = np.concatenate(ys) if ys else np.zeros(0, np.int64)
    return X, y, runs, used


# ------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/levels")
    ap.add_argument("--start-brain", default="runs/main/brain_e10.npz")
    ap.add_argument("--until", default="20:00")
    ap.add_argument("--only", default=None, help="stop after this level (e.g. 1-1 for a quick check)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    hh, mm = (int(v) for v in args.until.split(":"))
    deadline = dt.datetime.now().replace(hour=hh, minute=mm, second=0).timestamp()
    os.environ["NUMBA_NUM_THREADS"] = "6"
    import cadence
    import numba
    from cadence import Brain
    rng = np.random.default_rng(int(time.time()))
    pool = Workers(args.workers, log)

    def pmap(tasks):
        if game_running():
            log("  hoi4 running: 2 cores")
            numba.set_num_threads(1)
            res = []
            for i in range(0, len(tasks), 2):
                res += pool.map(tasks[i:i + 2])
            return res
        return pool.map(tasks)

    state_p = out / "state.json"; brain_p = out / "brain.npz"
    if state_p.exists():
        state = json.loads(state_p.read_text())
        log(f"resumed at level {state['level']} round {state['round']}")
    else:
        if args.start_brain == "fresh":
            from game import N_IN as _n
            Brain.compose(_n, 7, modules=(256,), seed=0, teaching="normalized", learning_surprise=True).save(brain_p)
        else:
            shutil.copyfile(HERE / args.start_brain, brain_p)
        state = dict(level=0, round=-1, start_brain=args.start_brain, menu=MENU, levels={}, history=[],
                     retention=[], rollbacks=[], cadence=cadence.__version__,
                     senses=os.environ.get("MARIO_SENSES", "blind"))
        log(f"new job: cadence {cadence.__version__}, brain from {args.start_brain}, menu {MENU}")
    state["no_coach"] = sorted(set(state.get("no_coach", [])) | NO_COACH)
    brain = Brain.load(brain_p)

    def save_state():
        tmp = state_p.with_suffix(".tmp"); tmp.write_text(json.dumps(state)); os.replace(tmp, state_p)

    def solo(lv):
        brain.save(brain_p)
        ev = pmap([dict(mode="eval", level=list(lv), start=s, brain=str(brain_p)) for s in EVAL_STARTS])
        return [dict(start=r["task"]["start"], max_x=r["max_x"], flag=r["flag"], decisions=r["decisions"],
                     actions=r["actions"]) for r in ev]

    def old_data(idx):
        parts = [load_data(out / f"data_{level_name(l)}.npz") for l in LEVELS[:idx]]
        parts = [p for p in parts if p is not None]
        return None if not parts else {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "W", "R")}

    stop_after = LEVELS.index(tuple(int(v) for v in args.only.split("-"))) if args.only else len(LEVELS) - 1
    while state["level"] <= stop_after and time.time() < deadline:
        li = state["level"]; lv = LEVELS[li]; name = level_name(lv)
        rec = state["levels"].setdefault(name, dict(index=li, coached_attempts=0, coach_labels=0, self_rows=0,
                                                    self_decisions=0, practice_s=0.0, rounds=0, cleared=None))
        data_p = out / f"data_{name}.npz"
        if "first_sight" not in rec:
            numba.set_num_threads(6)
            ev = solo(lv); sc = score(ev)
            rec["first_sight"] = dict(eval=ev, score=sc, time=time.time())
            rec["best"] = dict(round=0, score=sc, time=time.time())
            shutil.copyfile(brain_p, out / f"best_{name}.npz")
            rec["entered"] = time.time(); state["round"] = 0
            log(f"=== {name} FIRST SIGHT x={[e['max_x'] for e in ev]} flags={sc[0]}/3")
            state["history"].append(dict(level=name, round=0, time=time.time(), eval=ev, score=sc, first_sight=True))
            if sc[0] >= 2:
                rec["cleared"] = dict(round=0, time=time.time(), date=time.strftime("%Y-%m-%d %H:%M"))
            save_state()
        if rec["cleared"] is None and rec["practice_s"] < CAP_MIN * 60 and time.time() < deadline:
            t_round = time.time(); rnd = state["round"] + 1
            cur = load_data(data_p); old = old_data(li)
            # best solo attempt -> practise from a little before where it ended
            best_ev = max((e for h in state["history"] if h["level"] == name for e in h["eval"]),
                          key=lambda e: e["max_x"])
            acts = best_ev["actions"]
            prefixes = [acts[:max(0, len(acts) - k)] for k in (20, 40, 60) if len(acts) > k]
            n_coached = max(COACHED[2], int(round(COACHED[0] * (1 - COACHED[1] * li))))
            if name in NO_COACH:
                n_coached = 0  # the coach can't clear this level; the brain learns from its own play only
            brain.save(brain_p)
            tasks = []
            for i in range(n_coached):
                tk = dict(mode="coached", level=list(lv), start=int(rng.integers(0, 32)), brain=str(brain_p))
                if prefixes and i % 2 == 1:
                    tk["prefix"] = prefixes[int(rng.integers(len(prefixes)))]
                tasks.append(tk)
            hoi = game_running()
            if hoi:
                log("  hoi4 running: 1 worker + main")
            futs = pool.submit_all(tasks) if not hoi else None
            numba.set_num_threads(1)
            set_rates(brain, rnd >= LATE_FROM, log)
            old_budget = int(OLD_PLAY * SELF_DECISIONS) if li > 0 else 0
            Xs, ys, runs, used = self_play(brain, lv, SELF_DECISIONS - old_budget, prefixes, rng, log)
            if old_budget:  # play an earlier level again (reward learning + its own good stretches)
                oi = int(rng.integers(li)); olv = LEVELS[oi]; oname = level_name(olv)
                oX, oy, oruns, oused = self_play(brain, olv, old_budget, [], rng, log)
                od_p = out / f"data_{oname}.npz"
                od = add(load_data(od_p), oX, oy, np.full(len(oy), SELF_WEIGHT), -1)
                if od is not None:
                    save_data(od_p, od)
                state.setdefault("old_play", []).append(dict(level=name, round=rnd, played=oname, decisions=int(oused),
                                                             runs=oruns, time=time.time()))
                rec["old_play_decisions"] = rec.get("old_play_decisions", 0) + int(oused)
                log(f"  replayed {oname}: {oused} decisions, x {[r[1] for r in oruns]}")
            if futs is not None:
                res = pool.gather(tasks, futs)
            else:
                res = pmap(tasks)
            rows = [r for r in res if len(r["labels"])]
            if rows:
                cur = add(cur, np.concatenate([r["obs"] for r in rows]), np.concatenate([r["labels"] for r in rows]),
                          np.concatenate([r["weights"] for r in rows]), rnd)
            n_coach = sum(len(r["labels"]) for r in rows)
            cur = add(cur, Xs, ys, np.full(len(ys), SELF_WEIGHT), rnd)
            if cur is not None:
                save_data(data_p, cur)
            rec["coached_attempts"] += len(tasks); rec["coach_labels"] += n_coach
            rec["self_rows"] += int(len(ys)); rec["self_decisions"] += int(used)
            log(f"{name} r{rnd}: self-play {used} decisions {runs[:8]}{'...' if len(runs) > 8 else ''} "
                f"+{len(ys)} own rows; coached {len(tasks)} attempts +{n_coach} labels "
                f"{[(r['prefix_len'], r['max_x']) for r in res]}")
            numba.set_num_threads(2 if game_running() else 6)
            if cur is not None:
                teach(brain, cur, old, LESSONS, rnd, rng, log)
            ev = solo(lv); sc = score(ev)
            shutil.copyfile(brain_p, out / "ckpt" / f"{name}_r{rnd:02d}.npz")
            rec["practice_s"] += time.time() - t_round; rec["rounds"] = rnd; state["round"] = rnd
            improved = tuple(sc) > tuple(rec["best"]["score"])
            log(f"{name} r{rnd} SOLO x={[e['max_x'] for e in ev]} flags={sc[0]}/3 median={sc[1]:.0f} "
                f"practice {rec['practice_s'] / 60:.0f} min{'  NEW BEST' if improved else ''}")
            state["history"].append(dict(level=name, round=rnd, time=time.time(), eval=ev, score=sc,
                                         coach_labels=n_coach, self_rows=int(len(ys)), self_decisions=int(used)))
            if improved:
                shutil.copyfile(brain_p, out / f"best_{name}.npz")
                rec["best"] = dict(round=rnd, score=sc, time=time.time()); rec["below"] = 0
            else:
                rec["below"] = rec.get("below", 0) + 1
                if rec["below"] >= ROLLBACK and rec["best"]["score"][1] > 1.15 * sc[1]:
                    brain = Brain.load(out / f"best_{name}.npz")
                    state["rollbacks"].append(dict(level=name, round=rnd, to=rec["best"]["round"]))
                    log(f"  rollback to the round-{rec['best']['round']} state (labels kept)"); rec["below"] = 0
            if sc[0] >= 2:
                rec["cleared"] = dict(round=rnd, time=time.time(), date=time.strftime("%Y-%m-%d %H:%M"))
                log(f"*** {name} CLEARED SOLO at round {rnd} ***")
                (out / "CLEARED.txt").open("a").write(f"{name} r{rnd} {time.strftime('%H:%M')}\n")
            save_state()
            continue
        # ---- level change: leave with the best state of this level, test retention
        if rec["cleared"] is None:
            rec["capped"] = True
            if rec["best"]["round"] != state["round"]:
                brain = Brain.load(out / f"best_{name}.npz")
                log(f"  leaving {name} with its best state (round {rec['best']['round']})")
        log(f"=== leaving {name}: {'cleared r%d' % rec['cleared']['round'] if rec['cleared'] else 'capped'}; "
            f"coach labels {rec['coach_labels']}, own rows {rec['self_rows']}")
        numba.set_num_threads(6)
        ret = {}
        for l in LEVELS[:li + 1]:
            e2 = solo(l)
            ret[level_name(l)] = dict(flags=sum(e["flag"] for e in e2), xs=[e["max_x"] for e in e2], eval=e2)
        state["retention"].append(dict(after=name, time=time.time(), results=ret))
        log("  retention: " + ", ".join(f"{k} {v['flags']}/3 {v['xs']}" for k, v in ret.items()))
        state["level"] += 1; state["round"] = -1
        brain.save(brain_p); save_state()
    log("job ended")
    try:
        pool.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
