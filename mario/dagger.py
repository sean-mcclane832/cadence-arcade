"""DAgger on World 1-1 with a choice of learner and senses: the controlled experiment of
LEARNING_FIX.md (E3/E4).

    python dagger.py --out runs/fix_mlp_body --learner mlp --senses body --hours 1
    python dagger.py --out runs/fix_cad_body --learner cadence --senses body --hours 3

senses
  blind  the overnight eyes: 8 x 8 grey blocks of the present and previous decision's frame,
         as distance from the frame's background grey (1,664 numbers). Nothing else.
  body   the same eyes, plus what Mario's own body and thumb feel: the buttons held during the
         last decision (A, B, right, left) and the proprioception of game.proprio (x and y speed,
         on ground / jumping / falling, still rising, facing, power, hurt and star timers, time
         left). No enemy, tile or pit RAM: the world still comes only from the eyes.
         (42 numbers x 8 copies = 336; 2,000 inputs.)

Why "body": SMB jumps on the PRESS of A. Holding A from the last decision makes a new
run+jump do nothing, so the coach's label for the same screen flips with the held button
(release / press). A memoryless screen-only policy cannot see which, and on a static screen
(pressed against a pipe) its greedy answer repeats forever: 31 of the last 60 overnight solo
attempts ended that way (diag/replay_eval.py, diag/alias_pipe.py).

Rounds: coach demonstrations (round 0), then the learner plays greedily from the start and from
points along the coach's path while the coach labels every screen it reaches; all labels so far
are taught; 5 solo starts evaluated per round. The coach never plays during the solo eval.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, os.environ.get("MARIO_THREADS", "1"))
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EVAL_STARTS = (0, 5, 11, 17, 23)
MAX_DECISIONS = 900
STUCK = 40
COPIES = 8
HELD_BITS = (0x01, 0x02, 0x80, 0x40)  # A, B, right, left


def n_inputs(senses: str) -> int:
    from game import N_FRAME, N_PROPRIO_ONE
    return 2 * N_FRAME + (0 if senses == "blind" else (N_PROPRIO_ONE + len(HELD_BITS)) * COPIES)


class Body:
    """A Game plus the senses of this experiment."""

    def __init__(self, start: int, senses: str, menu: str, level=(1, 1)):
        from game import Game
        self.g = Game(noop_start=start, level=tuple(level), menu=menu, bg_per_frame=True)
        self.senses = senses
        self.held = 0x00

    def obs(self) -> np.ndarray:
        g = self.g
        eyes = np.concatenate([g.cur, g.prev])
        if self.senses == "blind":
            return eyes
        from game import proprio
        held = np.array([1.0 if self.held & b else 0.0 for b in HELD_BITS])
        body = np.concatenate([proprio(g.env), held])
        return np.concatenate([eyes, np.tile(body, COPIES)])

    def step(self, a: int):
        from game import ACTIONS
        self.g.step(a)
        self.held = ACTIONS[a]


# ---------------------------------------------------------------------------------- learners
class MLP:
    """One hidden layer of 256, ReLU, Adam (the backprop control)."""

    def __init__(self, n_in, hidden=256, n_out=7, seed=0, lr=1e-3):
        r = np.random.default_rng(seed)
        self.p = [r.normal(0, 1 / np.sqrt(n_in), (n_in, hidden)), np.zeros(hidden),
                  r.normal(0, 1 / np.sqrt(hidden), (hidden, n_out)), np.zeros(n_out)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]
        self.t, self.lr = 0, lr

    def logits(self, x):
        h = np.maximum(0, x @ self.p[0] + self.p[1])
        return h, h @ self.p[2] + self.p[3]

    def predict(self, x):
        return np.argmax(self.logits(x)[1], axis=1)

    def step(self, x, y, w=None):
        h, z = self.logits(x)
        pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        g = pr; g[np.arange(len(y)), y] -= 1
        g = g * (np.ones(len(y)) if w is None else w)[:, None] / len(y)
        gh = (g @ self.p[2].T) * (h > 0)
        grads = [x.T @ gh, gh.sum(0), h.T @ g, g.sum(0)]
        self.t += 1
        for i, gr in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr; self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def save(self, path):
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, *self.p); os.replace(tmp, path)

    @staticmethod
    def load(path):
        d = np.load(path); m = MLP.__new__(MLP); m.p = [d[f"arr_{i}"] for i in range(4)]
        return m


class Vote:
    """Majority vote of several saved brains (ties go to the last listed, the newest)."""

    def __init__(self, paths):
        from cadence import Brain
        self.bs = [Brain.load(p) for p in paths]

    def predict(self, x):
        votes = np.stack([np.asarray(b.predict(x)).reshape(-1) for b in self.bs])  # (k, n)
        out = []
        for col in votes.T:
            c = np.bincount(col, minlength=7); top = np.flatnonzero(c == c.max())
            out.append(next(v for v in col[::-1] if v in top))
        return np.array(out)


_POLICY, _KEY = None, None


def _policy(path, kind):
    global _POLICY, _KEY
    st = os.stat(path); key = (path, st.st_mtime_ns, st.st_size)
    if key != _KEY:
        if str(path).endswith(".ens.json"):
            _POLICY = Vote(json.loads(Path(path).read_text()))
        elif kind == "mlp":
            _POLICY = MLP.load(path)
        else:
            from cadence import Brain
            _POLICY = Brain.load(path)
        _KEY = key
    return _POLICY


def _worker_init():
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def _play(task, pol, label_at=None, rng=None):
    """One attempt. label_at: None (no labels), "all", or a set of decision indices to label."""
    b = Body(task["start"], task["senses"], task["menu"], task.get("level", (1, 1)))
    for a in task.get("prefix", []):
        b.step(a)
        if b.g.done:
            break
    g = b.g; obs, labels, weights, acts = [], [], [], []
    script = task.get("script")
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        i = len(acts)
        if script is not None and i >= len(script):
            break
        o = b.obs()
        want = label_at == "all" or (label_at is not None and i in label_at)
        lab, wt = g.coach_weighted() if want else (-1, 0.0)
        if script is not None:
            a = script[i]
        elif task["mode"] == "coach":
            a = lab if rng.random() >= task.get("eps", 0.0) else int(rng.choice(task["menu_buttons"]))
        else:
            a = int(np.asarray(pol.predict(o[None])).reshape(-1)[0])
        if want:
            obs.append(np.round(o * 255).astype(np.uint8)); labels.append(lab); weights.append(wt)
        acts.append(a)
        b.step(a)
    out = dict(start=task["start"], prefix_len=len(task.get("prefix", [])), max_x=int(g.max_x), flag=bool(g.flag),
               died=bool(g.done and not g.flag), decisions=g.decisions, actions=acts,
               end="flag" if g.flag else ("died" if g.done else "stuck"))
    n = len(b.obs())
    out["obs"] = np.array(obs, np.uint8).reshape(-1, n)
    out["labels"] = np.array(labels, np.int64); out["weights"] = np.array(weights)
    g.close()
    return out


def episode(task: dict) -> dict:
    """coach: the coach plays (eps random buttons) and labels every screen. eval: the learner
    plays alone, no labels. dagger: the learner plays alone first; then the same attempt is
    replayed (the emulator is deterministic) and the coach labels the screens it reached: every
    decision of the last LAST_ALL before the end, and every EVERY-th one before that (the coach
    costs ~1-2 s a label; the end is where the attempt failed)."""
    rng = np.random.default_rng(task.get("seed", 0))
    if task["mode"] == "coach":
        return _play(task, None, "all", rng)
    pol = _policy(task["policy"], task["kind"])
    played = _play(task, pol, None, rng)
    if task["mode"] == "eval":
        return played
    acts = played["actions"]; n = len(acts)
    last_all, every = task.get("last_all", 60), task.get("every", 3)
    idx = {i for i in range(n) if i >= n - last_all or i % every == 0}
    lab = _play(dict(task, script=acts), None, idx, rng)
    order = sorted(idx)
    played.update(obs=lab["obs"], labels=lab["labels"], weights=lab["weights"],
                  own=np.array([acts[i] for i in order[:len(lab["labels"])]], np.int64))
    return played


# ---------------------------------------------------------------------------------- teaching
def teach(learner, kind, X, y, W, lessons, rng, log, batch=32, old=None, share=0.0):
    """Weighted 32-row lessons. old=(X, y, W) of earlier levels: round(share * 32) rows of every
    lesson come from there (replay), drawn by the same weights."""
    p = W / W.sum(); t = time.time()
    k_old = int(round(share * batch)) if old is not None else 0
    po = None if not k_old else old[2] / old[2].sum()
    for i in range(lessons):
        idx = rng.choice(len(y), batch - k_old, p=p)
        x = X[idx]; yy = y[idx]
        if k_old:
            j = rng.choice(len(old[1]), k_old, p=po)
            x = np.concatenate([x, old[0][j]]); yy = np.concatenate([yy, old[1][j]])
        x = x.astype(np.float64) / 255.0
        if kind == "mlp":
            learner.step(x, yy)
        else:
            learner.learner.step(learner.stimulus(x, memory=False), yy)
    chk = rng.choice(len(y), min(len(y), 2000), replace=False)
    pr = np.concatenate([np.asarray(learner.predict(X[chk[i:i + 500]].astype(np.float64) / 255.0)).reshape(-1)
                         for i in range(0, len(chk), 500)])
    c = W[chk] > 1
    rep = dict(lessons=lessons, s=round(time.time() - t), agree=round(float((pr == y[chk]).mean()), 3),
               agree_crit=round(float((pr[c] == y[chk][c]).mean()), 3) if c.any() else None)
    log(f"  taught {lessons} lessons on {len(y)} rows: agree {rep['agree']} crit {rep['agree_crit']} {rep['s']}s")
    return rep


def compose_cadence(n_in, arm):
    from cadence import Brain
    kw = dict(teaching="normalized", learning_surprise="nosurp" not in arm)
    b = Brain.compose(n_in, 7, modules=(256,), seed=0, **kw)
    if "e1" in arm:  # teaching steps 0.001 / 0.0001 instead of the normalized point's 0.003 / 0.0003
        c = b.learner.config
        b.learner.config = dataclasses.replace(c, eta=0.001, eta_bias=0.0001)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--learner", choices=("mlp", "cadence"), default="mlp")
    ap.add_argument("--senses", choices=("blind", "body"), default="body")
    ap.add_argument("--menu", default="lean")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lessons", type=int, default=3000)
    ap.add_argument("--arm", default="base")
    ap.add_argument("--max-rounds", type=int, default=99)
    ap.add_argument("--init-data", default=None, help="run dir whose data.npz + coach_paths.json start this run")
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--episodes", type=int, default=8, help="DAgger attempts per round (1/3 from the start)")
    ap.add_argument("--init-brain", default=None, help="a saved Cadence brain to continue (one life)")
    ap.add_argument("--replay", default=None, help="comma list of run dirs: earlier levels' coach labels")
    ap.add_argument("--replay-share", type=float, default=0.5)
    ap.add_argument("--watch", default="", help="comma list of earlier levels to test solo every round")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--eval-only", default="", help="comma list of levels: solo-test --init-brain, write first_sight.json, exit")
    ap.add_argument("--eval-starts", default="0,5,11,17,23")
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    for k_ in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k_] = os.environ.get("MARIO_THREADS", "1")
    from game import MENUS
    menu_buttons = sorted({p[0] for p in MENUS[args.menu]} | {a for p in MENUS[args.menu] for a in p})
    n_in = n_inputs(args.senses)
    deadline = time.time() + args.hours * 3600
    rng = np.random.default_rng(1)
    ex = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_worker_init)
    def pmap(ts, part=None):
        futs = [ex.submit(episode, t_) for t_ in ts]; res = []
        for k, f in enumerate(futs):
            r = f.result(); res.append(r)
            if part is not None and len(r.get("labels", [])):
                (out / "parts").mkdir(exist_ok=True)
                np.savez(out / "parts" / f"{part}_{k:02d}.npz", X=r["obs"], y=r["labels"], W=r["weights"])
        return res
    level = tuple(int(v) for v in args.level.split("-"))
    starts = [int(v) for v in args.eval_starts.split(",")]
    base = dict(senses=args.senses, menu=args.menu, kind=args.learner, menu_buttons=menu_buttons, level=level)
    pol_p = out / ("policy.npz")
    if args.eval_only:
        pol_p.write_bytes((HERE / args.init_brain).read_bytes())
        res = {}
        for lv in args.eval_only.split(","):
            lt = tuple(int(v) for v in lv.split("-"))
            ev = pmap([dict(base, level=lt, mode="eval", start=s_, policy=str(pol_p)) for s_ in starts])
            res[lv] = [{k: r[k] for k in ("start", "max_x", "flag", "end", "actions")} for r in ev]
            log(f"eval-only {lv}: x={[r['max_x'] for r in ev]} ends={[r['end'][0] for r in ev]} median={np.median([r['max_x'] for r in ev]):.0f}")
        (out / "first_sight.json").write_text(json.dumps(res)); ex.shutdown(); return
    old = None
    if args.replay:
        parts = [np.load((HERE / r_).resolve() / "data.npz") for r_ in args.replay.split(",")]
        old = tuple(np.concatenate([d_[k] for d_ in parts]) for k in ("X", "y", "W"))
        log(f"replay: {len(old[1])} rows from {args.replay}, share {args.replay_share}")
    log(f"new run: learner {args.learner} senses {args.senses} ({n_in} inputs) menu {args.menu} arm {args.arm}")

    if args.init_data:  # start from another run's coach labels (same senses), skip round 0
        src = (HERE / args.init_data).resolve()
        d = np.load(src / "data.npz"); X, y, W = d["X"], d["y"], d["W"]
        coach_paths = [tuple(c) for c in json.loads((src / "coach_paths.json").read_text())]
        np.savez(out / "data.npz", X=X, y=y, W=W); (out / "coach_paths.json").write_text(json.dumps(coach_paths))
        log(f"init data from {args.init_data}: {len(y)} rows, critical {int((W > 1).sum())}")
    else:
        # round 0: coach demonstrations, some with random wrong buttons (recoveries)
        tasks = [dict(base, mode="coach", start=s, eps=0.0, seed=s) for s in (0, 3, 8, 13, 20, 27)]
        tasks += [dict(base, mode="coach", start=s, eps=0.12, seed=100 + s) for s in (1, 15, 24)]
        tasks += [dict(base, mode="coach", start=s, eps=0.25, seed=200 + s) for s in (10, 22)]
        t = time.time(); res = pmap(tasks, "r00")
        X = np.concatenate([r["obs"] for r in res]); y = np.concatenate([r["labels"] for r in res])
        W = np.concatenate([r["weights"] for r in res])
        coach_paths = [(t_["start"], r["actions"]) for t_, r in zip(tasks, res) if r["flag"] and t_["eps"] == 0.0]
        log(f"round 0 coach: {len(y)} rows in {time.time() - t:.0f}s, flags {sum(r['flag'] for r in res)}/{len(res)}, "
            f"labels {np.bincount(y, minlength=7).tolist()}, critical {int((W > 1).sum())}")
        np.savez(out / "data.npz", X=X, y=y, W=W)
        (out / "coach_paths.json").write_text(json.dumps(coach_paths))
        if args.collect_only:
            ex.shutdown(); log("collected; exit"); return
    if args.learner == "mlp":
        learner = MLP(n_in)
    elif args.init_brain:
        from cadence import Brain
        learner = Brain.load(HERE / args.init_brain)
        if "e1" in args.arm:
            c = learner.learner.config
            learner.learner.config = dataclasses.replace(c, eta=0.001, eta_bias=0.0001)
        log(f"continuing brain {args.init_brain} (eta {learner.learner.config.eta}, surprise {learner.learner.config.surprise})")
    else:
        learner = compose_cadence(n_in, args.arm)
    history = []
    rnd = 0
    while True:
        rep = teach(learner, args.learner, X, y, W, args.lessons, rng, log, old=old, share=args.replay_share)
        learner.save(pol_p)
        ev = pmap([dict(base, mode="eval", start=s, policy=str(pol_p)) for s in starts])
        xs = [r["max_x"] for r in ev]; flags = sum(r["flag"] for r in ev)
        log(f"round {rnd} SOLO x={xs} ends={[r['end'][0] for r in ev]} flags={flags}/5 median={np.median(xs):.0f}"
            f"  ({(time.time() - deadline + args.hours * 3600) / 60:.0f} min)")
        history.append(dict(round=rnd, rows=int(len(y)), teach=rep, eval=[{k: r[k] for k in ("start", "max_x", "flag", "end", "actions")} for r in ev],
                            minutes=(time.time() - deadline + args.hours * 3600) / 60))
        for lv in [v for v in args.watch.split(",") if v]:
            lt = tuple(int(v) for v in lv.split("-"))
            ew = pmap([dict(base, level=lt, mode="eval", start=s_, policy=str(pol_p)) for s_ in starts])
            history[-1].setdefault("watch", {})[lv] = [{k: r[k] for k in ("start", "max_x", "flag", "end")} for r in ew]
            log(f"  retention {lv}: x={[r['max_x'] for r in ew]} flags={sum(r['flag'] for r in ew)}")
        (out / "history.json").write_text(json.dumps(history))
        (out / f"policy_r{rnd:02d}.npz").write_bytes(pol_p.read_bytes())
        if time.time() > deadline or rnd >= args.max_rounds:
            break
        if flags >= 4 and len(history) >= 2 and sum(e["flag"] for e in history[-2]["eval"]) >= 4:
            log("flag on 4+ of 5 two rounds running; stopping"); break
        rnd += 1
        tasks = [dict(base, mode="dagger", start=int(s), policy=str(pol_p), seed=rnd * 100 + i)
                 for i, s in enumerate(rng.integers(0, 32, args.episodes // 3))]
        for i in range(args.episodes - args.episodes // 3):
            s0, path = coach_paths[int(rng.integers(len(coach_paths)))]
            k = int(rng.integers(0, max(1, len(path) - 20)))
            tasks.append(dict(base, mode="dagger", start=s0, prefix=path[:k], policy=str(pol_p),
                              seed=rnd * 100 + 50 + i))
        t = time.time(); res = pmap(tasks, f"r{rnd:02d}")
        res = [r for r in res if len(r["labels"])]
        own = float(np.mean(np.concatenate([r["own"] == r["labels"] for r in res])))
        X = np.concatenate([X] + [r["obs"] for r in res]); y = np.concatenate([y] + [r["labels"] for r in res])
        W = np.concatenate([W] + [r["weights"] for r in res])
        np.savez(out / "data.npz", X=X, y=y, W=W)
        log(f"round {rnd} dagger: +{sum(len(r['labels']) for r in res)} rows ({len(y)}) {time.time() - t:.0f}s; own-path agreement "
            f"{own:.3f}; reached {[(r['prefix_len'], r['max_x'], r['end'][0]) for r in res]}")
    ex.shutdown()
    log("done")


if __name__ == "__main__":
    main()
