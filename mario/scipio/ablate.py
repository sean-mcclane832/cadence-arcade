"""Play-time ablation of Scipio's memory (registered 11/14 before launch; deep review fix 1, harness fix 5b).

Plays SEL8 (24 levels x cs.SEL8, greedy, act mode, stuck limit 225) on ONE checkpoint in three modes:
  none   the brain as trained
  time   TIME-ONLY ablation (the verdict for rule 1(i)): before every act the heard history is set to the previous
         observation (what a decay-0 history holds; zeros at the attempt start) and the efference copy runs at decay 0
         (retune(efference_decay=0): it holds the last command). The amplitudes and learned gains are untouched, so the
         drive keeps its size and only the time scale is removed.
  amp0   history_amplitude = 0 and efference_amplitude = 0 (mem.PART_OFF "echo" + "button"); a drive-size shock,
         reported beside the time-only ablation, never the verdict.
Everything is play-time state in this process and its workers. The brain is never taught and never saved.

Per mode: SEL8 mean, flags, died, stuck split into "never left the start" (stuck, max x <= 60) and mid-level, and the
shock check: agreement with the coach on coach run 5 of every level (first 200 decisions, teacher forced through
mem.prime at HORIZON 250, the same frames as scipio/cato_fit.py; every coach run is taught to Scipio, so these are
taught frames, there is no held-out coach run).
    py099.sh ablate.py ../coldstart/runs/ScipioA/ScipioA_ck0050.npz --out scratch/ablate/ScipioA_ck0050.json --workers 2
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coldstart")); sys.path.insert(0, str(HERE.parent / "memory"))
import cs  # noqa: E402
import mem  # noqa: E402

mem.HORIZON = 250  # Scipio's prime window (set here, never in mem.py)
FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
LEVELS = [f"{w}-{s}" for w in range(1, 7) for s in range(1, 5)]
MODES = ("none", "time", "amp0")


class TimeOnly(mem.MemPlayer):
    """Greedy act-mode player whose memories hold only the last moment (decay 0), amplitudes untouched."""

    def __init__(self, path, senses):
        super().__init__(path, "greedy", senses=senses)
        if self.b.efference is not None:
            self.b.retune(efference_decay=0.0)

    def begin(self, seed=0):
        super().begin(seed); self.prev = None

    def act(self, o):
        h = self.b.history
        if h is not None:
            h.trace = (np.zeros((1, h.width)) if self.prev is None else self.prev[None].astype(np.float64)).copy()
        a = super().act(o)
        self.prev = np.asarray(o, np.float64).copy()
        return a


_P = {}


def _player(path, mode, senses):
    key = (path, os.stat(path).st_mtime_ns, mode)
    if key not in _P:
        _P.clear()
        if mode == "time":
            _P[key] = TimeOnly(path, senses)
        else:
            _P[key] = mem.MemPlayer(path, "greedy", off=("echo", "button") if mode == "amp0" else (), senses=senses)
    return _P[key]


def tail_same(acts):
    n = 0
    for a in reversed(acts):
        if a != acts[-1]:
            break
        n += 1
    return n


def episode(task):
    pl = _player(task["policy"], task["mode"], task["senses"])
    r = mem.play_attempt(pl, task["level"], task["start"])
    return dict(level=task["level"], start=task["start"], mode=task["mode"], max_x=r["max_x"], end=r["end"],
                decisions=r["decisions"], tail=tail_same(r["actions"]) if r["actions"] else 0)


def summarize(res):
    st = [r for r in res if r["end"] == "stuck"]
    return dict(n=len(res), mean=round(float(np.mean([r["max_x"] for r in res])), 1), flags=sum(r["end"] == "flag" for r in res),
                died=sum(r["end"] == "died" for r in res), stuck=len(st), stuck_nostart=sum(r["max_x"] <= 60 for r in st),
                stuck_mid=sum(r["max_x"] > 60 for r in st), stuck_tail100=sum(r["tail"] >= 100 for r in st))


def coach_frames(T=200):
    items, Y, P = [], [], []
    for L in LEVELS:
        z = np.load(FAB / L / f"seq_{L}_coach.npz"); off = z["off"]
        lab = np.full(len(z["O"]), -1); lab[z["li"]] = z["y"]
        O, A, y = z["O"][off[5]:off[6]], np.asarray(z["A"][off[5]:off[6]], np.int64), lab[off[5]:off[6]]
        for t in range(min(T, len(O))):
            if y[t] >= 0:
                items.append((O, A, t)); Y.append(int(y[t])); P.append(int(A[t - 1]) if t else -1)
    return items, np.array(Y), np.array(P)


def agreement(path, mode, items, Y, P):
    """Teacher-forced coach agreement under a mode (exact traces via mem.prime; time: last moment only; amp0: silent)."""
    from cadence import Brain
    b = mem.make_cold(Brain.load(path))
    if mode == "amp0":
        b.retune(**mem.PART_OFF["echo"], **mem.PART_OFF["button"])
    ans = []
    for s in range(0, len(items), 32):
        it = items[s:s + 32]
        x = mem.prime(b, it, 0)
        if mode == "time":
            if b.history is not None:
                b.history.trace = np.stack([O[t - 1].astype(np.float64) / 255.0 if t else np.zeros(O.shape[1]) for O, _, t in it])
            if b.efference is not None:
                b.efference.trace = np.stack([mem.onehot(int(A[t - 1])) if t else np.zeros(7) for _, A, t in it])
        ans.append(np.asarray(b.act(x, greedy=True)).reshape(len(it), -1)[:, 0])
    ans = np.concatenate(ans); ok = ans == Y; c = Y != P
    return dict(all=round(float(ok.mean()), 4), repeat=round(float(ok[~c].mean()), 4), change=round(float(ok[c].mean()), 4),
                n=int(len(Y)), n_change=int(c.sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--levels", default=",".join(LEVELS)); ap.add_argument("--starts", default=",".join(map(str, cs.SEL8)))
    ap.add_argument("--no-agree", action="store_true")
    a = ap.parse_args()
    assert mem.D.STUCK == 225, mem.D.STUCK
    path = str(Path(a.ckpt).resolve()); modes = a.modes.split(","); levels = a.levels.split(","); starts = [int(s) for s in a.starts.split(",")]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    rep = dict(ckpt=path, stuck_limit=mem.D.STUCK, levels=levels, starts=starts, when=time.strftime("%Y-%m-%d %H:%M"), modes={})
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=cs.worker_init, initargs=({},)) as ex:
        for m in modes:
            t0 = time.time()
            tasks = [dict(policy=path, mode=m, level=L, start=s, senses="fabius") for L in levels for s in starts]
            res = list(ex.map(episode, tasks, chunksize=1))
            rep["modes"][m] = dict(summary=summarize(res), seconds=round(time.time() - t0), attempts=res)
            print(m, rep["modes"][m]["summary"], rep["modes"][m]["seconds"], "s", flush=True)
            outp.write_text(json.dumps(rep, indent=1))
    if not a.no_agree:
        items, Y, P = coach_frames()
        for m in modes:
            rep["modes"][m]["coach_agreement"] = agreement(path, m, items, Y, P)
            print(m, "coach agreement", rep["modes"][m]["coach_agreement"], flush=True)
    if "none" in rep["modes"]:
        base = rep["modes"]["none"]["summary"]
        rep["stuck_rise"] = {m: rep["modes"][m]["summary"]["stuck"] - base["stuck"] for m in rep["modes"] if m != "none"}
    outp.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep.get("stuck_rise")), flush=True)


if __name__ == "__main__":
    main()
