"""Coach coverage harness: the coach plays clean runs (no random buttons) from fixed starts, exactly as
coldstart/coachcollect.py's clean runs do (Game(..., menu="lean", bg_per_frame=True) like dagger.Body, stuck
limit 40, 900 decisions), but without collecting screens, so it is cheaper. Reports clears, max x, ends, and
seconds per coach call.

    python harness.py --coach base --levels 1-1,2-1 --out runs/cov_base --workers 4

World 8 is the final test (Sean, 2026-11-06): this harness refuses to run it. Worlds 1-7 are the coached training set.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "memory"))

STARTS = (0, 3, 8, 13, 20, 27)  # coachcollect.py's clean-run starts
TRAIN = [f"{w}-{s}" for w in (1, 2, 3) for s in (1, 2, 3, 4)]
STUCK, MAX_DECISIONS = int(os.environ.get("MARIO_STUCK", 225)), 900  # 225 decisions = 15 s (Sean, 2026-11-06; was 40). Earlier coverage numbers (72/72) were at 40.


def below_normal():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


def worker_init():
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    below_normal()


def run_one(task, cb=None):
    """One clean coach run. task: level 'W-S', start, coach name, opts (dict for the coach)."""
    import game
    import smartcoach
    L = task["level"]
    lvl = tuple(int(v) for v in L.split("-"))
    assert lvl[0] in (1, 2, 3, 4, 5, 6, 7), "World 8 is the final test: no coach runs there"
    g = game.Game(noop_start=int(task["start"]), level=lvl, menu="lean", bg_per_frame=True)
    e = g.env
    for a in task.get("prefix", []):  # replay a recorded action prefix (tests from a hard spot)
        # a coach call leaves a trace in the emulator (restore after a simulated frame is not the same as no
        # simulation: 2-1 drifts 1 px at decision 72 without it). One dummy simulated frame per decision
        # reproduces a coached trajectory exactly (checked: 300/300 decisions on 2-1).
        sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
        e._time_last, e._x_position_last, e.done = sv
        g.step(int(a))
    smartcoach.install(g, task.get("coach", "base"), **task.get("opts", {}))
    t0 = time.time(); calls = 0; csecs = 0.0; slow = []
    xs, acts = [], []
    while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
        t = time.time()
        lab, wt = g.coach_weighted()
        dt = time.time() - t; calls += 1; csecs += dt; slow.append(dt)
        if cb is not None:
            cb(g, lab, wt, dt)
        xs.append(int(g.x)); acts.append(int(lab))
        g.step(lab)
    end = "flag" if g.flag else ("died" if g.done else "stuck")
    r = dict(level=L, start=int(task["start"]), coach=task.get("coach", "base"), flag=bool(g.flag), end=end,
             max_x=int(g.max_x), decisions=int(g.decisions), secs=round(time.time() - t0, 1), calls=calls,
             coach_secs=round(csecs, 2), per_call=round(csecs / max(calls, 1), 4),
             p95_call=round(float(np.percentile(slow, 95)), 4) if slow else 0.0, xs=xs, acts=acts)
    sm = getattr(g, "_smart", None)
    if sm is not None:
        r.update(found=getattr(sm, "found", []), stalls=getattr(sm, "stalls", 0), cache_hits=getattr(sm, "cache_hits", 0),
                 fallbacks=getattr(sm, "fallbacks", 0))
    g.close()
    return r


def workers_allowed(want):
    try:
        import gamecheck
        if gamecheck.gaming():
            return 1
    except Exception:
        pass
    return min(want, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coach", default="base"); ap.add_argument("--levels", default=",".join(TRAIN))
    ap.add_argument("--starts", default=",".join(map(str, STARTS))); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--opts", default="{}")
    a = ap.parse_args()
    levels = a.levels.split(",")
    assert all(L.split("-")[0] in "1234567" for L in levels), "World 8 is held out (the final test)"
    out = (HERE / a.out); out.mkdir(parents=True, exist_ok=True)
    opts = json.loads(a.opts)
    nw = workers_allowed(a.workers)
    below_normal()
    tasks = [dict(level=L, start=int(s), coach=a.coach, opts=opts) for L in levels for s in a.starts.split(",")]
    print(f"{time.strftime('%H:%M:%S')} {len(tasks)} runs, coach {a.coach} {opts}, {nw} workers", flush=True)
    res = []
    t0 = time.time()
    # keep only as many runs in flight as allowed right now: 1 while a game runs, else --workers (max 4);
    # rechecked every 60 s, so the job speeds up when Sean stops playing and slows when he starts
    pending = list(tasks); inflight = {}; last = 0.0; allowed = nw
    from concurrent.futures import wait, FIRST_COMPLETED
    with ProcessPoolExecutor(min(a.workers, 4), mp_context=get_context("spawn"), initializer=worker_init) as ex:
        while pending or inflight:
            if time.time() - last > 60:
                last = time.time(); na = workers_allowed(a.workers)
                if na != allowed:
                    print(f"{time.strftime('%H:%M:%S')} workers {allowed} -> {na}", flush=True); allowed = na
            while pending and len(inflight) < allowed:
                t = pending.pop(0); inflight[ex.submit(run_one, t)] = t
            done, _ = wait(list(inflight), timeout=60, return_when=FIRST_COMPLETED)
            for f in done:
                inflight.pop(f); r = f.result(); res.append(r)
                print(f"{time.strftime('%H:%M:%S')} {r['level']} s{r['start']}: {r['end']} x{r['max_x']} dec {r['decisions']} "
                      f"{r['secs']}s {r['per_call'] * 1000:.0f} ms/call", flush=True)
                (out / "runs.json").write_text(json.dumps(res))
    summ = {}
    for L in levels:
        m = sorted([r for r in res if r["level"] == L], key=lambda r: r["start"])
        summ[L] = dict(clears=sum(r["flag"] for r in m), n=len(m), max_x=[r["max_x"] for r in m],
                       ends="".join(r["end"][0] for r in m),
                       per_call=round(sum(r["coach_secs"] for r in m) / max(sum(r["calls"] for r in m), 1), 4))
        print(f"{L}: {summ[L]['clears']}/{summ[L]['n']} {summ[L]['ends']} {summ[L]['max_x']} {summ[L]['per_call'] * 1000:.0f} ms/call")
    allc = sum(r["calls"] for r in res); alls = sum(r["coach_secs"] for r in res)
    summ["_all"] = dict(calls=allc, coach_secs=round(alls, 1), per_call=round(alls / max(allc, 1), 4), wall=round(time.time() - t0))
    print(f"all: {allc} calls, {alls / max(allc, 1) * 1000:.1f} ms/call, wall {time.time() - t0:.0f}s")
    (out / "summary.json").write_text(json.dumps(dict(coach=a.coach, opts=opts, summary=summ), indent=1))


if __name__ == "__main__":
    main()
