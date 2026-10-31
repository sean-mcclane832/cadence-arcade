"""steady.py for the concepts run: chunks of lessons, every checkpoint saved and solo-tested on SEL, best per
level -> HELD, joint best, tie-aware DAgger. Rows are stored with the full sense set (ramscreen) and sliced to
what the brain reads (--senses body | ram | ramscreen).

Phase 2 extras:
  --clinic      death clinic: each DAgger round also rewinds the current checkpoint's SEL deaths/stucks to 1, 2
                and 4 s (15 / 30 / 60 decisions) before the end and lets the brain play on from there while the
                coach labels (the usual DAgger labelling).
  --survivals   keep the brain's OWN buttons from those retries when they get past the spot it died at
                (>= 16 px further), from the rewind point to that spot; sampled at weight 0.5, at most
                --own-cap (0.2) of each lesson.

    python csteady.py --out runs/p1/r1 --fresh --senses ram --data 1-1=runs/p1/coach_1-1,... --levels 1-1,1-2,1-3,1-4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cplay as C  # noqa: E402

SEL, TEST, WIDE, HELD, POOL = C.SEL, C.TEST, C.WIDE, C.HELD, C.POOL
GAMES = ("hoi4.exe", "javaw.exe")


def gaming():
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=20).stdout.lower()
        return any(g in out for g in GAMES)
    except Exception:
        return False


class Runner:
    def __init__(self, workers):
        self.ex = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=C.worker_init)
        self.workers = workers; self.slow = False

    def map(self, tasks):
        if not self.slow:
            return list(self.ex.map(C.episode, tasks, chunksize=1))
        res = []
        for i in range(0, len(tasks), 2):
            res += list(self.ex.map(C.episode, tasks[i:i + 2]))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-brain", default=None); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--senses", default="ram")
    ap.add_argument("--data", required=True)
    ap.add_argument("--practice", default=None); ap.add_argument("--mix", default=None)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--chunk", type=int, default=250); ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--hours", type=float, default=3); ap.add_argument("--stop-at", default=None, help="HH:MM local hard stop")
    ap.add_argument("--eta", type=float, default=0.001); ap.add_argument("--crit", type=float, default=3.0)
    ap.add_argument("--dagger-every", type=int, default=6); ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--from-start", type=float, default=0.67)
    ap.add_argument("--last-all", type=int, default=60)
    ap.add_argument("--clinic", action="store_true"); ap.add_argument("--clinic-max", type=int, default=6)
    ap.add_argument("--survivals", action="store_true"); ap.add_argument("--own-cap", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    from cadence import Brain
    R = Runner(args.workers)
    levels = args.levels.split(",")
    S = args.senses
    data = {}
    for item in args.data.split(","):
        L, d = item.split("="); src = (HERE / d).resolve(); z = np.load(src / "data.npz")
        data[L] = dict(X=z["X"], y=z["y"], W=z["W"].astype(np.float64), src=src,
                       paths=[tuple(c) for c in json.loads((src / "coach_paths.json").read_text())])
        if (src / "own.npz").exists():
            o = np.load(src / "own.npz"); data[L]["oX"], data[L]["oy"] = o["X"], o["y"]
        else:
            data[L]["oX"] = np.zeros((0, z["X"].shape[1]), np.uint8); data[L]["oy"] = np.zeros(0, np.int64)
        log(f"data {L}: {len(z['y'])} rows ({int((z['W'] > 1).sum())} critical) own {len(data[L]['oy'])} from {d}")
    widths = {data[L]["X"].shape[1] for L in data}; assert len(widths) == 1, widths
    STORE = {2000: "body", C.n_inputs("ramscreen"): "ramscreen"}[widths.pop()]
    log(f"rows stored as {STORE}, brain reads {S}")
    practice = args.practice.split(",") if args.practice else list(data)
    mix = dict(zip(data, [float(v) for v in args.mix.split(",")] if args.mix else [1.0] * len(data)))
    n_in = C.n_inputs(S)
    if args.fresh:
        b = Brain.compose(n_in, 7, modules=(args.width,), seed=0, teaching="normalized", learning_surprise=False)
        log(f"fresh brain: {n_in} inputs ({S}), one region of {args.width}")
    else:
        b = Brain.load((HERE / args.init_brain).resolve())

    def set_eta(e):
        c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=e, eta_bias=e / 10)

    set_eta(args.eta)
    rng = np.random.default_rng(11)
    best = {L: None for L in levels}
    bestf = out / "best.json"
    deadline = time.time() + args.hours * 3600
    if args.stop_at:
        hh, mm = (int(v) for v in args.stop_at.split(":")); lt = time.localtime()
        st = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        if st < time.time():
            st += 86400
        deadline = min(deadline, st)
    last_solo = {}

    def solo(path, L, starts):
        return R.map([C.task(path, S, L, s) for s in starts])

    def summ(ev):
        return dict(x=[r["max_x"] for r in ev], flags=sum(r["flag"] for r in ev),
                    mean=round(float(np.mean([r["max_x"] for r in ev]))))

    def consider(path, tag, k):
        rec = {}
        for L in levels:
            ev = solo(path, L, SEL); last_solo[L] = ev
            s = summ(ev); rec[L] = s
            cur = best.get(L)
            if cur is None or (s["flags"], s["mean"]) > (cur["sel_flags"], cur["sel_mean"]):
                h = summ(solo(path, L, HELD))
                shutil.copyfile(path, out / f"best_{L}.npz")
                best[L] = dict(ckpt=path.name, chunk=k, sel_flags=s["flags"], sel_mean=s["mean"], held_flags=h["flags"],
                               held_mean=h["mean"])
                bestf.write_text(json.dumps(best, indent=1))
                log(f"  NEW BEST {L} ({tag}): SEL {s['flags']}/{len(SEL)} mean {s['mean']} | HELD {h['flags']}/50 mean {h['mean']}")
        if len(levels) > 1:
            key = [min(rec[L]["flags"] for L in levels), sum(rec[L]["flags"] for L in levels), sum(rec[L]["mean"] for L in levels)]
            cur = best.get("joint")
            if cur is None or key > cur["key"]:
                shutil.copyfile(path, out / "best_joint.npz")
                best["joint"] = dict(ckpt=path.name, chunk=k, key=key, sel=[rec[L]["flags"] for L in levels])
                bestf.write_text(json.dumps(best, indent=1))
                log(f"  NEW JOINT BEST ({tag}): SEL {[rec[L]['flags'] for L in levels]}")
        return rec

    def xs_of(X):
        return C.view(X, STORE, S).astype(np.float64) / 255.0

    def lessons(n):
        Ls = [L for L in data if L in practice or mix[L] > 0]
        share = np.array([mix[L] for L in Ls]); share = share / share.sum()
        n_own = sum(len(data[L]["oy"]) for L in Ls); n_coach = sum(len(data[L]["y"]) for L in Ls)
        own_share = min(args.own_cap, 0.5 * n_own / (0.5 * n_own + n_coach)) if n_own else 0.0
        oX = np.concatenate([data[L]["oX"] for L in Ls]) if n_own else None
        oy = np.concatenate([data[L]["oy"] for L in Ls]) if n_own else None
        ps = {L: data[L]["W"] * np.where(data[L]["W"] > 1, args.crit, 1.0) for L in Ls}
        ps = {L: v / v.sum() for L, v in ps.items()}
        r2 = np.random.default_rng(rng.integers(1 << 30))
        for _ in range(n):
            k_own = int(r2.binomial(32, own_share)) if n_own else 0
            c = r2.multinomial(32 - k_own, share)
            xs, ys = [], []
            for L, kk in zip(Ls, c):
                if kk:
                    i = rng.choice(len(data[L]["y"]), kk, p=ps[L]); xs.append(data[L]["X"][i]); ys.append(data[L]["y"][i])
            if k_own:
                i = rng.choice(len(oy), k_own); xs.append(oX[i]); ys.append(oy[i])
            b.learner.step(b.stimulus(xs_of(np.concatenate(xs)), memory=False), np.concatenate(ys))
        return own_share

    def fit(L, n=1500):
        d = data[L]; i = np.random.default_rng(0).choice(len(d["y"]), min(n, len(d["y"])), replace=False)
        pr = np.concatenate([np.asarray(b.predict(xs_of(d["X"][i[k:k + 500]]))).reshape(-1) for k in range(0, len(i), 500)])
        c = d["W"][i] > 1
        return round(float((pr == d["y"][i]).mean()), 3), round(float((pr[c] == d["y"][i][c]).mean()), 3)

    def dagger_round(k, pol_path):
        for L in practice:
            d = data[L]; tasks = []; meta = []
            n3 = int(round(args.episodes * args.from_start))
            base = dict(senses=STORE, pol_senses=S, last_all=args.last_all, tiew=True)
            for i, s in enumerate(rng.choice(POOL, n3)):
                tasks.append(C.task(pol_path, STORE, L, int(s), mode="dagger", seed=k * 100 + i, **{kk: v for kk, v in base.items() if kk != "senses"}))
                meta.append(None)
            for i in range(args.episodes - n3):
                if not d["paths"]:
                    break
                s0, path = d["paths"][int(rng.integers(len(d["paths"])))]
                j = int(rng.integers(0, max(1, len(path) - 20)))
                tasks.append(C.task(pol_path, STORE, L, s0, mode="dagger", prefix=path[:j], seed=k * 100 + 50 + i,
                                    **{kk: v for kk, v in base.items() if kk != "senses"}))
                meta.append(None)
            if args.clinic and L in last_solo:
                fails = [r for r in last_solo[L] if not r["flag"]]
                rng.shuffle(fails)
                for r in fails[:args.clinic_max]:
                    n = len(r["actions"])
                    for back in (15, 30, 60):
                        tasks.append(C.task(pol_path, STORE, L, r["start"], mode="dagger", prefix=r["actions"][:max(0, n - back)],
                                            seed=k * 100 + 80 + back, keep_obs=args.survivals,
                                            **{kk: v for kk, v in base.items() if kk != "senses"}))
                        meta.append(dict(death_x=r["max_x"], back=back))
            t = time.time(); res = R.map(tasks)
            keep = [r for r in res if len(r["labels"])]
            W = np.concatenate([r["weights"] for r in keep]).astype(np.float64)
            d["X"] = np.concatenate([d["X"]] + [r["obs"] for r in keep]); d["y"] = np.concatenate([d["y"]] + [r["labels"] for r in keep])
            d["W"] = np.concatenate([d["W"], W])
            surv = 0; cl = [(m, r) for m, r in zip(meta, res) if m is not None]
            if args.survivals:
                for m, r in cl:
                    goal = m["death_x"] + 16
                    if r["max_x"] >= goal and len(r.get("own_obs", [])):
                        j = next(i for i, x in enumerate(r["xs"] + [10 ** 9]) if x >= goal)
                        j = min(j + 1, len(r["actions"]))
                        if j:
                            d["oX"] = np.concatenate([d["oX"], r["own_obs"][:j]]); d["oy"] = np.concatenate([d["oy"], np.array(r["actions"][:j], np.int64)])
                            surv += 1
            np.savez(out / f"data_{L}.npz", X=d["X"], y=d["y"], W=d["W"])
            if len(d["oy"]):
                np.savez(out / f"own_{L}.npz", X=d["oX"], y=d["oy"])
            ends = [r["end"][0] for r in res if True]
            clinic_txt = ""
            if cl:
                passed = sum(r["max_x"] >= m["death_x"] + 16 for m, r in cl)
                clinic_txt = f", clinic {len(cl)} retries passed {passed}, survivals kept {surv} (own rows {len(d['oy'])})"
            log(f"  dagger {L}: +{len(W)} rows ({len(d['y'])}) {time.time() - t:.0f}s, f/d/s {ends.count('f')}/{ends.count('d')}/{ends.count('s')}{clinic_txt}")

    import numba
    p0 = out / "ck_0000.npz"; b.save(p0)
    log(f"start: {vars(args)}")
    rec0 = consider(p0, "ck_0000", 0)
    log("chunk 0: " + " ".join(f"{L} {rec0[L]['flags']}/{len(SEL)} m{rec0[L]['mean']}" for L in levels))
    last_check = time.time()
    for k in range(1, args.chunks + 1):
        if time.time() - last_check > 600 or k == 1:
            g = gaming(); last_check = time.time()
            if g != R.slow:
                R.slow = g; numba.set_num_threads(2 if g else 8)
                log(f"  throttle {'ON (game running): 2 games, 2 threads' if g else 'OFF'}")
        if args.dagger_every and k % args.dagger_every == 0:
            pol = out / "policy_dagger.npz"; b.save(pol); dagger_round(k, str(pol))
        t = time.time(); own_share = lessons(args.chunk)
        p = out / f"ck_{k:04d}.npz"; b.save(p)
        rec = dict(chunk=k, teach_s=round(time.time() - t), own_share=round(own_share, 3), fit={L: fit(L) for L in data})
        rec["solo"] = consider(p, p.stem, k)
        log(f"chunk {k} ({rec['teach_s']}s, own {rec['own_share']}, fit {rec['fit']}): " +
            " ".join(f"{L} {rec['solo'][L]['flags']}/{len(SEL)} m{rec['solo'][L]['mean']}" for L in levels))
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if time.time() > deadline:
            log("deadline"); break
    log("done. best: " + json.dumps(best))


if __name__ == "__main__":
    main()
