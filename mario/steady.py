"""Fragility work (LEARNING_FIX.md E10+): teach in small chunks, solo-test after EVERY chunk,
save every checkpoint, keep a best-so-far brain per level, optional DAgger rounds, weight
averaging, critical-row up-weighting and an eta schedule.

    python steady.py --out runs/st_x --init-brain runs/fix_cad_body3/policy_r02.npz \
        --data 1-1=runs/fix_cad_body3 --levels 1-1 --chunk 250 --chunks 40

Selection uses SEL starts (idle frames 0,3,...,27: inside the DAgger jitter range 0-31).
"Consistent" is judged on TEST starts 32-41, which no training or selection ever used.
Best per level = most SEL flags, then mean x; on a new best the TEST starts are played too.
Throttle: if hoi4.exe or javaw.exe runs, at most 2 games at once and 2 numba threads, no GPU.
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
sys.path.insert(0, str(HERE))
import dagger as D  # noqa: E402

SEL = tuple(s for s in range(0, 80, 4) if not 32 <= s <= 41)  # 18 starts, all inside the DAgger pool
TEST = tuple(range(32, 42))
WIDE = tuple(range(100, 140, 2))  # late jitters, reported on every new best (never selected on)
HELD = TEST + tuple(range(80, 100)) + WIDE  # 50 held-out starts: never in training or selection
POOL = np.array([s for s in range(80) if s not in TEST])  # DAgger jitters: never a TEST start
GAMES = ("hoi4.exe", "javaw.exe")


def gaming() -> bool:
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=20).stdout.lower()
        return any(g in out for g in GAMES)
    except Exception:
        return False


def lv(s):
    return tuple(int(v) for v in s.split("-"))


def average(paths, out):
    """Mean of the learned arrays of several checkpoints of ONE brain (same wiring)."""
    ds = [np.load(p, allow_pickle=True) for p in paths]
    keep = {k: ds[0][k] for k in ds[0].files}
    for k in ("efficacy", "bias", "log_gain"):
        keep[k] = np.mean([d[k] for d in ds], axis=0)
    tmp = str(out) + ".tmp.npz"
    np.savez(tmp, **keep); os.replace(tmp, out)


class Runner:
    def __init__(self, workers):
        self.ex = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=D._worker_init)
        self.workers = workers
        self.slow = False

    def map(self, tasks):
        lim = 2 if self.slow else self.workers
        res = []
        for i in range(0, len(tasks), lim):
            res += list(self.ex.map(D.episode, tasks[i:i + lim]))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-brain", required=True)
    ap.add_argument("--data", required=True, help="comma list LEVEL=RUN_DIR (data.npz + coach_paths.json)")
    ap.add_argument("--practice", default=None, help="comma list of levels taught (default: all in --data)")
    ap.add_argument("--mix", default=None, help="comma list of lesson shares per --data level (default equal)")
    ap.add_argument("--levels", required=True, help="levels solo-tested after every chunk")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--hours", type=float, default=99)
    ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--eta-end", type=float, default=None, help="geometric eta schedule to this value")
    ap.add_argument("--crit", type=float, default=1.0, help="multiply the weight of critical rows")
    ap.add_argument("--avg", type=int, default=0, help="also test the mean of the last K checkpoints")
    ap.add_argument("--dagger-every", type=int, default=0, help="DAgger round every N chunks (0 = never)")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--from-start", type=float, default=1 / 3, help="share of DAgger attempts from the level start")
    ap.add_argument("--cost-weight", action="store_true", help="DAgger rows where the brain's own button loses get the critical weight")
    ap.add_argument("--tie-own", action="store_true", help="DAgger rows where the brain's button ties the coach's best keep the brain's button")
    ap.add_argument("--last-all", type=int, default=60)
    ap.add_argument("--ratchet", action="store_true", help="keep a chunk only if the practised level's SEL (flags, mean) did not get worse; else go back")
    ap.add_argument("--keep-min", type=int, default=0, help="ratchet: other levels must keep >= this many SEL flags")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-only", nargs="*", default=None, help="just test these brains on SEL+TEST")
    ap.add_argument("--extra", default="", help="eval-only: also play these starts, e.g. 80-99")
    ap.add_argument("--stop-flags", type=int, default=0, help="stop when every level has >= this many TEST flags")
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    from game import MENUS
    mb = sorted({a for p in MENUS["lean"] for a in p})
    base = dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb)
    R = Runner(args.workers)
    levels = args.levels.split(",")

    def solo(path, level, starts):
        ts = [dict(base, level=lv(level), mode="eval", start=s, policy=str(path)) for s in starts]
        return R.map(ts)

    def summ(ev):
        return dict(x=[r["max_x"] for r in ev], flags=sum(r["flag"] for r in ev),
                    mean=round(float(np.mean([r["max_x"] for r in ev]))))

    if args.eval_only is not None:
        res = {}
        for p in args.eval_only:
            for L in levels:
                s, t = summ(solo((HERE / p).resolve(), L, SEL)), summ(solo((HERE / p).resolve(), L, TEST))
                hh = summ(solo((HERE / p).resolve(), L, HELD))
                res[f"{p}|{L}"] = dict(sel=s, test=t, held=hh)
                log(f"   HELD {hh['flags']}/50 mean {hh['mean']}")
                if args.extra:
                    a_, b_ = (int(v) for v in args.extra.split("-"))
                    e = summ(solo((HERE / p).resolve(), L, range(a_, b_ + 1))); res[f"{p}|{L}"]["extra"] = e
                    log(f"   extra {args.extra}: {e['flags']}/{b_ - a_ + 1} mean {e['mean']}")
                log(f"{p} {L}: SEL {s['flags']}/{len(SEL)} mean {s['mean']} | TEST {t['flags']}/10 mean {t['mean']} x={t['x']}")
        (out / "eval_only.json").write_text(json.dumps(res)); return

    from cadence import Brain
    data = {}
    for item in args.data.split(","):
        L, d = item.split("="); src = (HERE / d).resolve(); z = np.load(src / "data.npz")
        W = z["W"].astype(np.float64).copy(); W[W > 1] *= args.crit
        data[L] = dict(X=z["X"], y=z["y"], W=W, src=src,
                       paths=[tuple(c) for c in json.loads((src / "coach_paths.json").read_text())])
        log(f"data {L}: {len(z['y'])} rows ({int((z['W'] > 1).sum())} critical, x{args.crit}) from {d}")
    practice = args.practice.split(",") if args.practice else list(data)
    mix = [float(v) for v in args.mix.split(",")] if args.mix else [1.0] * len(data)
    mix = dict(zip(data, mix))
    b = Brain.load((HERE / args.init_brain).resolve())

    def set_eta(e):
        c = b.learner.config; b.learner.config = dataclasses.replace(c, eta=e, eta_bias=e / 10)

    set_eta(args.eta)
    rng = np.random.default_rng(7)
    best = {L: None for L in levels}
    bestf = out / "best.json"
    if bestf.exists():
        best.update(json.loads(bestf.read_text()))
    hist = []
    deadline = time.time() + args.hours * 3600
    ckpts = []
    lineage = None

    def consider(path, tag, chunk_i):
        rec = {}
        for L in levels:
            s = summ(solo(path, L, SEL)); rec[L] = s
            cur = best.get(L)
            key = (s["flags"], s["mean"])
            if cur is None or key > (cur["sel_flags"], cur["sel_mean"]):
                t = summ(solo(path, L, TEST)); w = summ(solo(path, L, WIDE)); h80 = summ(solo(path, L, range(80, 100)))
                held = t["flags"] + w["flags"] + h80["flags"]
                dst = out / f"best_{L}.npz"; shutil.copyfile(path, dst)
                best[L] = dict(ckpt=str(path.name), tag=tag, chunk=chunk_i, sel_flags=s["flags"], sel_mean=s["mean"],
                               test_flags=t["flags"], test_mean=t["mean"], test_x=t["x"], wide_flags=w["flags"], held_flags=held)
                bestf.write_text(json.dumps(best, indent=1))
                log(f"  NEW BEST {L} ({tag}): SEL {s['flags']}/{len(SEL)} mean {s['mean']} | TEST {t['flags']}/10 mean {t['mean']} | WIDE {w['flags']}/20 | HELD {held}/50")
                rec[L + "_test"] = t
        if len(levels) > 1:
            key = [min(rec[L]["flags"] for L in levels), sum(rec[L]["flags"] for L in levels), sum(rec[L]["mean"] for L in levels)]
            cur = best.get("joint")
            if cur is None or key > cur["key"]:
                tt = {L: summ(solo(path, L, TEST))["flags"] for L in levels}
                shutil.copyfile(path, out / "best_joint.npz")
                best["joint"] = dict(ckpt=str(path.name), key=key, test=tt); bestf.write_text(json.dumps(best, indent=1))
                log(f"  NEW JOINT BEST ({tag}): SEL {[rec[L]['flags'] for L in levels]} TEST {list(tt.values())}")
        return rec

    def lessons(n):
        Ls = [L for L in data if L in practice or mix[L] > 0]
        share = np.array([mix[L] for L in Ls]); share = share / share.sum()
        counts = np.random.default_rng(rng.integers(1 << 30)).multinomial(32, share, size=n)
        ps = {L: data[L]["W"] / data[L]["W"].sum() for L in Ls}
        for c in counts:
            xs, ys = [], []
            for L, k in zip(Ls, c):
                if k:
                    i = rng.choice(len(data[L]["y"]), k, p=ps[L]); xs.append(data[L]["X"][i]); ys.append(data[L]["y"][i])
            x = np.concatenate(xs).astype(np.float64) / 255.0
            b.learner.step(b.stimulus(x, memory=False), np.concatenate(ys))

    def fit(L, n=1500):
        d = data[L]; i = np.random.default_rng(0).choice(len(d["y"]), min(n, len(d["y"])), replace=False)
        pr = np.concatenate([np.asarray(b.predict(d["X"][i[k:k + 500]] / 255.0)).reshape(-1) for k in range(0, len(i), 500)])
        c = d["W"][i] > 1
        return round(float((pr == d["y"][i]).mean()), 3), round(float((pr[c] == d["y"][i][c]).mean()), 3)

    def dagger_round(k):
        pol = out / "policy_dagger.npz"; b.save(pol)
        res_all = []
        for L in practice:
            d = data[L]; tasks = []
            n3 = int(round(args.episodes * args.from_start))
            for i, s in enumerate(rng.choice(POOL, n3)):
                tasks.append(dict(base, level=lv(L), mode="dagger", start=int(s), policy=str(pol), seed=k * 100 + i,
                                  last_all=args.last_all, costw=args.cost_weight, tiew=args.tie_own))
            for i in range(args.episodes - n3):
                s0, path = d["paths"][int(rng.integers(len(d["paths"])))]
                j = int(rng.integers(0, max(1, len(path) - 20)))
                tasks.append(dict(base, level=lv(L), mode="dagger", start=s0, prefix=path[:j], policy=str(pol),
                                  seed=k * 100 + 50 + i, last_all=args.last_all, costw=args.cost_weight, tiew=args.tie_own))
            t = time.time(); res = [r for r in R.map(tasks) if len(r["labels"])]
            W = np.concatenate([r["weights"] for r in res]).astype(np.float64); W[W > 1] *= args.crit
            d["X"] = np.concatenate([d["X"]] + [r["obs"] for r in res]); d["y"] = np.concatenate([d["y"]] + [r["labels"] for r in res])
            d["W"] = np.concatenate([d["W"], W])
            raw = d["W"].copy(); raw[raw > 1] /= args.crit
            np.savez(out / f"data_{L}.npz", X=d["X"], y=d["y"], W=raw)
            ends = [r["end"][0] for r in res]
            log(f"  dagger {L}: +{len(W)} rows ({len(d['y'])}) {time.time() - t:.0f}s, attempts f/d/s "
                f"{ends.count('f')}/{ends.count('d')}/{ends.count('s')}")
            res_all += res
        return res_all

    import numba
    p0 = out / "ck_0000.npz"; b.save(p0)
    log(f"start: brain {args.init_brain}, eta {args.eta}->{args.eta_end or args.eta}, chunk {args.chunk} x {args.chunks}, "
        f"crit x{args.crit}, avg {args.avg}, dagger every {args.dagger_every}, practice {practice}, mix {mix}")
    hist.append(dict(chunk=0, solo=consider(p0, "ck_0000", 0)))
    ckpts.append(p0)
    if args.ratchet:
        P0 = practice[0]; lineage = ((hist[-1]["solo"][P0]["flags"], hist[-1]["solo"][P0]["mean"]), p0,
                   {L: hist[-1]["solo"][L]["flags"] for L in levels})
    log(f"chunk 0: " + " ".join(f"{L} {hist[-1]['solo'][L]['flags']}/{len(SEL)} m{hist[-1]['solo'][L]['mean']}" for L in levels))
    for k in range(1, args.chunks + 1):
        g = gaming()
        if g != R.slow:
            R.slow = g; numba.set_num_threads(2 if g else min(8, numba.config.NUMBA_NUM_THREADS))
            log(f"  throttle {'ON (game running): 2 games, 2 threads' if g else 'OFF'}")
        if args.eta_end:
            set_eta(args.eta * (args.eta_end / args.eta) ** ((k - 1) / max(1, args.chunks - 1)))
        if args.dagger_every and k % args.dagger_every == 0:
            dagger_round(k)
        t = time.time(); lessons(args.chunk)
        p = out / f"ck_{k:04d}.npz"; b.save(p); ckpts.append(p)
        rec = dict(chunk=k, eta=b.learner.config.eta, teach_s=round(time.time() - t),
                   fit={L: fit(L) for L in data}, solo=consider(p, p.stem, k))
        line = " ".join(f"{L} {rec['solo'][L]['flags']}/{len(SEL)} m{rec['solo'][L]['mean']}" for L in levels)
        if args.avg and len(ckpts) >= args.avg:
            pa = out / f"avg_{k:04d}.npz"; average(ckpts[-args.avg:], pa)
            rec["avg"] = consider(pa, pa.stem, k)
            line += " | avg " + " ".join(f"{L} {rec['avg'][L]['flags']}/{len(SEL)} m{rec['avg'][L]['mean']}" for L in levels)
        if args.ratchet:
            P = practice[0]
            kk = (rec["solo"][P]["flags"], rec["solo"][P]["mean"])
            okk = all(rec["solo"][L]["flags"] >= min(args.keep_min, lineage[2].get(L, 0)) for L in levels if L != P)
            if kk >= lineage[0] and okk:
                lineage = (kk, p, {L: rec["solo"][L]["flags"] for L in levels})
                line_tag = "kept"
            else:
                e_ = b.learner.config.eta; b = Brain.load(lineage[1]); set_eta(e_); line_tag = f"back to {lineage[1].stem}"
            rec["ratchet"] = line_tag
        log(f"chunk {k} ({rec['teach_s']}s, fit {rec['fit']}): {line}")
        if args.ratchet:
            log(f"  ratchet: {rec['ratchet']}")
        hist.append(rec)
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if args.stop_flags and all(best[L] and best[L]["test_flags"] >= args.stop_flags for L in levels):
            log("every level at the stop TEST flags; stopping"); break
        if time.time() > deadline:
            log("deadline"); break
    log("done. best: " + json.dumps(best))


if __name__ == "__main__":
    main()
