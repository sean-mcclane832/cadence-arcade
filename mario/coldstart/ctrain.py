"""Cold-start night driver (COLDSTART.md). memory/mtrain.py's recipe (burn-in lessons, act-mode play, cold settles,
stationary memory guess for pile rows, DAgger), changed for a FRESH brain on 12 levels:

  - 12 training levels (1-1..3-4), equal lesson share per level; piles + ordered coach rows from coldstart/data
    (data/levels.json maps level -> folder; coach horizon per level in data/horizon.json, used by DAgger labels too).
  - Equal LESSONS per arm: the x-axis of every curve is lessons (chunk x batches), not hours.
  - SEL (24 starts) on all 12 levels after every chunk -> curve.csv (mean max x per level, flags). No HELD during
    the night (scored in the morning on fixed checkpoints), no auto-stop (no stall rule), stop at --stop-at.
  - Throttle: FIFO tickets for 2 shared worker slots while a game runs, with a game recheck INSIDE the wait (an arm
    that just released its slot queues behind the ones already waiting: no starvation).
  - 1 play worker per arm.

    python ctrain.py --name CatoA --init brains0/CatoA_ck0.npz --out runs/CatoA --stop-at 08:00
"""
from __future__ import annotations

import argparse, dataclasses, json, os, shutil, sys, time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402
import gamecheck  # noqa: E402  (memory/)
import seqcollect  # noqa: E402  (memory/)

mem = cs.mem
SEL, POOL = cs.SEL, cs.POOL
SLOTS = HERE / "runs" / "slots"
NSLOTS = 2


class Seq:
    """Ordered attempts of one level: screens, pressed buttons, labelled rows (as mtrain.Seq)."""

    def __init__(self):
        self.atts = []; self.rows = []

    def add(self, O, A, lab_idx, y, W):
        k = len(self.atts); self.atts.append((np.asarray(O, np.uint8), np.asarray(A, np.int64)))
        self.rows += [(k, int(i), int(l), float(w)) for i, l, w in zip(lab_idx, y, W)]

    def load(self, path):
        z = np.load(path); off = z["off"]
        for k in range(len(off) - 1):
            lo, hi = off[k], off[k + 1]
            m = (z["li"] >= lo) & (z["li"] < hi)
            self.add(z["O"][lo:hi], z["A"][lo:hi], z["li"][m] - lo, z["y"][m], z["W"][m])


def _live(p, age):
    try:
        return time.time() - p.stat().st_mtime < age
    except FileNotFoundError:
        return False


def slot_acquire(log, name):
    """FIFO: take a ticket, wait until it is among the oldest tickets that fit the free slots. Recheck the game every
    60 s inside the wait; returns None (and the caller turns the throttle off) when no game runs any more."""
    SLOTS.mkdir(parents=True, exist_ok=True)
    tk = SLOTS / f"t{time.time_ns():020d}_{name}.tk"; tk.write_text(str(os.getpid()))
    last_g = time.time(); waited = False
    try:
        while True:
            tk.touch()
            tickets = sorted(p for p in SLOTS.glob("t*.tk") if p == tk or _live(p, 600))  # untouched 10 min = dead
            locks = [SLOTS / f"slot{s}.lock" for s in range(NSLOTS)]
            for lp in locks:
                if lp.exists() and not _live(lp, 4 * 3600):
                    lp.unlink(missing_ok=True)
            free = [lp for lp in locks if not lp.exists()]
            rank = tickets.index(tk)
            if rank < len(free):
                try:
                    fd = os.open(free[rank], os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, name.encode()); os.close(fd)
                    if waited:
                        log(f"  got {free[rank].name}")
                    return free[rank]
                except FileExistsError:
                    pass
            if not waited:
                log(f"  game running: queued for one of {NSLOTS} worker slots (place {rank + 1})"); waited = True
            if time.time() - last_g > 60:
                last_g = time.time()
                if not gamecheck.gaming():
                    log("  game gone while waiting: throttle off"); return None
            time.sleep(10)
    finally:
        tk.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default=",".join(cs.TRAIN))
    ap.add_argument("--p-seq", type=float, default=0.5, help="share of ordered (burn-in) rows per lesson batch")
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--chunk", type=int, default=500, help="lesson batches (32 rows) per chunk")
    ap.add_argument("--max-chunks", type=int, default=1000); ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--crit", type=float, default=3.0)
    ap.add_argument("--dagger-every", type=int, default=4); ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--stop-at", default="08:00")
    ap.add_argument("--hours", type=float, default=24)
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    from cadence import Brain
    levels = args.levels.split(",")
    deep = cs.deep_levels()
    lv_dirs = json.loads((cs.DATA / "levels.json").read_text())
    ex = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=cs.worker_init, initargs=(deep,))
    state = dict(slow=False)

    def pmap(fn, tasks):
        return list(ex.map(fn, tasks, chunksize=1))

    piles, seqs = {}, {L: Seq() for L in levels}
    for L in levels:
        d = (HERE / lv_dirs[L]).resolve()
        z = np.load(d / "data.npz"); piles[L] = dict(X=z["X"], y=z["y"], W=z["W"].astype(np.float64))
        seqs[L].load(d / f"seq_{L}_coach.npz")
        log(f"data {L} ({lv_dirs[L]}, coach horizon {deep.get(L, 16)}): pile {len(piles[L]['y'])} rows, "
            f"ordered {len(seqs[L].rows)} rows in {len(seqs[L].atts)} attempts")
    b = mem.make_cold(Brain.load((HERE / args.init).resolve()))  # settles from rest, as the act-mode players do
    b.learner.config = dataclasses.replace(b.learner.config, eta=args.eta, eta_bias=args.eta / 10)
    rng = np.random.default_rng(11)
    deadline = time.time() + args.hours * 3600
    if args.stop_at:
        hh, mm = (int(v) for v in args.stop_at.split(":")); lt = time.localtime()
        st = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        if st < time.time():
            st += 86400
        deadline = min(deadline, st)
    best = {}; bestf = out / "best.json"
    curve = out / "curve.csv"
    if not curve.exists():
        curve.write_text("chunk,lessons,rows_taught,time,sel_mean_all,sel_flags_all," + ",".join(f"m{L},f{L}" for L in levels) + "\n")

    def summ(ev):
        return dict(x=[r["max_x"] for r in ev], flags=sum(r["flag"] for r in ev), mean=round(float(np.mean([r["max_x"] for r in ev]))))

    def consider(path, k):
        rec = {}
        tasks = [dict(policy=str(path), kind="act", level=L, start=s) for L in levels for s in SEL]
        res = pmap(mem.episode, tasks)
        for i, L in enumerate(levels):
            s = summ(res[i * len(SEL):(i + 1) * len(SEL)]); rec[L] = s
            cur = best.get(L)
            if cur is None or (s["flags"], s["mean"]) > (cur["sel_flags"], cur["sel_mean"]):
                shutil.copyfile(path, out / f"best_{L}.npz")
                best[L] = dict(ckpt=path.name, chunk=k, sel_flags=s["flags"], sel_mean=s["mean"])
        key = [sum(rec[L]["flags"] for L in levels), sum(rec[L]["mean"] for L in levels)]
        cur = best.get("joint")
        if cur is None or key > cur["key"]:
            shutil.copyfile(path, out / "best_joint.npz")
            best["joint"] = dict(ckpt=path.name, chunk=k, key=key)
        bestf.write_text(json.dumps(best, indent=1))
        mall = float(np.mean([rec[L]["mean"] for L in levels])); fall = sum(rec[L]["flags"] for L in levels)
        with open(curve, "a") as f:
            f.write(f"{k},{k * args.chunk},{k * args.chunk * 32},{time.time():.0f},{mall:.1f},{fall}," +
                    ",".join(f"{rec[L]['mean']},{rec[L]['flags']}" for L in levels) + "\n")
        return rec, key

    pile_p = {L: (lambda w: w / w.sum())(piles[L]["W"] * np.where(piles[L]["W"] > 1, args.crit, 1.0)) for L in levels}

    def seq_p(L):
        w = np.array([r[3] for r in seqs[L].rows]); w = w * np.where(w > 1, args.crit, 1.0); return w / w.sum()

    def batch_items(n, r2):
        share = np.ones(len(levels)) / len(levels)
        c = r2.multinomial(n, share); items, ys = [], []
        sp = {L: seq_p(L) for L in levels if seqs[L].rows}
        for L, kk in zip(levels, c):
            for _ in range(kk):
                if L in sp and r2.random() < args.p_seq:
                    a, i, y, _ = seqs[L].rows[int(r2.choice(len(sp[L]), p=sp[L]))]
                    O, A = seqs[L].atts[a]; items.append((O, A, i)); ys.append(y)
                else:
                    j = int(r2.choice(len(piles[L]["y"]), p=pile_p[L]))
                    items.append((piles[L]["X"][j:j + 1], np.zeros(0, np.int64), 0)); ys.append(int(piles[L]["y"][j]))
        return items, np.array(ys, np.int64)

    def lessons(n):
        r2 = np.random.default_rng(rng.integers(1 << 30))
        for _ in range(n):
            items, y = batch_items(32, r2)
            x = mem.prime(b, items, args.K)
            b.lesson(x, y)
        b.reset()

    def fit(n=320):
        r2 = np.random.default_rng(0); agree = []; agree_seq = []
        for _ in range(n // 32):
            items, y = batch_items(32, r2)
            x = mem.prime(b, items, args.K)
            a = np.asarray(b.act(x, greedy=True)).reshape(-1)
            agree.append(a == y); agree_seq.append(np.array([len(it[1]) > 0 for it in items]))
        b.reset()
        ag = np.concatenate(agree); sq = np.concatenate(agree_seq)
        return dict(all=round(float(ag.mean()), 3), ordered=round(float(ag[sq].mean()), 3) if sq.any() else None,
                    pile=round(float(ag[~sq].mean()), 3) if (~sq).any() else None)

    def dagger_round(k, pol):
        t = time.time(); tasks, tl = [], []
        for L in levels:
            for i, s in enumerate(rng.choice(POOL, args.episodes, replace=False)):
                tasks.append(dict(policy=str(pol), kind="act", level=L, start=int(s), keep_obs=True, label=True, seed=k * 100 + i)); tl.append(L)
        res = pmap(mem.episode, tasks)
        msg = []
        for L in levels:
            mine = [r for r, l in zip(res, tl) if l == L]
            atts = []
            for r in mine:
                seqs[L].add(r["obs"], r["actions"], r["lab_idx"], r["labels"], r["weights"])
                atts.append(dict(obs=r["obs"], actions=r["actions"], lab_idx=r["lab_idx"], labels=r["labels"], weights=r["weights"]))
            seqcollect.save_attempts(out / f"seq_{L}_d{k:04d}.npz", atts)
            msg.append(f"{L} +{sum(len(r['labels']) for r in mine)} x{[r['max_x'] for r in mine]}")
        log(f"  dagger ({time.time() - t:.0f}s): " + "; ".join(msg))

    p0 = out / f"{args.name}_ck0000.npz"; b.save(p0)
    log(f"start: {vars(args)} deep={deep} pid={os.getpid()}")
    rec0, _ = consider(p0, 0)
    log("chunk 0: " + " ".join(f"{L} {rec0[L]['flags']}/{len(SEL)} m{rec0[L]['mean']}" for L in levels))
    with open(out / "history.jsonl", "a") as f:
        f.write(json.dumps(dict(chunk=0, solo=rec0, t=time.time())) + "\n")
    last_check = 0; slot = None
    for k in range(1, args.max_chunks + 1):
        if time.time() - last_check > 600:
            g = gamecheck.gaming(); last_check = time.time()
            if g != state["slow"]:
                state["slow"] = g
                log(f"  throttle {'ON (game running): ' + str(NSLOTS) + ' FIFO worker slots shared by all arms' if g else 'OFF'}")
        if state["slow"]:
            slot = slot_acquire(log, args.name)
            if slot is None:
                state["slow"] = False; last_check = time.time()
        try:
            if args.dagger_every and k % args.dagger_every == 0:
                pol = out / "policy_dagger.npz"; b.save(pol); dagger_round(k, pol)
            t1 = time.time(); lessons(args.chunk); teach_s = round(time.time() - t1)
            p = out / f"{args.name}_ck{k:04d}.npz"; b.save(p)
            rec = dict(chunk=k, lessons=k * args.chunk, teach_s=teach_s, fit=fit())
            t2 = time.time(); rec["solo"], rec["key"] = consider(p, k); rec["sel_s"] = round(time.time() - t2); rec["t"] = time.time()
        finally:
            if slot is not None:
                slot.unlink(missing_ok=True); slot = None
        mall = np.mean([rec["solo"][L]["mean"] for L in levels])
        log(f"chunk {k} (teach {rec['teach_s']}s, SEL {rec['sel_s']}s, fit {rec['fit']}): mean {mall:.0f} flags "
            f"{rec['key'][0]} | " + " ".join(f"{L} {rec['solo'][L]['flags']}/m{rec['solo'][L]['mean']}" for L in levels))
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if time.time() > deadline:
            log("deadline"); break
    log("done. best: " + json.dumps(best))
    ex.shutdown()


if __name__ == "__main__":
    main()
