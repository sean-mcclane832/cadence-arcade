"""Night driver for the memory brains (MEMORY.md): csteady.py's recipe with burn-in lessons and act-mode play.

Every lesson batch (32 rows) mixes
  - pile rows (W1's own old piles, no order known): taught as the first moment of an attempt (all memories empty);
  - ordered rows (coach paths replayed in order + tonight's DAgger attempts, every screen kept): each row is
    primed with K burn-in acts on the K screens before it plus the exact heard history / efference copy, then
    `lesson` (which reads every memory the act reads).
Play (SEL, HELD, DAgger) is act mode: reset() at the attempt start, greedy acts, memories on.
Same driver for Cato / Cicero / Livy (burn-in for all, so compute is equal).

    python mtrain.py --name Cicero --init brains0/Cicero_ck0.npz --out runs/night/Cicero --workers 2 --stop-at 07:15
"""
from __future__ import annotations

import argparse, dataclasses, json, os, shutil, sys, time

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mem  # noqa: E402
import gamecheck  # noqa: E402
import seqcollect  # noqa: E402

C = mem.C
SEL, HELD, POOL = C.SEL, C.HELD, C.POOL
P4 = HERE.parent / "concepts" / "runs" / "p4"
PILES = {"1-1": P4 / "pile_1-1" / "data.npz", "1-2": P4 / "pile_1-2" / "data.npz", "1-3": P4 / "F2" / "data_1-3.npz",
         "1-4": P4 / "pile_1-4" / "data.npz"}  # exactly W1_F2ck110's lesson piles (F2 grew 1-3)
SLOTS = HERE / "runs" / "night" / "slots"


class Seq:
    """Ordered attempts of one level: screens, pressed buttons, labelled rows."""

    def __init__(self):
        self.atts = []  # list of (O, A)
        self.rows = []  # (attempt, i, y, w)

    def add(self, O, A, lab_idx, y, W):
        k = len(self.atts); self.atts.append((np.asarray(O, np.uint8), np.asarray(A, np.int64)))
        self.rows += [(k, int(i), int(l), float(w)) for i, l, w in zip(lab_idx, y, W)]

    def load(self, path):
        z = np.load(path); off = z["off"]
        for k in range(len(off) - 1):
            lo, hi = off[k], off[k + 1]
            m = (z["li"] >= lo) & (z["li"] < hi)
            self.add(z["O"][lo:hi], z["A"][lo:hi], z["li"][m] - lo, z["y"][m], z["W"][m])


def slot_acquire(log):
    SLOTS.mkdir(parents=True, exist_ok=True)
    waited = False
    while True:
        for s in range(2):
            p = SLOTS / f"slot{s}.lock"
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return p
            except FileExistsError:
                if time.time() - p.stat().st_mtime > 3 * 3600:
                    p.unlink(missing_ok=True)
        if not waited:
            log("  game running: waiting for one of 2 worker slots"); waited = True
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="runs/seq"); ap.add_argument("--levels", default="1-1,1-2,1-3,1-4")
    ap.add_argument("--mix", default="0.15,0.25,0.4,0.2", help="lesson share per level (F2's)")
    ap.add_argument("--p-seq", type=float, default=0.5, help="share of ordered (burn-in) rows per lesson batch")
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--chunks", type=int, default=1000); ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--crit", type=float, default=3.0)
    ap.add_argument("--dagger-every", type=int, default=6); ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--dagger-levels", default="1-1,1-2,1-3,1-4")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--stop-at", default=None)
    ap.add_argument("--hours", type=float, default=20); ap.add_argument("--stall-hours", type=float, default=3.0)
    args = ap.parse_args()
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    from cadence import Brain
    import numba
    levels = args.levels.split(",")
    ex = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=mem.worker_init)
    state = dict(slow=False)

    def pmap(fn, tasks):
        if not state["slow"]:
            return list(ex.map(fn, tasks, chunksize=1))
        return [r for i in range(len(tasks)) for r in ex.map(fn, tasks[i:i + 1])]  # one worker at a time

    piles = {}
    for L in levels:
        z = np.load(PILES[L]); piles[L] = dict(X=z["X"], y=z["y"], W=z["W"].astype(np.float64))
    seqs = {L: Seq() for L in levels}
    for L in levels:
        p = (HERE / args.seq / f"seq_{L}_coach.npz").resolve()
        if p.exists():
            seqs[L].load(p)
    for L in levels:
        log(f"data {L}: pile {len(piles[L]['y'])} rows, ordered {len(seqs[L].rows)} rows in {len(seqs[L].atts)} attempts")
    mix = dict(zip(levels, [float(v) for v in args.mix.split(",")]))
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
    best = {}; bestf = out / "best.json"; last_progress = [time.time()]

    def solo(path, L, starts):
        return pmap(mem.episode, [dict(policy=str(path), kind="act", level=L, start=s) for s in starts])

    def summ(ev):
        return dict(x=[r["max_x"] for r in ev], flags=sum(r["flag"] for r in ev), mean=round(float(np.mean([r["max_x"] for r in ev]))))

    def consider(path, k):
        rec = {}
        for L in levels:
            s = summ(solo(path, L, SEL)); rec[L] = s
            cur = best.get(L)
            if cur is None or (s["flags"], s["mean"]) > (cur["sel_flags"], cur["sel_mean"]):
                h = summ(solo(path, L, HELD))
                shutil.copyfile(path, out / f"best_{L}.npz")
                best[L] = dict(ckpt=path.name, chunk=k, sel_flags=s["flags"], sel_mean=s["mean"], held_flags=h["flags"], held_mean=h["mean"])
                bestf.write_text(json.dumps(best, indent=1)); last_progress[0] = time.time()
                log(f"  NEW BEST {L} ({path.stem}): SEL {s['flags']}/{len(SEL)} mean {s['mean']} | HELD {h['flags']}/50 mean {h['mean']}")
        key = [min(rec[L]["flags"] for L in levels), sum(rec[L]["flags"] for L in levels), sum(rec[L]["mean"] for L in levels)]
        cur = best.get("joint")
        if cur is None or key > cur["key"]:
            shutil.copyfile(path, out / "best_joint.npz")
            best["joint"] = dict(ckpt=path.name, chunk=k, key=key, sel=[rec[L]["flags"] for L in levels])
            bestf.write_text(json.dumps(best, indent=1)); last_progress[0] = time.time()
            log(f"  NEW JOINT BEST ({path.stem}): SEL {[rec[L]['flags'] for L in levels]}")
        return rec, key

    pile_p = {L: (lambda w: w / w.sum())(piles[L]["W"] * np.where(piles[L]["W"] > 1, args.crit, 1.0)) for L in levels}

    def seq_p(L):
        w = np.array([r[3] for r in seqs[L].rows]); w = w * np.where(w > 1, args.crit, 1.0); return w / w.sum()

    def batch_items(n, r2):
        share = np.array([mix[L] for L in levels]); share = share / share.sum()
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
        for L in args.dagger_levels.split(","):
            starts = rng.choice(POOL, args.episodes, replace=False)
            tasks = [dict(policy=str(pol), kind="act", level=L, start=int(s), keep_obs=True, label=True, seed=k * 100 + i)
                     for i, s in enumerate(starts)]
            t = time.time(); res = pmap(mem.episode, tasks)
            atts = []
            for r in res:
                seqs[L].add(r["obs"], r["actions"], r["lab_idx"], r["labels"], r["weights"])
                atts.append(dict(obs=r["obs"], actions=r["actions"], lab_idx=r["lab_idx"], labels=r["labels"], weights=r["weights"]))
            seqcollect.save_attempts(out / f"seq_{L}_d{k:04d}.npz", atts)
            ends = [r["end"][0] for r in res]
            log(f"  dagger {L}: +{sum(len(r['labels']) for r in res)} rows (ordered {len(seqs[L].rows)}) {time.time() - t:.0f}s, "
                f"f/d/s {ends.count('f')}/{ends.count('d')}/{ends.count('s')}, max x {[r['max_x'] for r in res]}")

    p0 = out / f"{args.name}_ck0000.npz"; b.save(p0)
    log(f"start: {vars(args)}")
    rec0, key0 = consider(p0, 0)
    log("chunk 0: " + " ".join(f"{L} {rec0[L]['flags']}/{len(SEL)} m{rec0[L]['mean']}" for L in levels))
    with open(out / "history.jsonl", "a") as f:
        f.write(json.dumps(dict(chunk=0, solo=rec0, key=key0, t=time.time())) + "\n")
    last_check = 0; slot = None
    for k in range(1, args.chunks + 1):
        if time.time() - last_check > 600:
            g = gamecheck.gaming(); last_check = time.time()
            if g != state["slow"]:
                state["slow"] = g; numba.set_num_threads(1 if g else int(os.environ["NUMBA_NUM_THREADS"]))
                log(f"  throttle {'ON (game running): 2 worker slots shared by all arms, 1 thread' if g else 'OFF'}")
        if state["slow"]:
            slot = slot_acquire(log)
        try:
            if args.dagger_every and k % args.dagger_every == 0:
                pol = out / "policy_dagger.npz"; b.save(pol); dagger_round(k, pol)
            t = time.time(); lessons(args.chunk); teach_s = round(time.time() - t)
            p = out / f"{args.name}_ck{k:04d}.npz"; b.save(p)
            rec = dict(chunk=k, teach_s=teach_s, fit=fit())
            rec["solo"], rec["key"] = consider(p, k); rec["t"] = time.time()
        finally:
            if slot is not None:
                slot.unlink(missing_ok=True); slot = None
        log(f"chunk {k} ({rec['teach_s']}s, fit {rec['fit']}): " + " ".join(f"{L} {rec['solo'][L]['flags']}/{len(SEL)} m{rec['solo'][L]['mean']}" for L in levels))
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if time.time() > deadline:
            log("deadline"); break
        if time.time() - last_progress[0] > args.stall_hours * 3600:
            log(f"no new SEL best (any level or joint) in {args.stall_hours} h: stopping (3 h no-progress rule)"); break
    log("done. best: " + json.dumps(best))
    ex.shutdown()


if __name__ == "__main__":
    main()
