"""B1: advantage-weighted self-imitation through the normalized teaching learner (COACHFREE.md).
Coach-free: the brain's own sampled attempts on a dev level (worlds 6-7) are the only signal there. No coach call can
run in this process (cf.no_coach). World 4 is refused (cf.check_level).

One process = one arm, played serially (no pool). Arms:
  b1      practice -> own good segments -> lessons (own rows + worlds 1-3 anchor rows)
  anch    same practice and lesson count, lessons are anchor rows only (drift control: what lessons alone do to L)
  rand    same as b1 but own rows are ALL decisions of ALL attempts, uniform (is the advantage filter doing the work?)
  reload  practice only, frozen brain, no lessons (the sampled-attempt curve of the unchanged brain)

Per round: `attempts` sampled attempts from POOL starts (same starts + sampling seeds in every arm), then `lessons`
batches of 32 rows (b1/rand: `own` own rows + 32-own anchors; anch: 32 anchors). Greedy eval on the 50 HELD starts of
L every `eval_every` rounds. Forgetting check at the end: SEL (24 starts) on the 12 training levels, start vs end.

Sampling: greedy settle, then a draw from the motor softmax at the calibrated temperature T (calib.json: T with mean
P(greedy action) = target on the start brain's greedy path, POOL starts). The draw is ours, so no eligibility phases
run (no actor learning, about 3x cheaper than act(temperature=)). Cato only: brains with an efference copy would
remember the greedy button, not the pressed one.

    python b1.py calib --level 6-1 --ckpt CatoA
    python b1.py run --arm b1 --level 6-1 --ckpt CatoA --out runs/pilot_6-1/b1
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf  # noqa: E402

mem = cf.mem
LOCK = HERE / "runs" / "gameslot.lock"


class SPlayer:
    """Act-mode player on an in-memory brain: greedy settle, then our own softmax draw at T (T=0: greedy)."""

    def __init__(self, b, T=0.0, rec=None):
        self.b, self.T, self.rec = b, float(T), rec
        self.xkey = False
        assert b.efference is None, "SPlayer is for brains without an efference copy (Cato)"

    def begin(self, seed=0):
        self.b.reset(); self.rng = np.random.default_rng(seed)

    def act(self, o):
        x = o[None].astype(np.float64)
        a = int(np.asarray(self.b.act(x, greedy=True)).reshape(-1)[0])
        if self.rec is not None or self.T > 0:
            bg = self.b.basal_ganglia
            if self.rec is not None:
                self.rec.append(bg.probabilities(bg._free, 1.0)[0].copy())  # T=1 probs; rescale later
            if self.T > 0:
                p = np.asarray(bg.probabilities(bg._free, self.T))[0]
                a = int(min((p.cumsum() < self.rng.random()).sum(), len(p) - 1))
        return a

    def end(self, o, outcome):
        pass


def p_greedy_at(P1, T):
    """Mean probability of the greedy action at temperature T, from T=1 probabilities (softmax(s/T) = p1^(1/T))."""
    lg = np.log(np.clip(P1, 1e-300, None)) / T
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg); p /= p.sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def load_brain(path, eta=0.001):
    from cadence import Brain
    b = mem.make_cold(Brain.load(str(path)))
    b.learner.config = dataclasses.replace(b.learner.config, eta=eta, eta_bias=eta / 10)
    return b


def calib(args):
    L = cf.check_level(args.level); b = load_brain(cf.CKPTS[args.ckpt])
    P = []
    for s in [int(v) for v in cf.POOL[:args.n]]:
        pl = SPlayer(b, 0.0, rec=P)
        mem.play_attempt(pl, L, s)
    P = np.array(P)
    grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
    tab = {T: round(p_greedy_at(P, T), 4) for T in grid}
    lo, hi = 1e-3, 5.0  # bisection on T for mean P(greedy) = target (decreasing in T)
    for _ in range(50):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if p_greedy_at(P, mid) > args.target else (lo, mid)
    T = round((lo * hi) ** 0.5, 4)
    top2 = np.sort(P, axis=1)[:, -2:]
    out = dict(level=L, ckpt=args.ckpt, target=args.target, T=T, decisions=len(P), p_greedy_by_T=tab,
               T1_margin_median=float(np.median(np.log(top2[:, 1] / top2[:, 0]))))
    d = HERE / "runs" / f"pilot_{L}"; d.mkdir(parents=True, exist_ok=True)
    (d / f"calib_{args.ckpt}.json").write_text(json.dumps(out, indent=1)); print(out)


def anchors(per_level=3000, crit=3.0, seed=7):
    """A fixed anchor set from the coached worlds 1-3 piles: rows drawn with p ~ W x (crit if W>1), as ctrain samples
    pile rows (draws with replacement, so frequencies carry the weights)."""
    lv = json.loads((cf.cs.DATA / "levels.json").read_text())
    r = np.random.default_rng(seed); X, Y = [], []
    for L in cf.TRAIN:
        z = np.load((cf.cs.HERE / lv[L]).resolve() / "data.npz")
        W = z["W"].astype(np.float64); p = W * np.where(W > 1, crit, 1.0); p /= p.sum()
        j = r.choice(len(p), per_level, p=p)
        X.append(z["X"][j]); Y.append(z["y"][j])
    return np.concatenate(X), np.concatenate(Y).astype(np.int64)


def good_rows(atts, margin=32, scale=256.0, back=128, drop_dead=3):
    """B1's filter (pre-registered): baseline B = median max x of this round's attempts. An attempt with max x M > B +
    margin is good, weight w = min(1, (M - B) / scale). Its rows: decisions with x >= B - back, up to the decision that
    first reached M; if it died, the last `drop_dead` decisions are dropped. Returns (obs uint8, actions, weights)."""
    B = float(np.median([a["max_x"] for a in atts])); O, A, W = [], [], []
    for a in atts:
        M = a["max_x"]
        if M <= B + margin:
            continue
        w = min(1.0, (M - B) / scale); xs = np.array(a["xs"]); n = len(xs)
        tM = int(np.argmax(np.maximum.accumulate(xs) >= M)) if (xs >= M).any() else n - 1
        hi = min(tM + 1, n - drop_dead) if a["end"] == "died" else (n if a["flag"] else tM + 1)
        idx = [t for t in range(hi) if xs[t] >= B - back]
        O.append(a["obs"][idx]); A.append(np.array(a["actions"])[idx]); W.append(np.full(len(idx), w))
    if not O:
        return np.zeros((0, 2000), np.uint8), np.zeros(0, np.int64), np.zeros(0), B
    return np.concatenate(O), np.concatenate(A).astype(np.int64), np.concatenate(W), B


def gameslot(state, log):
    """When a game runs, arms take turns on one slot (file lock). Game rechecked at most every 5 min."""
    if time.time() - state["t"] > 300:
        state["t"] = time.time(); g = cf.gamecheck.gaming()
        if g != state["g"]:
            log(f"game {'running: one arm at a time' if g else 'gone'}"); state["g"] = g
    if not state["g"]:
        return None
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return LOCK
        except FileExistsError:
            try:
                if time.time() - LOCK.stat().st_mtime > 3600:
                    LOCK.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            time.sleep(15)


def run(args):
    cf.idle(); cf.no_coach()
    L = cf.check_level(args.level)
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    T = json.loads((HERE / "runs" / f"pilot_{L}" / f"calib_{args.ckpt}.json").read_text())["T"] if args.T is None else args.T
    b = load_brain(cf.CKPTS[args.ckpt], args.eta)
    AX, AY = anchors()
    log(f"start {vars(args)} T={T} anchors={len(AY)} pid={os.getpid()}")
    rng = np.random.default_rng(1234)  # practice starts + seeds: identical across arms
    lrng = np.random.default_rng(99)   # lesson sampling
    bufO, bufA, bufW = np.zeros((0, 2000), np.uint8), np.zeros(0, np.int64), np.zeros(0)
    hist = out / "history.jsonl"; state = dict(t=0.0, g=False)

    def evaluate(levels, starts):
        res = {}
        for LL in levels:
            pl = SPlayer(b, 0.0); xs, fl = [], 0
            for s in starts:
                r = mem.play_attempt(pl, cf.check_level(LL), int(s)); xs.append(r["max_x"]); fl += r["flag"]
            res[LL] = dict(x=xs, mean=round(float(np.mean(xs)), 1), flags=fl)
        b.reset(); return res

    ev0 = evaluate([L], cf.HELD)[L]
    log(f"round 0 HELD {L}: mean {ev0['mean']} flags {ev0['flags']}")
    with open(hist, "a") as f:
        f.write(json.dumps(dict(round=0, held=ev0, t=time.time())) + "\n")
    for k in range(1, args.rounds + 1):
        slot = gameslot(state, log)
        try:
            t0 = time.time()
            starts = rng.choice(cf.POOL, args.attempts, replace=False); seeds = rng.integers(1 << 30, size=args.attempts)
            pl = SPlayer(b, T); atts = []
            for s, sd in zip(starts, seeds):
                r = mem.play_attempt(pl, L, int(s), seed=int(sd), keep_obs=True); atts.append(r)
            b.reset()
            px = [a["max_x"] for a in atts]; pf = sum(a["flag"] for a in atts)
            rec = dict(round=k, practice=dict(x=px, mean=round(float(np.mean(px)), 1), median=float(np.median(px)), flags=pf,
                                              starts=[int(s) for s in starts]))
            t1 = time.time()
            if args.arm in ("b1", "rand"):
                if args.arm == "b1":
                    O, A, W, B = good_rows(atts)
                else:
                    O = np.concatenate([a["obs"] for a in atts]); A = np.concatenate([a["actions"] for a in atts]).astype(np.int64)
                    W = np.ones(len(A)); B = float(np.median(px))
                bufO = np.concatenate([bufO, O])[-args.buf:]; bufA = np.concatenate([bufA, A])[-args.buf:]
                bufW = np.concatenate([bufW, W])[-args.buf:]
                rec.update(baseline=B, new_rows=int(len(A)), buf=int(len(bufA)))
            if args.arm != "reload":
                n_own = args.own if (args.arm in ("b1", "rand") and len(bufA)) else 0
                pw = bufW / bufW.sum() if len(bufA) else None
                for _ in range(args.lessons):
                    ja = lrng.integers(len(AY), size=32 - n_own)
                    X = AX[ja]; Y = AY[ja]
                    if n_own:
                        jo = lrng.choice(len(bufA), n_own, p=pw)
                        X = np.concatenate([X, bufO[jo]]); Y = np.concatenate([Y, bufA[jo]])
                    b.reset(); b.lesson(X.astype(np.float64) / 255.0, Y)
                b.reset()
            rec["teach_s"] = round(time.time() - t1)
            if k % args.eval_every == 0 or k == args.rounds:
                if args.arm != "reload" or k == args.rounds:
                    rec["held"] = evaluate([L], cf.HELD)[L]
                    b.save(out / f"ck{k:03d}.npz")
            rec["s"] = round(time.time() - t0); rec["t"] = time.time()
        finally:
            if slot is not None:
                slot.unlink(missing_ok=True)
        log(f"round {k} ({rec['s']}s): practice mean {rec['practice']['mean']} med {rec['practice']['median']} flags {pf}"
            + (f" | new rows {rec['new_rows']} (B {rec['baseline']:.0f}) buf {rec['buf']}" if "buf" in rec else "")
            + (f" | HELD mean {rec['held']['mean']} flags {rec['held']['flags']}" if "held" in rec else ""))
        with open(hist, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if args.forget:
        t = time.time(); fg = evaluate(cf.TRAIN, cf.SEL)
        m = float(np.mean([fg[LL]["mean"] for LL in cf.TRAIN]))
        log(f"forgetting check SEL 12 training levels ({time.time() - t:.0f}s): mean {m:.1f} flags {sum(fg[LL]['flags'] for LL in cf.TRAIN)} | "
            + " ".join(f"{LL} {fg[LL]['mean']}/{fg[LL]['flags']}" for LL in cf.TRAIN))
        (out / "forget.json").write_text(json.dumps(dict(mean=m, levels=fg), indent=1))
    log("done")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["calib", "run"])
    ap.add_argument("--level", required=True); ap.add_argument("--ckpt", default="CatoA")
    ap.add_argument("--target", type=float, default=0.85); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--arm", default="b1", choices=["b1", "anch", "rand", "reload"]); ap.add_argument("--out", default=None)
    ap.add_argument("--T", type=float, default=None); ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--rounds", type=int, default=30); ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--lessons", type=int, default=100); ap.add_argument("--own", type=int, default=8)
    ap.add_argument("--buf", type=int, default=4000); ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--forget", type=int, default=1)
    args = ap.parse_args()
    calib(args) if args.cmd == "calib" else run(args)


if __name__ == "__main__":
    main()
