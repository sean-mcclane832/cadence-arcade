"""B3: Brain.live with a progress reward (COACHFREE.md). Coach-free, worlds 6-7 only (cf.no_coach, World 4 refused).

The cold-start checkpoints were composed without arousal genes and `retune` cannot add them, so `arousal_brain` writes
a copy of the checkpoint whose layout is recomposed with arousal=True (same seed, so the identical connectome: pre/post
checked) carrying every learned array of the checkpoint; only the arousal section of the brain record is new.
Greedy play of the copy is checked identical to the checkpoint on a few starts before the run.

Practice: the same 16 POOL starts per round as b1.py. Every decision goes through `live`: calm = greedy settle and no
learning; aroused (surprise or want) = sampled at the learner temperature, eligibility, actor/critic learning.
Reward per decision: clip(dx / 20, -1, 1) (shaped progress, dx = x change), death -1 and done, flag +1 and done,
stuck (40 decisions without a new max x) done with 0. Actor rate `--actor` (composed 1.0 erodes skills, upstream #169).
Eval and forgetting check exactly as b1.py (greedy HELD every 2 rounds, SEL 12 training levels at the end).

    python b3.py --level 6-1 --out runs/pilot_6-1/b3
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf  # noqa: E402
import b1  # noqa: E402

mem = cf.mem
C, D = mem.C, mem.D


def arousal_brain(src, out, seed=0):
    from cadence import Brain
    tmp = Path(str(out) + ".fresh.npz")
    Brain.compose(2000, 7, modules=(256,), seed=seed, teaching="normalized", learning_surprise=False,
                  arousal=True).save(tmp)
    old = dict(np.load(src, allow_pickle=True)); new = dict(np.load(tmp, allow_pickle=True)); tmp.unlink()
    assert np.array_equal(old["pre"], new["pre"]) and np.array_equal(old["post"], new["post"]), "connectome differs"
    ob, nb = json.loads(str(old["brain"])), json.loads(str(new["brain"]))
    ob["arousal"] = nb["arousal"]
    old["brain"] = np.array(json.dumps(ob))
    np.savez(out, **old)
    return Brain.load(out)


def live_attempt(b, level, start, scale=20.0):
    lvl = tuple(int(v) for v in level.split("-"))
    body = C.CBody(int(start), "body", "lean", lvl); g = body.g
    b.reset()
    x_prev = None; acts, xs = [], []; modes = dict(routine=0, aroused=0)
    while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
        o = body.obs()[None].astype(np.float64)
        if x_prev is None:
            a = b.live(o)
        else:
            a = b.live(o, reward=[float(np.clip((g.x - x_prev) / scale, -1, 1))], done=[False])
        a = int(np.asarray(a).reshape(-1)[0])
        m = (b.last_arousal or {}).get("mode")
        if m in modes:
            modes[m] += 1
        x_prev = g.x; acts.append(a); xs.append(int(g.x))
        body.step(a)
    end = "flag" if g.flag else ("died" if g.done else "stuck")
    r = {"flag": 1.0, "died": -1.0}.get(end, 0.0)
    b.live(body.obs()[None].astype(np.float64), reward=[r], done=[True])
    out = dict(start=int(start), max_x=int(g.max_x), flag=bool(g.flag), end=end, decisions=g.decisions, modes=modes)
    g.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True); ap.add_argument("--ckpt", default="CatoA"); ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30); ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2); ap.add_argument("--actor", type=float, default=0.02)
    ap.add_argument("--youth", type=int, default=0); ap.add_argument("--forget", type=int, default=1)
    args = ap.parse_args()
    cf.idle(); cf.no_coach()
    L = cf.check_level(args.level)
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    src = cf.CKPTS[args.ckpt]
    b = mem.make_cold(arousal_brain(str(src), str(out / "start_arousal.npz")))
    ref = b1.load_brain(src)
    for s in (0, 1, 2):  # greedy parity: recomposed copy plays as the checkpoint
        ra = mem.play_attempt(b1.SPlayer(b, 0.0), L, s); rr = mem.play_attempt(b1.SPlayer(ref, 0.0), L, s)
        assert ra["actions"] == rr["actions"], f"parity failed at start {s}"
    del ref
    b.retune(actor_eta=args.actor, actor_eta_bias=args.actor / 10, arousal_youth=args.youth)
    b.arousal.reset()
    log(f"start {vars(args)} parity ok (3 starts) pid={os.getpid()}")
    rng = np.random.default_rng(1234)  # same practice starts as b1.py
    hist = out / "history.jsonl"; state = dict(t=0.0, g=False)

    def evaluate(levels, starts):
        res = {}
        for LL in levels:
            pl = b1.SPlayer(b, 0.0); xs, fl = [], 0
            for s in starts:
                r = mem.play_attempt(pl, cf.check_level(LL), int(s)); xs.append(r["max_x"]); fl += r["flag"]
            res[LL] = dict(x=xs, mean=round(float(np.mean(xs)), 1), flags=fl)
        b.reset(); return res

    ev0 = evaluate([L], cf.HELD)[L]
    log(f"round 0 HELD {L}: mean {ev0['mean']} flags {ev0['flags']}")
    with open(hist, "a") as f:
        f.write(json.dumps(dict(round=0, held=ev0, t=time.time())) + "\n")
    for k in range(1, args.rounds + 1):
        slot = b1.gameslot(state, log)
        try:
            t0 = time.time()
            starts = rng.choice(cf.POOL, args.attempts, replace=False); rng.integers(1 << 30, size=args.attempts)
            atts = [live_attempt(b, L, int(s)) for s in starts]
            b.reset()
            px = [a["max_x"] for a in atts]; pf = sum(a["flag"] for a in atts)
            md = dict(routine=sum(a["modes"]["routine"] for a in atts), aroused=sum(a["modes"]["aroused"] for a in atts))
            rec = dict(round=k, practice=dict(x=px, mean=round(float(np.mean(px)), 1), median=float(np.median(px)), flags=pf,
                                              starts=[int(s) for s in starts]), modes=md)
            if k % args.eval_every == 0 or k == args.rounds:
                rec["held"] = evaluate([L], cf.HELD)[L]; b.save(out / f"ck{k:03d}.npz")
            rec["s"] = round(time.time() - t0); rec["t"] = time.time()
        finally:
            if slot is not None:
                slot.unlink(missing_ok=True)
        log(f"round {k} ({rec['s']}s): practice mean {rec['practice']['mean']} med {rec['practice']['median']} flags {pf} "
            f"| moments routine {md['routine']} aroused {md['aroused']}"
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


if __name__ == "__main__":
    main()
