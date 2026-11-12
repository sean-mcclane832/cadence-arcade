"""Leg 2a: the copycat tiny test on the existing piles (no game is played). Cadence 0.98.0, CPU.

    idle.sh 098 tinycopy.py --arm plain --seed 0 [--lessons 300]

One arm per process. A fresh Cato2-recipe brain (3,536 inputs, 7 buttons, one module of 256, recall off, normalized
teaching, every row taught, working trace as composed) is taught ctrain-style: 32 rows a lesson, equal share per
level, rows drawn by pile weight (critical x3), ordered coach rows with K = 8 burn-in acts, cold settles.
Split: per level the 11 coach runs; run 5 (clean, start 27) and run 8 (noisy) are HELD OUT, the other 9 are taught.
Score: the brain answers the held-out clean run of every level in order (teacher forced, as in play), first 200
decisions; agreement split into frames where the coach's button repeats the previous one and frames where it changes.
Arms: plain | change (rows where the label differs from the held button weigh 4) | drop (held-button inputs blanked
on half the lesson rows) | once (body sense once, not 8 copies: copies 2-8 blanked everywhere) | trace0 (working
trace amplitude 0) | slow (association<->motor steps at 0.25).
Writes scratch/tiny/<arm>_<seed>.json.
"""
import argparse, json, os, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
from pathlib import Path
import numpy as np

FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
HERE = Path(__file__).resolve().parent
LEVELS = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
HELD_OUT, K, T_EVAL = (5, 8), 8, 200


def load_split():
    train, held = {}, {}
    for L in LEVELS:
        z = np.load(FAB / L / f"seq_{L}_coach.npz"); off = z["off"]; O, A = z["O"], z["A"]
        lab = np.full(len(O), -1); lab[z["li"]] = z["y"]; w = np.zeros(len(O)); w[z["li"]] = z["W"]
        atts = [dict(O=O[off[k]:off[k + 1]], A=A[off[k]:off[k + 1]], y=lab[off[k]:off[k + 1]], w=w[off[k]:off[k + 1]]) for k in range(len(off) - 1)]
        for a in atts:
            a["prev"] = np.concatenate([[-1], a["A"][:-1]])
        train[L] = [a for k, a in enumerate(atts) if k not in HELD_OUT]; held[L] = atts[HELD_OUT[0]]
    return train, held


def body_view(x):
    return x[:, 1664:2000].reshape(len(x), 8, 42)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lessons", type=int, default=300); ap.add_argument("--every", type=int, default=50)
    a = ap.parse_args()
    import dataclasses
    from cadence import Brain
    opts = dict(modules=(256,), teaching="normalized", learning_surprise=False, learning_fast_settle=True, carried_gain=0.01,
                recall=False, backend="cpu")
    if a.arm == "trace0":
        opts["working_memory_amplitude"] = 0.0
    b = Brain.compose(3536, 7, seed=a.seed, **opts); b._activity = lambda: None
    b.learner.config = dataclasses.replace(b.learner.config, eta=0.001, eta_bias=0.0001)
    if a.arm == "slow":
        c = b.connectome; mot = np.asarray(b.learner.output_index); assoc = np.asarray(b.association_index)
        pair = (np.isin(c.pre, assoc) & np.isin(c.post, mot)) | (np.isin(c.pre, mot) & np.isin(c.post, assoc))
        rate = np.ones(len(c.pre)); rate[pair] = 0.25; b.learner.synapse_rate = rate

    def edit(x):
        if a.arm == "once":
            x = x.copy(); body_view(x)[:, 1:, :] = 0
        return x

    train, held = load_split()
    idx = {L: [(k, i) for k, t in enumerate(train[L]) for i in range(len(t["y"])) if t["y"][i] >= 0] for L in LEVELS}
    p = {}
    for L in LEVELS:
        w = np.array([train[L][k]["w"][i] for k, i in idx[L]]); w = w * np.where(w > 1, 3.0, 1.0); p[L] = w / w.sum()
    r = np.random.default_rng(11)

    def lesson():
        cnt = r.multinomial(32, np.ones(24) / 24); items = []
        for L, n in zip(LEVELS, cnt):
            for j in r.choice(len(idx[L]), n, p=p[L]):
                k, i = idx[L][int(j)]; items.append((train[L][k], i))
        b.reset()
        for j in range(K, 0, -1):
            x = np.stack([t["O"][max(i - j, 0)] for t, i in items]).astype(np.float64) / 255.0
            b.act(edit(x), greedy=True)
        x = edit(np.stack([t["O"][i] for t, i in items]).astype(np.float64) / 255.0)
        y = np.array([t["y"][i] for t, i in items]); ch = np.array([t["y"][i] != t["prev"][i] for t, i in items])
        if a.arm == "drop":
            m = r.random(32) < 0.5; body_view(x)[m, :, 38:42] = 0
        if a.arm == "change":
            b.lesson(x, y, weights=np.where(ch, 4.0, 1.0))
        else:
            b.lesson(x, y)

    def score(atts, T):
        S = len(atts); b.reset(); ans = np.full((S, T), -1); n = [min(T, len(t["y"])) for t in atts]
        for t in range(T):
            x = np.stack([q["O"][min(t, len(q["O"]) - 1)] for q in atts]).astype(np.float64) / 255.0
            ans[:, t] = np.asarray(b.act(edit(x), greedy=True)).reshape(S, -1)[:, 0]
        ok, ch, hb = [], [], []
        for s, q in enumerate(atts):
            ok.append(ans[s, :n[s]] == q["y"][:n[s]]); ch.append(q["y"][:n[s]] != q["prev"][:n[s]]); hb.append(ans[s, :n[s]] == q["prev"][:n[s]])
        ok, ch, hb = np.concatenate(ok), np.concatenate(ch), np.concatenate(hb)
        b.reset()
        return dict(all=round(float(ok.mean()), 4), repeat=round(float(ok[~ch].mean()), 4), change=round(float(ok[ch].mean()), 4),
                    n=int(ok.size), n_change=int(ch.sum()), answers_held_button=round(float(hb.mean()), 4))

    out = dict(arm=a.arm, seed=a.seed, lessons=a.lessons, curve=[]); t0 = time.time()
    held_atts = [held[L] for L in LEVELS]; seen_atts = [train[L][0] for L in LEVELS]
    for n in range(1, a.lessons + 1):
        lesson()
        if n % a.every == 0 or n == 3:
            rec = dict(lesson=n, s=round(time.time() - t0), held=score(held_atts, T_EVAL))
            if n == a.lessons:
                rec["seen"] = score(seen_atts, 100)
            out["curve"].append(rec); print(json.dumps(rec), flush=True)
            (HERE / "scratch" / "tiny").mkdir(exist_ok=True)
            json.dump(out, open(HERE / "scratch" / "tiny" / (f"{a.arm}_{a.seed}" + ("" if a.lessons == 300 else f"_L{a.lessons}") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
