"""Self-teaching: Corvus keeps learning with no teacher, through the teaching protocol (`lesson`) with labels from its
own outcomes (Sean, 2026-11-06: "make the learning work without a teacher, a teachers protocol").
Each round, 32 games: Corvus plays greedily, 5% of moves flipped at random to explore. Labels:
  - a pipe passed: its own moves since the previous pipe (learning_surprise=True, so only the moves that differ from
    its current answer teach anything, in practice the explored flips that worked)
  - a death: the opposite of its last --death-frames moves (default 0 = off: the first try used 6 and one round took
    the dev median from 81 to 0; with learning_surprise only disagreeing labels teach, so the flips dominated)
Lessons are applied at the end of each round. Dev check after each round: 20 games, seed 1000, cap 150, greedy, no
learning (the same dev games as train.py). Starts from brains/corvus.npz, writes brains/corvus_self.npz.
    python selfteach.py [--rounds 8]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from cadence import Brain
from cadence.recording import record_settlements
import game as G
from view import View
from train import last_activation

HERE = Path(__file__).resolve().parent
STREAMS, FRAMES, EXPLORE = 32, 1500, 0.05
DEV_CAP = 150   # train.py stopped at its dev cap of 50, so the self-teaching check needs headroom above it


def dev_check(b, v, label, motor, assoc):
    """20 dev games, greedy, no learning, drawing game 0 (or the first game still alive)."""
    b.reset(); g = G.Flappy(20, seed=1000, cap=DEV_CAP); t = 0
    while not g.dead.all() and v.pump():
        t0 = time.time(); obs = g.senses(); recs = []
        with record_settlements(recs.append, label="dev"):
            act = np.asarray(b.act(obs, greedy=True)).reshape(-1)
        g.step(act); t += 1
        r = int(np.argmin(g.dead))
        if not v.fast or t % 10 == 0:
            m = last_activation(recs); mv = m[motor]; e = np.exp(mv - mv.max())
            v.draw_game(g, r, int(act[r]))
            v.draw_brain(obs[r], m, assoc, motor, float(e[1] / e.sum()),
                         f"{label} | {int((~g.dead).sum())}/20 dev games alive | best {int(g.score.max())}")
            v.flip(); pace(v, t0)
    return g.score.copy()


def pace(v, t0):
    if not v.fast:
        time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def teach(b, obs, labels):
    """Apply lessons in batches of exactly STREAMS rows (the brain's stream count); the remainder is dropped."""
    b.reset(); n = (len(obs) // STREAMS) * STREAMS
    for i in range(0, n, STREAMS):
        b.lesson(obs[i:i + STREAMS], labels[i:i + STREAMS])
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--death-frames", type=int, default=0); a = ap.parse_args()
    b = Brain.load(HERE / "brains" / "corvus.npz")
    assoc, motor = np.asarray(b.association_index), np.asarray(b.motor_index)
    v = View("Corvus teaches itself Flappy Bird (no teacher)"); v.chart_title = "dev median pipes per round (cap 150)"
    rng = np.random.default_rng(42)
    base = dev_check(b, v, "starting point: dev check (no learning)", motor, assoc); log = [dict(round=0, dev_median=float(np.median(base)), dev_mean=float(base.mean()), lessons=0)]
    v.history.append(("teach", float(np.median(base)))); print(log[-1], flush=True)
    best, best_med = None, float(np.median(base))
    for rnd in range(1, a.rounds + 1):
        g = G.Flappy(STREAMS, seed=500 + rnd, cap=10**9); b.reset()
        buf = [[] for _ in range(STREAMS)]           # (obs, action) since that game's last pipe
        L_obs, L_lab = [], []
        for t in range(FRAMES):
            if not v.pump():
                break
            while v.paused and v.pump():
                time.sleep(0.05)
            t0 = time.time(); obs = g.senses(); recs = []
            with record_settlements(recs.append, label="self"):
                act = np.asarray(b.act(obs, greedy=True)).reshape(-1)
            flip = rng.random(STREAMS) < EXPLORE
            act = np.where(flip, 1 - act, act)
            before = g.score.copy()
            died = g.step(act)
            for r in range(STREAMS):
                buf[r].append((obs[r], int(act[r])))
                if g.score[r] > before[r]:            # passed a pipe: keep what it did
                    L_obs += [o for o, _ in buf[r]]; L_lab += [x for _, x in buf[r]]; buf[r] = []
                elif died[r] and a.death_frames:      # died: the opposite of its last moves (off by default)
                    tail = buf[r][-a.death_frames:]
                    L_obs += [o for o, _ in tail]; L_lab += [1 - x for _, x in tail]; buf[r] = []
                elif died[r]:
                    buf[r] = []
            if died.any():
                g.reset(died)
            if not v.fast or t % 10 == 0:
                v.draw_game(g, 0, int(act[0]))
                m = last_activation(recs); mv = m[motor]; e = np.exp(mv - mv.max())
                v.draw_brain(obs[0], m, assoc, motor, float(e[1] / e.sum()),
                             f"round {rnd}: playing + exploring | frame {t} | {len(L_obs)} self-labels so far")
                v.flip(); pace(v, t0)
        if v.quit:
            break
        n = teach(b, np.asarray(L_obs, np.float32), np.asarray(L_lab, np.int64))
        s = dev_check(b, v, f"round {rnd}: dev check after {n} self-lessons", motor, assoc); med = float(np.median(s))
        log.append(dict(round=rnd, dev_median=med, dev_mean=float(s.mean()), dev_min=int(s.min()), lessons=n))
        v.history.append(("own", med)); print(log[-1], flush=True)
        if med >= best_med:
            best_med = med; b.save(HERE / "brains" / "corvus_self.npz")
    (HERE / "selfteach_log.json").write_text(json.dumps(log, indent=1))
    print("best dev median", best_med, flush=True)


if __name__ == "__main__":
    main()
