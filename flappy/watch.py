"""Demo: the trained brain (Corvus) plays Flappy Bird on the main monitor, game and brain side by side, no learning.
    python watch.py [--speed 1.0]
Keys: space pause, f fast, q quit. A new game starts when the bird dies.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
from cadence import Brain
from cadence.recording import record_settlements
import game as G
from view import View
from train import last_activation

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--seed", type=int, default=9000)
    a = ap.parse_args()
    b = Brain.load(HERE / "brains" / "corvus.npz")
    assoc, motor = np.asarray(b.association_index), np.asarray(b.motor_index)
    v = View("Corvus plays Flappy Bird (Cadence 0.94, no learning)"); v.chart_title = "pipes per game"
    games, seed = 0, a.seed
    while not v.quit:
        g = G.Flappy(1, seed=seed, cap=10**9); b.reset(); games += 1; seed += 1
        while not g.dead.all() and v.pump():
            while v.paused and v.pump():
                time.sleep(0.05)
            t0 = time.time()
            obs = g.senses(); recs = []
            with record_settlements(recs.append, label="watch"):
                act = np.asarray(b.act(obs, greedy=True)).reshape(-1)
            g.step(act)
            m = last_activation(recs)
            mv = m[motor]; e = np.exp(mv - mv.max())
            v.draw_game(g, 0, int(act[0]))
            v.draw_brain(obs[0], m, assoc, motor, float(e[1] / e.sum()), f"Corvus | game {games} | best so far {max([s for _, s in v.history] + [0]):.0f}")
            v.flip()
            if not v.fast:
                time.sleep(max(0.0, (1 / 60) / a.speed - (time.time() - t0)))
        v.history.append((f"game {games}", float(g.score[0])))
        if not v.quit:
            time.sleep(0.8)


if __name__ == "__main__":
    main()
