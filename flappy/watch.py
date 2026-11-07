"""Demo: the trained brain (Corvus) plays Flappy Bird on the main monitor, game and brain side by side.
    python watch.py [--speed 1.0] [--learn | --reward]
--learn   the teacher keeps correcting it: after each greedy move the teacher's label for that state is taught
          (DAgger, as in training). Saves to brains/corvus_live.npz after every game.
--reward  no teacher at all: Corvus samples its own moves and learns from its score through Cadence's reward rule
          (`Brain.step`, docs/reward.md): +1 per pipe passed, -1 for dying, +0.01 per frame alive.
          actor_eta 0.1 (a tenth of the composed rate, as reward.md advises for one noisy stream), sampling
          temperature 0.001 (see REWARD_T).
          Saves to brains/corvus_reward.npz after every game.
brains/corvus.npz (the tested brain) is never overwritten.
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
PIPE_REWARD, DEATH_REWARD, ALIVE_REWARD = 1.0, -1.0, 0.01
# The imitation brain's two motor neurons are nearly tied (greedy agrees with the teacher, sampling does not):
# sampled play on 20 dev games: T 0.2 -> 0 pipes, 0.01 -> 0, 0.001 -> median 9 / mean 14, greedy -> 70.
REWARD_T = 0.001


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--learn", action="store_true"); ap.add_argument("--reward", action="store_true")
    a = ap.parse_args()
    b = Brain.load(HERE / "brains" / "corvus.npz")
    mode = "own score, no teacher" if a.reward else "teacher corrects" if a.learn else "off"
    if a.reward:
        b.retune(actor_eta=0.1, temperature=REWARD_T)
    assoc, motor = np.asarray(b.association_index), np.asarray(b.motor_index)
    v = View(f"Corvus plays Flappy Bird (Cadence 0.94, learning: {mode})"); v.chart_title = "pipes per game"
    games, seed = 0, a.seed
    pending = None   # (reward, done) owed for the previous sampled move (reward mode)
    b.reset()
    while not v.quit:
        g = G.Flappy(1, seed=seed, cap=10**9); games += 1; seed += 1
        if not a.reward:
            b.reset()
        while not g.dead.all() and v.pump():
            while v.paused and v.pump():
                time.sleep(0.05)
            t0 = time.time()
            obs = g.senses(); recs = []
            with record_settlements(recs.append, label="watch"):
                if a.reward:
                    if pending is None:
                        act = b.step(obs)
                    else:
                        act = b.step(obs, reward=np.array([pending[0]]), done=np.array([pending[1]]))
                    act = np.asarray(act).reshape(-1)
                else:
                    act = np.asarray(b.act(obs, greedy=True)).reshape(-1)
            if a.learn:
                b.lesson(obs, G.teacher(g))
            before = int(g.score[0])
            died = bool(g.step(act)[0])
            if a.reward:   # owed to this move; paid at the next step (the next game's first frame after a death)
                pending = (DEATH_REWARD if died else ALIVE_REWARD + PIPE_REWARD * (int(g.score[0]) - before), died)
            m = last_activation(recs)
            mv = m[motor]; e = np.exp(mv - mv.max())
            v.draw_game(g, 0, int(act[0]))
            v.draw_brain(obs[0], m, assoc, motor, float(e[1] / e.sum()),
                         f"Corvus | learning: {mode} | game {games} | best {max([s for _, s in v.history] + [0]):.0f}")
            v.flip()
            if not v.fast:
                time.sleep(max(0.0, (1 / 60) / a.speed - (time.time() - t0)))
        v.history.append((f"game {games}", float(g.score[0])))
        print(f"game {games}: {int(g.score[0])} pipes", flush=True)
        if a.learn or a.reward:
            b.save(HERE / "brains" / ("corvus_reward.npz" if a.reward else "corvus_live.npz"))
        if not v.quit:
            time.sleep(0.8)


if __name__ == "__main__":
    main()
