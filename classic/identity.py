"""Check: under reward-only play the position term never moves, so it cannot help or hurt.

Two Cadence brains (recall on, recall_position=32 vs 0, same seed) live the same Breakout game
through `Brain.live` with the bricks as reward, no lessons, for MOMENTS moments. The script
records whether every action matched and whether the position maps stayed exactly zero.

    ../.venv-classic/Scripts/python.exe identity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from breakout import N_IN, Breakout

MOMENTS = 5000
SEEDS = (10, 11)


def life(seed: int, position: int):
    from cadence import Brain
    brain = Brain.compose(N_IN, 3, seed=seed, modules=(32,), arousal=True, recall=True,
                          recall_position=position, episodic=False)
    env = Breakout(seed=seed)
    actions, bricks, reward = [], 0.0, None
    for _ in range(MOMENTS):
        kw = {} if reward is None else {"reward": [reward]}
        a = int(np.asarray(brain.live(env.observe()[None], **kw)).reshape(-1)[0])
        actions.append(a)
        reward, done = env.step(a)
        bricks += reward
        if done:
            env.reset()
            reward = -1.0  # losing the ball costs one
    moved = 0.0
    if brain.recall.position:
        moved = float(sum(np.abs(m).sum() for maps in brain.recall.position_maps for m in maps))
    return actions, bricks, moved


def main():
    out = {}
    for seed in SEEDS:
        a32, b32, m32 = life(seed, 32)
        a0, b0, _ = life(seed, 0)
        out[seed] = {"same_actions": a32 == a0, "bricks_pos32": b32, "bricks_pos0": b0,
                     "position_maps_abs_sum": m32, "moments": MOMENTS}
        print(seed, out[seed], flush=True)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/identity.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
