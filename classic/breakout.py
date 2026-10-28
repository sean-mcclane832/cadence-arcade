"""MinAtar Breakout as a single-frame game, plus a coach that knows where the ball is going.

The game is MinAtar's Breakout (Young & Tian 2019, GPL-3.0, `pip install minatar`), run
through its own `Environment` wrapper with the standard sticky-action probability of 0.1.
Nothing about the rules is changed here.

What changes is what the player sees. MinAtar gives the ball a "trail" channel (where the ball
was a moment ago) so a single frame shows which way it is moving. This wrapper drops the trail.
The player gets one frame of hand features:

    ball column (10, one-hot), ball row (10), paddle column (10),
    ball column minus paddle column (19, one-hot from -9 to +9)       -> 49 numbers

From one frame like that, a ball going up-left and a ball coming down-right look the same.
The only way to know is to remember where the ball was a moment ago.

The coach reads the true game state (ball direction included), plays the ball forward on a copy
of the game and moves the paddle toward the column where the ball will come down.
"""

from __future__ import annotations

import copy

import numpy as np
from minatar import Environment

ACTIONS = (0, 1, 3)          # MinAtar's minimal set for Breakout: stay, left, right
NAMES = ("stay", "left", "right")
N_IN = 49


def features(ball_x: int, ball_y: int, pos: int) -> np.ndarray:
    x = np.zeros(N_IN)
    x[ball_x] = 1.0
    x[10 + ball_y] = 1.0
    x[20 + pos] = 1.0
    x[30 + (ball_x - pos) + 9] = 1.0
    return x


class Breakout:
    def __init__(self, seed: int, sticky: float = 0.1):
        self.env = Environment("breakout", sticky_action_prob=sticky)
        self.env.seed(seed)
        self.env.reset()

    @property
    def g(self):
        return self.env.env  # the MinAtar game object

    def reset(self) -> None:
        self.env.reset()

    def observe(self) -> np.ndarray:
        g = self.g
        return features(g.ball_x, g.ball_y, g.pos)

    def previous(self) -> np.ndarray:
        """The same features for where the ball was a moment ago (MinAtar's trail)."""
        g = self.g
        return features(g.last_x, g.last_y, g.pos)

    def step(self, action: int) -> tuple[float, bool]:
        r, done = self.env.act(ACTIONS[action])
        return float(r), bool(done)

    def frame(self) -> list:
        """Ball (x, y), trail (x, y), paddle x, the 30 brick cells of rows 1-3 as a bit string."""
        g = self.g
        bricks = "".join(str(int(v)) for v in g.brick_map[1:4].reshape(-1))
        return [int(g.ball_x), int(g.ball_y), int(g.last_x), int(g.last_y), int(g.pos), bricks]

    def landing(self) -> int | None:
        """Column the paddle should stand in when the ball comes down, from a copy of the game."""
        sim = copy.deepcopy(self.g)
        for _ in range(60):
            if sim.ball_y == 8 and sim.ball_dir in (2, 3):
                return int(sim.ball_x)
            sim.act(0)
            if sim.terminal:
                return int(sim.last_x)
        return None

    def coach(self) -> int:
        """The coach's move: toward where the ball will land (ties and arrivals: stay)."""
        target = self.landing()
        if target is None:
            target = self.g.ball_x
        pos = self.g.pos
        if target < pos:
            return 1
        if target > pos:
            return 2
        return 0
