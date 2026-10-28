"""Ember Field: a small original catching game for a continuing brain.

A lantern creature walks the bottom row of a 9 x 7 field. Embers drift down one row per
moment, amber or blue. Catching an ember (it lands in the creature's column) pays +1 if
that colour is food this season and -1 if it stings. Missing an ember pays nothing.
At each season turn the colours swap: the food becomes the sting and the sting the food.

The creature sees an egocentric window: for each of the 7 columns from 3 left to 3 right
of itself and each colour, how close the lowest ember of that colour is (0 none, then
1/3, 2/3, 1 as it nears the ground), plus a wall flag on each side. 16 numbers. It does
not see the season; it has to notice the turn from what its catches pay.
"""

from __future__ import annotations

import numpy as np

W, H = 9, 7
VIEW = 3  # default columns seen on each side
ACTIONS = ("left", "stay", "right")


def inputs(view: int = VIEW) -> int:
    return (2 * view + 1) * 2 + 2


INPUTS = inputs()


class EmberField:
    def __init__(self, seed: int = 0, spawn: float = 0.45, turn: int | None = None,
                 view: int = VIEW, turns: tuple[int, ...] = ()):
        self.rng = np.random.default_rng(seed)
        self.view = view
        self.spawn = spawn
        # moments at which the colours swap (turn: a single one, kept for old calls)
        self.turns = tuple(sorted(turns)) + ((turn,) if turn is not None else ())
        self.x = W // 2
        self.embers: list[list[int]] = []  # [col, row, colour 0 amber / 1 blue]
        self.t = 0

    def food(self) -> int:
        """The colour that pays this season."""
        return sum(self.t >= k for k in self.turns) % 2

    def observe(self) -> np.ndarray:
        v = self.view
        obs = np.zeros(inputs(v))
        for col, row, colour in self.embers:
            d = col - self.x
            if -v <= d <= v:
                level = np.ceil((row + 1) / H * 3) / 3  # 1/3, 2/3, 1
                i = (d + v) * 2 + colour
                obs[i] = max(obs[i], level)
        obs[-2] = 1.0 if self.x == 0 else 0.0
        obs[-1] = 1.0 if self.x == W - 1 else 0.0
        return obs

    def step(self, action: int) -> tuple[float, dict]:
        """Move, let the embers fall, settle catches. Returns (reward, event)."""
        self.x = int(np.clip(self.x + (action - 1), 0, W - 1))
        reward, caught = 0.0, None
        kept = []
        food = self.food()
        for e in self.embers:
            e[1] += 1
            if e[1] >= H - 1:
                if e[0] == self.x:
                    good = e[2] == food
                    reward += 1.0 if good else -1.0
                    caught = (e[2], good)
                continue
            kept.append(e)
        self.embers = kept
        if self.rng.random() < self.spawn:
            self.embers.append([int(self.rng.integers(W)), 0, int(self.rng.integers(2))])
        self.t += 1
        return reward, {"caught": caught}

    def frame(self) -> list:
        """Compact frame for the page: [x, [[col,row,colour],...]]."""
        return [self.x, [list(e) for e in self.embers]]
