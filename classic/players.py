"""Players for single-frame Breakout. Every player learns from the same coach labels.

- CadenceStudent: a Cadence brain taught with `Brain.lesson` at every training moment, then
  acting greedily (the look-back benchmark's lesson-then-act recipe). Three arms differ only in
  the recall memory: on with the 0.89.0 position term, on with `recall_position=0`, or off.
- MLPStudent: one hidden layer of 32, softmax, online cross-entropy on the same labels. Sees the
  single frame, or (the "two-frame" reference arm) the single frame plus MinAtar's trail, i.e.
  it is *given* where the ball was a moment ago instead of having to remember it.
- RandomPlayer.
"""

from __future__ import annotations

import warnings

import numpy as np

from breakout import N_IN


class CadenceStudent:
    def __init__(self, seed: int, recall: bool, position: int, **settings):
        from cadence import Brain
        kw = dict(settings)
        if recall:
            kw.update(recall=True, recall_position=position)
        self.brain = Brain.compose(N_IN, 3, seed=seed, **kw)
        self.position_rows = 0.0
        self.warned: dict[str, int] = {}

    def new_episode(self) -> None:
        self.brain.reset()  # empties the recall slots and working trace; learned maps stay

    def train(self, obs, prev, label: int) -> int:
        # Warnings are counted, not raised: "free phase used its entire budget" (a lesson from
        # a state that did not fully settle) and "lesson on the observations the brain just
        # acted on" (here a NEW moment that happens to have the same features as the last).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rep = self.brain.lesson(obs[None], np.array([label]))
        for w in caught:
            key = "unsettled" if "free_steps" in str(w.message) else "same_features"                 if "just acted on" in str(w.message) else "other"
            self.warned[key] = self.warned.get(key, 0) + 1
        self.position_rows += float(rep.get("position_rows", 0.0))
        return int(np.asarray(self.brain.act(obs[None], greedy=True)).reshape(-1)[0])

    def play(self, obs, prev) -> int:
        return int(np.asarray(self.brain.act(obs[None], greedy=True)).reshape(-1)[0])

    def position_size(self) -> float:
        rc = getattr(self.brain, "recall", None)
        if rc is None or not getattr(rc, "position", 0):
            return 0.0
        return float(sum(np.abs(m).sum() for maps in rc.position_maps for m in maps))


class MLPStudent:
    def __init__(self, seed: int, two_frame: bool = False, hidden: int = 32, lr: float = 0.05):
        r = np.random.default_rng(seed + 999)
        self.rng, self.two, self.lr = r, two_frame, lr
        n = N_IN * (2 if two_frame else 1)
        self.W1 = r.normal(0, 1 / np.sqrt(n), (n, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 3)); self.b2 = np.zeros(3)

    def new_episode(self) -> None:
        pass

    def _x(self, obs, prev):
        return np.concatenate([obs, prev]) if self.two else obs

    def _fwd(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        p = np.exp(z - z.max()); p /= p.sum()
        return h, p

    def train(self, obs, prev, label: int) -> int:
        x = self._x(obs, prev)
        h, p = self._fwd(x)
        g = p.copy(); g[label] -= 1.0
        gh = (self.W2 @ g) * (h > 0)
        self.W2 -= self.lr * np.outer(h, g); self.b2 -= self.lr * g
        self.W1 -= self.lr * np.outer(x, gh); self.b1 -= self.lr * gh
        return self.play(obs, prev)

    def play(self, obs, prev) -> int:
        return int(np.argmax(self._fwd(self._x(obs, prev))[1]))

    def position_size(self) -> float:
        return 0.0


class RandomPlayer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed + 777)

    def new_episode(self) -> None:
        pass

    def train(self, obs, prev, label: int = 0) -> int:
        return int(self.rng.integers(3))

    play = train

    def position_size(self) -> float:
        return 0.0
