"""Players for Ember Field: a Cadence brain, a plain online MLP (Q-learning), random."""

from __future__ import annotations

import numpy as np


class CadencePlayer:
    name = "cadence"

    def __init__(self, n_in: int, seed: int, modules=(32,), memory_choice: bool = True, **kw):
        from cadence import Brain
        self.brain = Brain.compose(n_in, 3, modules=tuple(modules), arousal=True, seed=seed,
                                   memory_choice=memory_choice, **kw)

    def first(self, obs):
        return int(self.brain.live(obs[None])[0])

    def act(self, obs, reward):
        return int(self.brain.live(obs[None], reward=[reward])[0])


class RandomPlayer:
    name = "random"

    def __init__(self, n_in: int, seed: int, **kw):
        self.rng = np.random.default_rng(seed + 777)

    def first(self, obs):
        return int(self.rng.integers(3))

    def act(self, obs, reward):
        return int(self.rng.integers(3))


class MLPPlayer:
    """One hidden layer Q-learner, online, epsilon-greedy. Same inputs, same rewards."""
    name = "mlp"

    def __init__(self, n_in: int, seed: int, hidden: int = 32, lr: float = 0.01,
                 gamma: float = 0.8, eps: float = 0.1, **kw):
        r = np.random.default_rng(seed + 999)
        self.rng = r
        self.W1 = r.normal(0, 1 / np.sqrt(n_in), (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 3)); self.b2 = np.zeros(3)
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.prev = None

    def _q(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def _choose(self, q):
        return int(self.rng.integers(3)) if self.rng.random() < self.eps else int(np.argmax(q))

    def first(self, obs):
        h, q = self._q(obs)
        a = self._choose(q)
        self.prev = (obs, h, q, a)
        return a

    def act(self, obs, reward):
        x, h, q, a = self.prev
        _, qn = self._q(obs)
        target = reward + self.gamma * qn.max()
        err = q[a] - target
        gq = np.zeros(3); gq[a] = err
        gh = (self.W2 @ gq) * (h > 0)
        self.W2 -= self.lr * np.outer(h, gq); self.b2 -= self.lr * gq
        self.W1 -= self.lr * np.outer(x, gh); self.b1 -= self.lr * gh
        h2, q2 = self._q(obs)
        a2 = self._choose(q2)
        self.prev = (obs, h2, q2, a2)
        return a2
