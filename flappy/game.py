"""Flappy Bird for Cadence brains: N games at once in numpy (one per batch row), plus senses and a scripted teacher.
Physics per frame: gravity, a flap sets the speed, pipes scroll left. Score = pipes passed.
"""
from __future__ import annotations
import numpy as np

W, H = 288, 512
GROUND = 450          # top of the ground strip
BIRD_X, BIRD_R = 60, 12
GRAVITY, FLAP_V, MAX_FALL = 0.45, -7.0, 9.0
PIPE_W, GAP, SPEED, SPACING = 52, 140, 3.0, 180
STEP = 100            # a new gap centre is at most this far from the previous one (as in the original game)
GAP_MIN, GAP_MAX = 100, GROUND - 100   # range of gap centres
BUMPS = 12            # Gaussian bumps per sense number
INPUTS = 4 * BUMPS


class Flappy:
    """n games side by side. Each has 3 pipes on screen at most; pipes recycle to the right when they leave."""

    def __init__(self, n: int, seed: int = 0, cap: int = 100):
        self.n, self.cap = n, cap
        self.rng = np.random.default_rng(seed)
        self.reset(np.ones(n, bool))

    def _gaps(self, k):
        return self.rng.uniform(GAP_MIN, GAP_MAX, k)

    def reset(self, rows):
        k = int(rows.sum())
        if not hasattr(self, "y"):
            self.y = np.zeros(self.n); self.vy = np.zeros(self.n); self.score = np.zeros(self.n, int)
            self.px = np.zeros((self.n, 3)); self.pg = np.zeros((self.n, 3)); self.frames = np.zeros(self.n, int)
            self.dead = np.zeros(self.n, bool); self.passed = np.zeros((self.n, 3), bool)
        self.y[rows] = H * 0.4; self.vy[rows] = 0; self.score[rows] = 0; self.frames[rows] = 0; self.dead[rows] = False
        self.px[rows] = W + 40 + SPACING * np.arange(3)
        g0 = self._gaps(k)
        g1 = np.clip(g0 + self.rng.uniform(-STEP, STEP, k), GAP_MIN, GAP_MAX)
        g2 = np.clip(g1 + self.rng.uniform(-STEP, STEP, k), GAP_MIN, GAP_MAX)
        self.pg[rows] = np.stack([g0, g1, g2], 1)
        self.passed[rows] = False

    def _next(self):
        """Index of the first pipe whose right edge is still ahead of the bird's left edge, and the one after."""
        ahead = self.px + PIPE_W > BIRD_X - BIRD_R
        order = np.argsort(np.where(ahead, self.px, 1e9), axis=1)
        return order[:, 0], order[:, 1]

    def state(self):
        """Raw numbers: bird height vs next gap centre, speed, distance to next pipe, next gap vs the one after."""
        r = np.arange(self.n); i, j = self._next()
        dy = self.y - self.pg[r, i]
        dx = self.px[r, i] - BIRD_X
        dg = self.pg[r, j] - self.pg[r, i]
        return dy, self.vy.copy(), dx, dg

    def senses(self):
        dy, vy, dx, dg = self.state()
        cols = [bumps(dy, -200, 200), bumps(vy, FLAP_V, MAX_FALL), bumps(dx, -PIPE_W, SPACING + 40), bumps(dg, -250, 250)]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def step(self, flap):
        """Advance one frame for live games. Returns (newly dead rows, newly passed pipe count)."""
        live = ~self.dead
        f = (np.asarray(flap) == 1) & live
        self.vy = np.where(f, FLAP_V, np.minimum(self.vy + GRAVITY, MAX_FALL))
        self.y = np.where(live, self.y + self.vy, self.y)
        self.px[live] -= SPEED
        self.frames[live] += 1
        # passing: the pipe's right edge goes behind the bird
        newp = live[:, None] & ~self.passed & (self.px + PIPE_W < BIRD_X - BIRD_R)
        self.passed |= newp; self.score += newp.sum(1)
        # recycle pipes that left the screen
        gone = self.px + PIPE_W < 0
        if gone.any():
            rr, cc = np.nonzero(gone)
            last = self.px[rr].argmax(1)
            self.px[rr, cc] = self.px[rr, last] + SPACING
            prev = self.pg[rr, last]
            self.pg[rr, cc] = np.clip(prev + self.rng.uniform(-STEP, STEP, len(rr)), GAP_MIN, GAP_MAX)
            self.passed[rr, cc] = False
        # deaths: ground, ceiling, pipe overlap
        hit = (self.y + BIRD_R >= GROUND) | (self.y - BIRD_R <= 0)
        inx = (self.px < BIRD_X + BIRD_R) & (self.px + PIPE_W > BIRD_X - BIRD_R)
        out = (self.y[:, None] - BIRD_R < self.pg - GAP / 2) | (self.y[:, None] + BIRD_R > self.pg + GAP / 2)
        hit |= (inx & out).any(1)
        hit |= self.score >= self.cap
        died = live & hit
        self.dead |= died
        return died


def bumps(v, lo, hi, k=BUMPS):
    """Population code: k Gaussian bumps spread over [lo, hi]; values outside clip to the end bumps."""
    c = np.linspace(lo, hi, k); w = (hi - lo) / (k - 1)
    v = np.clip(np.asarray(v, float), lo, hi)[:, None]
    return np.exp(-0.5 * ((v - c) / w) ** 2)


def teacher(g: Flappy):
    """Scripted rule: flap when the bird is lower than the target height and not already rising.
    The target sits a bit below the gap centre (a flap overshoots upward)."""
    dy, vy, dx, dg = g.state()
    target = 35.0  # tuned on dev seed 7 (40 games): 200/200 pipes every game
    return ((dy > target) & (vy >= 0)).astype(np.int64)
