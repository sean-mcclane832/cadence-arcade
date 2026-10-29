"""Super Mario Bros. World 1-1 body for a Cadence brain, and the look-ahead coach.

The game is the real NES cartridge running in nes-py (gym-super-mario-bros ships the ROM inside
the pip package; no ROM bytes live in this repo). One decision holds a button combination for
FRAMES emulator frames.

What the brain sees (`Game.observation`): the playfield below the score bar, grey, averaged in
8 x 8 pixel blocks to 26 x 32, as distance from the sky's grey, for the present frame and the frame
of the previous decision. 1,664 numbers in [0, 1]. Nothing else: no RAM, no positions, no enemy list.

The coach (`Game.coach`) is NOT a learner. At every decision it saves the emulator, tries a fixed
menu of short button plans (run, run-jump of several heights, wait, back off...) for HORIZON
decisions each, and returns the first button of the plan that ends furthest right without dying.
It reads the emulator (Mario's x position and death flag) to score its plans; the brain never
does.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv  # noqa: E402

FRAMES = 4
HORIZON = int(__import__("os").environ.get("MARIO_HORIZON", 16))
CRITICAL_PX, CRITICAL_WEIGHT = 50.0, 10.0
FAST_GAIN = 140  # px over HORIZON decisions; full speed is ~150-190
# NES pad bits: right 0x80, left 0x40, down 0x20, up 0x10, start 0x08, select 0x04, B 0x02, A 0x01
ACTIONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40]
ACTION_NAMES = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
N_ACTIONS = len(ACTIONS)
TOP, ROWS, COLS, BLOCK = 32, 26, 32, 8
N_FRAME = ROWS * COLS
SKY = (0.299 * 92 + 0.587 * 148 + 0.114 * 252) / 255.0  # grey level of the 1-1 sky
SENSES = __import__("os").environ.get("MARIO_SENSES", "blind")
# "senses" input (MARIO_SENSES=senses): eyes at 4 x 4 px blocks for the present frame, 3 signed
# frame differences at 8 x 8, all per-frame normalized; plus proprioception of Mario's own body.
FINE = 4
N_FINE = (ROWS * BLOCK // FINE) * (COLS * BLOCK // FINE)  # 52 x 64 = 3,328
N_PROPRIO_ONE, PROPRIO_COPIES = 38, 8
N_IN = 2 * N_FRAME if SENSES == "blind" else N_FINE + 3 * N_FRAME + N_PROPRIO_ONE * PROPRIO_COPIES
LEVEL_END = 3161  # x of the flagpole in 1-1 (approx; the flag flag is what counts)

NOOP, RIGHT, RJUMP, RUN, RUNJUMP, JUMP, LEFT = range(7)


def _plans(h: int = HORIZON) -> list[list[int]]:
    out = [[RUN] * h]
    for k in (2, 3, 5, 8, 12):
        out.append([RUNJUMP] * k + [RUN] * (h - k))
    for m, k in ((2, 8), (4, 8), (1, 5)):
        out.append([RUN] * m + [RUNJUMP] * k + [RUN] * (h - m - k))
    for k in (3, 8):
        out.append([RJUMP] * k + [RIGHT] * (h - k))
    out.append([RIGHT] * h)
    for k in (2, 5):
        out.append([NOOP] * k + [RUN] * (h - k))
    for k in (3, 6):
        out.append([LEFT] * k + [RUN] * (h - k))
    out.append([JUMP] * 8 + [NOOP] * (h - 8))
    return out


PLANS = _plans()


def _lean_plans(h: int = HORIZON) -> list[list[int]]:
    """A 4-button menu (run, run+jump, wait, back off). The full menu has near-equal buttons
    (right vs run, right+jump vs run+jump) whose ties make the labels conflict; this one keeps
    waiting and backing off, which the 2-button coach lacked and needed."""
    out = [[RUN] * h]
    for k in (2, 3, 5, 8, 12):
        out.append([RUNJUMP] * k + [RUN] * (h - k))
    for m, k in ((2, 8), (4, 8), (1, 5)):
        out.append([RUN] * m + [RUNJUMP] * k + [RUN] * (h - m - k))
    for k in (3, 8):
        out.append([RUNJUMP] * k + [NOOP] * (h - k))
    for k in (2, 5, 10):
        out.append([NOOP] * k + [RUN] * (h - k))
    for k in (3, 6):
        out.append([LEFT] * k + [RUN] * (h - k))
    out.append([NOOP] * 3 + [RUNJUMP] * 8 + [RUN] * (h - 11))
    out.append([LEFT] * 3 + [RUNJUMP] * 8 + [RUN] * (h - 11))
    out.append([NOOP] * 6 + [RUNJUMP] * 6 + [RUN] * (h - 12))
    # run first, then a short hop: low ceilings over a gap (1-2 at x ~1,270) need a tap jump at the edge
    for m in (5, 7, 9):
        for k in (1, 2):
            out.append([RUN] * m + [RUNJUMP] * k + [RUN] * (h - m - k))
    if h >= 24:  # long back-offs for a running start under a low ceiling (1-2)
        for k, m in ((8, 4), (10, 6), (12, 4)):
            out.append([LEFT] * k + [RUN] * m + [RUNJUMP] * 8 + [RUN] * (h - k - m - 8))
        out.append([LEFT] * 6 + [RUN] * 2 + [RUNJUMP] * 12 + [RUN] * (h - 20))
    return out


LEAN_PLANS = _lean_plans()
MENUS = {"full": PLANS, "lean": LEAN_PLANS, "lean24": _lean_plans(24)}
LEVEL_ENDS = {(1, 1): 3161}


def background_grey(grey: np.ndarray) -> float:
    """The commonest grey level of the playfield (sky, or black underground / in castles)."""
    q = np.clip(np.round(grey * 255), 0, 255).astype(np.int64).reshape(-1)
    return float(np.bincount(q, minlength=256).argmax()) / 255.0


def frame_features(screen: np.ndarray, bg: float | None = SKY) -> np.ndarray:
    """240 x 256 x 3 uint8 screen -> 26 x 32 grey blocks (8 x 8 px each), as distance from the sky
    grey, flattened. Sky reads 0 and things (ground, pipes, Goombas, Mario) light up, the way an
    Atari screen is mostly black. Raw [0, 1] pixels on a bright sky gave a brain whose answer did
    not depend on the screen at all (sweep.py)."""
    s = screen[TOP:TOP + ROWS * BLOCK].astype(np.float32)
    grey = (s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114) / 255.0
    if bg is None:  # level-agnostic: distance from this frame's own background grey
        bg = background_grey(grey)
    blocks = grey.reshape(ROWS, BLOCK, COLS, BLOCK).mean(axis=(1, 3))
    return np.abs(blocks - bg).reshape(-1)


def _norm_blocks(grey: np.ndarray, block: int) -> np.ndarray:
    """Level-agnostic: distance from this frame's background grey, scaled by the frame's own
    contrast (95th percentile), so overground, underground and castles read alike."""
    bg = background_grey(grey)
    r, c = grey.shape[0] // block, grey.shape[1] // block
    b = np.abs(grey.reshape(r, block, c, block).mean(axis=(1, 3)) - bg)
    return np.clip(b / max(float(np.percentile(b, 95)), 0.08), 0, 1).reshape(-1)


def _pop(v: float, lo: float, hi: float, n: int) -> np.ndarray:
    """Population code: n bins over [lo, hi], a soft bump around v."""
    centres = np.linspace(lo, hi, n); w = (hi - lo) / (n - 1)
    return np.exp(-0.5 * ((np.clip(v, lo, hi) - centres) / (0.6 * w)) ** 2)


def proprio(env) -> np.ndarray:
    """Mario's own body only, from RAM: x speed, y speed, float state (ground / jumping / falling /
    pole), jump still rising, facing, power state, hurt and star timers, time left. No enemies, tiles
    or pits: the world comes from the eyes."""
    ram = env.ram
    xs = float(np.int8(np.uint8(ram[0x0057]))); ys = float(np.int8(np.uint8(ram[0x009F])))
    fl = int(ram[0x001D]); face = int(ram[0x0033]); pw = int(ram[0x0756])
    v = [_pop(xs, -40, 40, 9), _pop(ys, -5, 5, 9), np.eye(4)[min(fl, 3)], [1.0 if ys < 0 else 0.0],
         [1.0 if face == 1 else 0.0, 1.0 if face == 2 else 0.0], np.eye(3)[min(pw, 2)],
         [1.0 if ram[0x079E] > 0 else 0.0, 1.0 if ram[0x079F] > 0 else 0.0],
         _pop(float(env._time) / 400.0, 0, 1, 8)]
    return np.concatenate([np.asarray(a, dtype=np.float32) for a in v])


class Game:
    """One life in World 1-1. A new Game is a new attempt from the start."""

    def __init__(self, noop_start: int = 0, level: tuple = (1, 1), menu: str = "full",
                 bg_per_frame: bool = False):
        self.level = tuple(level)
        self.plans = MENUS[menu]
        self.env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=self.level)
        screen = self.env.reset()
        # background grey: None = every frame's own commonest grey (level-agnostic; on 1-1 that
        # is the sky grey SKY, so last night's brain sees the same numbers there)
        self.bg = None if bg_per_frame else SKY
        for _ in range(noop_start):  # optional start jitter: idle frames before the first decision
            screen, _, _, _ = self.env.step(0)
        self.cur = frame_features(screen, self.bg)
        self.prev = self.cur.copy()
        if SENSES != "blind":
            self._sense(screen, first=True)
        self.screen = screen.copy()
        self.done = False
        self.flag = False
        self.decisions = 0
        self.raw_x = int(self.env._x_position)
        self.off = 0  # x carried across an area change (exit pipe -> the flag area restarts x near 0)
        self.x = self.raw_x
        self.max_x = self.x
        self.stuck = 0

    # -- the brain's view -------------------------------------------------------------------
    def _sense(self, screen, first=False):
        s = screen[TOP:TOP + ROWS * BLOCK].astype(np.float32)
        grey = (s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114) / 255.0
        self.fine = _norm_blocks(grey, FINE)
        coarse = _norm_blocks(grey, BLOCK)
        self.hist = [coarse] * 4 if first else [coarse] + self.hist[:3]

    def observation(self) -> np.ndarray:
        if SENSES == "blind":
            return np.concatenate([self.cur, self.prev])
        h = self.hist
        diffs = [0.5 + 0.5 * (h[0] - h[k]) for k in (1, 2, 3)]
        return np.concatenate([self.fine, *diffs, np.tile(proprio(self.env), PROPRIO_COPIES)]).astype(np.float64)

    # -- the body ---------------------------------------------------------------------------
    def step(self, action: int, keep_frames: list | None = None) -> float:
        """Hold ACTIONS[action] for FRAMES frames. Returns the x progress of this decision."""
        x0 = self.x
        screen = self.screen
        for _ in range(FRAMES):
            screen, _, done, info = self.env.step(ACTIONS[action])
            if keep_frames is not None:
                keep_frames.append(screen.copy())
            if done:
                self.done = True
                self.flag = bool(info["flag_get"])
                break
        self.screen = screen.copy()
        self.prev, self.cur = self.cur, frame_features(screen, self.bg)
        if SENSES != "blind":
            self._sense(screen)
        raw = int(self.env._x_position)
        if raw < self.raw_x - 600 and not self.done:  # new area (pipe), not a step back
            self.off += self.raw_x + 16 - raw
        self.raw_x = raw
        self.x = raw + self.off
        if self.x > self.max_x:
            self.max_x, self.stuck = self.x, 0
        else:
            self.stuck += 1
        self.decisions += 1
        return float(self.x - x0)

    # -- the coach --------------------------------------------------------------------------
    def _simulate(self, plan: list[int]) -> float:
        env = self.env
        for a in plan:
            byte = ACTIONS[a]
            for _ in range(FRAMES):
                env._frame_advance(byte)
                if env._flag_get:
                    return 1e6 + env._x_position
                if env._is_dying or env._is_dead:
                    return -1e5 + env._x_position
                if env.ram[0x000E] in (2, 3):  # entering a pipe (1-2's exit to the flag area)
                    return 5e5 + env._x_position
        return float(env._x_position)

    def coach_scores(self) -> np.ndarray:
        env = self.env
        saved = (env._time_last, env._x_position_last, env.done)
        env._backup()
        scores = np.empty(len(self.plans))
        for i, plan in enumerate(self.plans):
            env._restore()
            scores[i] = self._simulate(plan)
        env._restore()
        env._time_last, env._x_position_last, env.done = saved
        return scores

    def coach(self) -> int:
        return self.coach_weighted()[0]

    def coach_weighted(self) -> tuple[int, float]:
        """The coach's button and how much it matters. Weight CRITICAL_WEIGHT when the best plan
        that starts with any OTHER button ends more than CRITICAL_PX behind (in practice: every
        other first button dies), else 1. Fast-path running on open ground is weight 1."""
        # Fast path: if plain running survives the horizon at near full speed, run. (The full
        # menu prefers running within 2 px anyway; this only skips the search on open ground.)
        env = self.env
        saved = (env._time_last, env._x_position_last, env.done)
        x0 = env._x_position
        env._backup()
        run = self._simulate(self.plans[0])
        env._restore()
        env._time_last, env._x_position_last, env.done = saved
        if run >= 1e6 or (run > 0 and run - x0 >= FAST_GAIN):
            return self.plans[0][0], 1.0
        scores = self.coach_scores()
        best = scores.max()
        # the first plan (menu order) within 2 pixels of the best: run unless something is better
        i = int(np.flatnonzero(scores >= best - 2.0)[0])
        label = self.plans[i][0]
        first = np.array([p[0] for p in self.plans])
        other = scores[first != label]
        alt = other.max() if len(other) else best
        return label, (CRITICAL_WEIGHT if best - alt > CRITICAL_PX else 1.0)

    def close(self) -> None:
        self.env.close()
