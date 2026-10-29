"""Super Mario Bros. World 1-1 body for a Cadence brain, and the look-ahead coach.

The game is the real NES cartridge running in nes-py (gym-super-mario-bros ships the ROM inside
the pip package; no ROM bytes live in this repo). One decision holds a button combination for
FRAMES emulator frames.

What the brain sees (`Game.observation`): the playfield below the score bar, grey, averaged in
4 x 4 blocks to 52 x 64, for the present frame and the frame of the previous decision. 6,656 numbers
in [0, 1]. Nothing else: no RAM, no positions, no enemy list.

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
HORIZON = 16
FAST_GAIN = 140  # px over HORIZON decisions; full speed is ~150-190
# NES pad bits: right 0x80, left 0x40, down 0x20, up 0x10, start 0x08, select 0x04, B 0x02, A 0x01
ACTIONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40]
ACTION_NAMES = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
N_ACTIONS = len(ACTIONS)
TOP, ROWS, COLS, BLOCK = 32, 52, 64, 4
N_FRAME = ROWS * COLS
N_IN = 2 * N_FRAME
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


def frame_features(screen: np.ndarray) -> np.ndarray:
    """240 x 256 x 3 uint8 screen -> 52 x 64 grey blocks in [0, 1] (flattened)."""
    s = screen[TOP:TOP + ROWS * BLOCK].astype(np.float32)
    grey = s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114
    blocks = grey.reshape(ROWS, BLOCK, COLS, BLOCK).mean(axis=(1, 3))
    return (blocks / 255.0).reshape(-1)


class Game:
    """One life in World 1-1. A new Game is a new attempt from the start."""

    def __init__(self, noop_start: int = 0):
        self.env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
        screen = self.env.reset()
        for _ in range(noop_start):  # optional start jitter: idle frames before the first decision
            screen, _, _, _ = self.env.step(0)
        self.cur = frame_features(screen)
        self.prev = self.cur.copy()
        self.screen = screen.copy()
        self.done = False
        self.flag = False
        self.decisions = 0
        self.x = int(self.env._x_position)
        self.max_x = self.x
        self.stuck = 0

    # -- the brain's view -------------------------------------------------------------------
    def observation(self) -> np.ndarray:
        return np.concatenate([self.cur, self.prev])

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
        self.prev, self.cur = self.cur, frame_features(screen)
        self.x = int(self.env._x_position)
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
        return float(env._x_position)

    def coach_scores(self) -> np.ndarray:
        env = self.env
        saved = (env._time_last, env._x_position_last, env.done)
        env._backup()
        scores = np.empty(len(PLANS))
        for i, plan in enumerate(PLANS):
            env._restore()
            scores[i] = self._simulate(plan)
        env._restore()
        env._time_last, env._x_position_last, env.done = saved
        return scores

    def coach(self) -> int:
        # Fast path: if plain running survives the horizon at near full speed, run. (The full
        # menu prefers running within 2 px anyway; this only skips the search on open ground.)
        env = self.env
        saved = (env._time_last, env._x_position_last, env.done)
        x0 = env._x_position
        env._backup()
        run = self._simulate(PLANS[0])
        env._restore()
        env._time_last, env._x_position_last, env.done = saved
        if run >= 1e6 or (run > 0 and run - x0 >= FAST_GAIN):
            return PLANS[0][0]
        scores = self.coach_scores()
        best = scores.max()
        # the first plan (menu order) within 2 pixels of the best: run unless something is better
        i = int(np.flatnonzero(scores >= best - 2.0)[0])
        return PLANS[i][0]

    def close(self) -> None:
        self.env.close()
