"""Play / label / evaluate for the concepts run (CONCEPTS.md). Same game, coach and body sense as
mario/dagger.py (imported, unchanged), plus one new sense:

  ram       CHEAT (ceiling test): a what-is-where map built from RAM, 13 rows x 16 columns of 16 px
            tiles around Mario (4 columns behind, 11 ahead, all 13 tile rows of the screen), channels
            solid / enemy / pit / powerup / mario / empty, plus the enemy channel of the previous
            decision (motion). 7 x 208 = 1,456 numbers. With the body sense (336), NO screen.
  ramscreen the same map plus the normal eyes + body (2,000 + 1,456).
  body      exactly dagger.py's senses (eyes + body), for the control.

Death clinic support: task["prefix"] replays buttons first (deterministic emulator), so "rewind to k
decisions before a death" = prefix of the brain's own buttons up to death - k.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dagger as D  # noqa: E402

SEL = tuple(s for s in range(0, 80, 3) if not 32 <= s <= 41)
TEST = tuple(range(32, 42))
WIDE = tuple(range(100, 140, 2))
HELD = TEST + tuple(range(80, 100)) + WIDE
POOL = np.array([s for s in range(80) if s not in TEST])

ROWS, COLS, BEHIND = 13, 16, 4
N_CELLS = ROWS * COLS
N_MAP = 7 * N_CELLS
COIN_TILES = (0xC2, 0xC3)  # coins: not solid
ENEMY_SLOTS = 5


def concept_map(ram) -> np.ndarray:
    """(6, 13, 16) current map from RAM. Columns are 16 px tiles in LEVEL coordinates, from 4 tiles behind
    Mario's tile to 11 ahead. Rows are the 13 block-buffer rows (screen y 32..240)."""
    m = np.zeros((6, ROWS, COLS), np.float32)
    mx = int(ram[0x006D]) * 256 + int(ram[0x0086])  # Mario level x (left edge)
    c0 = (mx + 8) // 16 - BEHIND
    for j in range(COLS):
        c = c0 + j
        if c < 0:
            m[0, :, j] = 1.0  # left of the level start: a wall
            continue
        base = 0x500 + ((c // 16) % 2) * 208 + (c % 16)
        col = np.array([int(ram[base + r * 16]) for r in range(ROWS)])
        solid = (col != 0) & (col != COIN_TILES[0]) & (col != COIN_TILES[1])
        m[0, :, j] = solid
        if not solid[11] and not solid[12]:
            m[2, 11:, j] = 1.0  # pit: nothing to stand on in the two bottom rows

    def put(ch, x, ytop, h=16):
        j = (x + 8) // 16 - c0
        if 0 <= j < COLS:
            for yy in (ytop, ytop + h - 1):
                r = (yy - 32) // 16
                if 0 <= r < ROWS:
                    m[ch, r, j] = 1.0

    for i in range(ENEMY_SLOTS):
        if ram[0x000F + i] and int(ram[0x00B6 + i]) == 1:
            put(1, int(ram[0x006E + i]) * 256 + int(ram[0x0087 + i]), int(ram[0x00CF + i]) + 8)
    if ram[0x0014] and int(ram[0x00BB]) == 1:
        put(3, int(ram[0x0073]) * 256 + int(ram[0x008C]), int(ram[0x00D4]) + 8)
    if int(ram[0x00B5]) == 1:
        big = int(ram[0x0754]) == 0  # 0x0754: 0 = big, 1 = small
        put(4, mx, int(ram[0x00CE]) + (0 if big else 16), 32 if big else 16)  # 0x00CE = top of a 32 px box
    m[5] = 1.0 - np.clip(m[:5].sum(0), 0, 1)
    return m


class CBody(D.Body):
    def __init__(self, start, senses, menu, level=(1, 1)):
        super().__init__(start, "body" if senses in ("ram", "ramscreen") else senses, menu, level)
        self.csenses = senses
        self.eprev = None

    def obs(self):
        o = super().obs()  # eyes (2 x 832) + body (336)
        if self.csenses not in ("ram", "ramscreen"):
            return o
        m = concept_map(self.g.env.ram)
        if self.eprev is None:
            self.eprev = m[1]
        cm = np.concatenate([m.reshape(-1), self.eprev.reshape(-1)])
        if self.csenses == "ram":
            return np.concatenate([o[-336:], cm])
        return np.concatenate([o, cm])

    def step(self, a):
        if self.csenses in ("ram", "ramscreen"):
            self.eprev = concept_map(self.g.env.ram)[1]
        super().step(a)


def n_inputs(senses):
    if senses == "ram":
        return 336 + N_MAP
    if senses == "ramscreen":
        return 2000 + N_MAP
    return D.n_inputs(senses)


def view(o, have, want):
    """Slice a stored observation to the senses a brain reads. ramscreen = eyes 1,664 | body 336 | map 1,456."""
    if not want or want == have:
        return o
    assert have == "ramscreen", (have, want)
    if want == "ram":
        return o[..., 1664:]
    if want == "body":
        return o[..., :2000]
    raise ValueError(want)


def worker_init():
    D._worker_init()


def task(policy, senses, level, start, mode="eval", **kw):
    from game import MENUS
    mb = sorted({a for p in MENUS["lean"] for a in p})
    lvl = tuple(int(v) for v in level.split("-")) if isinstance(level, str) else tuple(level)
    return dict(senses=senses, menu="lean", kind="cadence", menu_buttons=mb, level=lvl, mode=mode,
                start=int(start), policy=str(policy), **kw)


def _play(task, pol, label_at=None, rng=None):
    """dagger._play with CBody (tie-aware labels as dagger's tiew). Also returns x per decision."""
    b = CBody(task["start"], task["senses"], task["menu"], task.get("level", (1, 1)))
    for a in task.get("prefix", []):
        b.step(a)
        if b.g.done:
            break
    g = b.g
    obs, labels, weights, acts, xs = [], [], [], [], []
    script = task.get("script")
    while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
        i = len(acts)
        if script is not None and i >= len(script):
            break
        o = b.obs()
        want = label_at == "all" or (label_at is not None and i in label_at)
        lab, wt = g.coach_weighted() if want else (-1, 0.0)
        if want and script is not None and task.get("tiew") and script[i] != lab:
            sc = g.coach_scores(); first = np.array([p_[0] for p_ in g.plans])
            mine = sc[first == script[i]]
            if len(mine) and sc.max() - mine.max() <= 2.0:
                lab, wt = int(script[i]), 1.0
        if script is not None:
            a = script[i]
        elif task["mode"] == "coach":
            a = lab if rng.random() >= task.get("eps", 0.0) else int(rng.choice(task["menu_buttons"]))
        else:
            a = int(np.asarray(pol.predict(view(o, task["senses"], task.get("pol_senses"))[None])).reshape(-1)[0])
        if want or task.get("keep_obs"):
            obs.append(np.round(o * 255).astype(np.uint8)); labels.append(lab); weights.append(wt)
        acts.append(a); xs.append(int(g.x))
        b.step(a)
    out = dict(start=task["start"], prefix_len=len(task.get("prefix", [])), max_x=int(g.max_x), flag=bool(g.flag),
               died=bool(g.done and not g.flag), decisions=g.decisions, actions=acts, xs=xs,
               end="flag" if g.flag else ("died" if g.done else "stuck"))
    n = n_inputs(task["senses"])
    out["obs"] = np.array(obs, np.uint8).reshape(-1, n)
    out["labels"] = np.array(labels, np.int64); out["weights"] = np.array(weights)
    g.close()
    return out


def episode(task):
    rng = np.random.default_rng(task.get("seed", 0))
    if task["mode"] == "coach":
        return _play(task, None, "all", rng)
    pol = D._policy(task["policy"], task["kind"])
    played = _play(task, pol, None, rng)
    if task["mode"] == "eval":
        played.pop("obs"); return played
    if task.get("keep_obs"):
        played["own_obs"] = played["obs"]
    acts = played["actions"]; n = len(acts)
    last_all, every = task.get("last_all", 60), task.get("every", 3)
    idx = {i for i in range(n) if i >= n - last_all or i % every == 0}
    lab = _play(dict(task, script=acts, keep_obs=False), None, idx, rng)
    order = sorted(idx)
    played.update(obs=lab["obs"], labels=lab["labels"], weights=lab["weights"],
                  own=np.array([acts[i] for i in order[:len(lab["labels"])]], np.int64))
    return played
