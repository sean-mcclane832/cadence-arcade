"""Shared pieces for the cold-start night (COLDSTART.md): level lists, start sets, per-level coach horizon.

Start sets are World 1's (concepts/cplay.py), used unchanged on every level: SEL (24, selection / learning curve),
HELD (50, never trained or picked on), POOL (DAgger starts). startcheck.py verifies every start is playable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "memory")); sys.path.insert(0, str(HERE.parent / "concepts")); sys.path.insert(0, str(HERE.parent))
import mem  # noqa: E402

C = mem.C
SEL, HELD, POOL = C.SEL, C.HELD, C.POOL
TRAIN = [f"{w}-{s}" for w in (1, 2, 3) for s in (1, 2, 3, 4)]
TEST = [f"4-{s}" for s in (1, 2, 3, 4)]
ALL = TRAIN + TEST
DATA = HERE / "data"
HORIZON_FILE = DATA / "horizon.json"  # {level: coach horizon} for levels where HORIZON-16 was not enough


def deep_levels():
    return json.loads(HORIZON_FILE.read_text()) if HORIZON_FILE.exists() else {}


def patch_game(deep=None):
    """Give the coach a deeper plan menu (game._lean_plans(h)) on chosen levels. Same buttons, longer plans."""
    import game
    deep = deep_levels() if deep is None else deep
    if not deep or getattr(game.Game, "_cs_patched", False):
        return
    menus = {tuple(int(v) for v in L.split("-")): game._lean_plans(int(h)) for L, h in deep.items()}
    init = game.Game.__init__

    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        if self.level in menus:
            self.plans = menus[self.level]
    game.Game.__init__ = __init__
    game.Game._cs_patched = True


def worker_init(deep=None):
    mem.worker_init()
    patch_game(deep)
