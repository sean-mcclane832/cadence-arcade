"""Plug a smartcoach variant into every Game a worker process builds (DAgger labels, coach demos).

    from plug import worker_init          # coach/ on sys.path
    ProcessPoolExecutor(..., initializer=worker_init, initargs=(deep, "verify", {}))

name "base" changes nothing (game.Game's own coach).
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def patch_game_class(name="base", opts=None):
    import game
    import smartcoach
    if name == "base" or getattr(game.Game, "_smart_patched", None) == name:
        return
    init = game.Game.__init__

    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        if self.level[0] == 4:  # held-out test: brains may play it, the coach may not label it
            def refuse(*_a, **_k):
                raise RuntimeError("World 4 is the held-out test: no coach labels there")
            self.coach_weighted = self.coach_scores = self.coach = refuse
            return
        smartcoach.install(self, name, **(opts or {}))
    game.Game.__init__ = __init__
    game.Game._smart_patched = name


def worker_init(deep=None, name="base", opts=None):
    sys.path.insert(0, str(HERE.parent / "coldstart"))
    import cs
    cs.worker_init(deep)
    patch_game_class(name, opts)
