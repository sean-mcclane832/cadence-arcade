"""Shared pieces for coach-free work (COACHFREE.md). Worlds 6 and 7 are the coach-free development ground (Sean, 11/5):
they NEVER get coach labels or coach demonstrations. World 4 stays the clean never-coached final test (not touched here).

- DEV: the 8 dev levels; TRAIN: worlds 1-3 (coached, used for anchors and the forgetting check).
- idle(): this process at Idle priority (other agents run 8 training runs on this PC).
- workers(): 2 worker processes, 1 if a game runs (memory/gamecheck.py).
- no_coach(): makes any coach call in this process raise, so a bug can't leak coach labels into worlds 6-7.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coldstart")); sys.path.insert(0, str(HERE.parent / "memory"))
import cs  # noqa: E402
import gamecheck  # noqa: E402

mem = cs.mem
SEL, HELD, POOL = cs.SEL, cs.HELD, cs.POOL
DEV = [f"{w}-{s}" for w in (6, 7) for s in (1, 2, 3, 4)]
TRAIN = cs.TRAIN
FORBIDDEN_WORLDS = (4,)  # never run anything on World 4 from this folder
BRAINS = HERE / "brains0"
CKPTS = {"CatoA": BRAINS / "CatoA_ck0091.npz", "CatoB": BRAINS / "CatoB_ck0084.npz", "CiceroB": BRAINS / "CiceroB_ck0093.npz"}


def check_level(L):
    w = int(str(L).split("-")[0])
    assert w not in FORBIDDEN_WORLDS, f"World 4 is the held-out final test: {L} refused"
    return L


def idle():
    try:  # Windows SetPriorityClass(IDLE_PRIORITY_CLASS); spawned children inherit it
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        if not k.SetPriorityClass(k.GetCurrentProcess(), 0x40):
            raise OSError(ctypes.get_last_error())
    except Exception as e:  # pragma: no cover
        print("idle priority failed:", e, flush=True)


def no_coach():
    import game
    def refuse(*a, **k):
        raise RuntimeError("coach call in coach-free code")
    game.Game.coach = refuse; game.Game.coach_scores = refuse; game.Game.coach_weighted = refuse


def worker_init():
    idle()
    mem.worker_init()
    no_coach()


def workers():
    return 1 if gamecheck.gaming() else 2
