"""Is the 2 KB NES RAM a complete save state for SMB physics at decision boundaries?

From a root state: play branch A with nes-py's real restore and record RAM after every decision. Then play some
other branch B, write the root's RAM back into the emulator (CPU/PPU left as B left them), replay A, and compare
RAM after every decision. If every trial matches, RAM snapshots can stand in for extra save-state slots.
    python ramtest.py --level 1-1 --trials 40
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="1-1"); ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--advance", type=int, default=60, help="decisions of coach play between roots")
    a = ap.parse_args()
    harness.below_normal()
    import game
    from game import ACTIONS, FRAMES
    lvl = tuple(int(v) for v in a.level.split("-")); assert lvl[0] != 4
    g = game.Game(level=lvl, menu="lean", bg_per_frame=True)
    env = g.env; rng = np.random.default_rng(0)
    menu = [1, 2, 3, 4, 0, 6, 5]
    ok = bad = 0; firstbad = []

    def play(seq):
        out = []
        for act in seq:
            for _ in range(FRAMES):
                env._frame_advance(ACTIONS[act])
            out.append(env.ram.copy())
        return out

    for tr in range(a.trials):
        if g.done:
            break
        saved = (env._time_last, env._x_position_last, env.done)
        env._backup(); root = env.ram.copy()
        A = list(rng.choice(menu, 16)); B = list(rng.choice(menu, int(rng.integers(1, 16))))
        ra = play(A)
        env._restore(); play(B)
        env.ram[:] = root
        rb = play(A)
        diffs = [int((x != y).sum()) for x, y in zip(ra, rb)]
        if max(diffs) == 0:
            ok += 1
        else:
            if bad < 4:
                k = next(i for i, d in enumerate(diffs) if d)
                idx = np.flatnonzero(ra[k] != rb[k]); print('  diff addrs', [hex(i) for i in idx[:24]], 'at dec', k, 'later max', max(diffs), 'last', diffs[-1])
            bad += 1; firstbad.append((tr, int(g.x), diffs.index(next(d for d in diffs if d)), max(diffs)))
        env._restore(); env._time_last, env._x_position_last, env.done = saved
        for _ in range(a.advance // 4):  # move the root along (cheap: run / run+jump)
            if g.done:
                break
            g.step(int(rng.choice([3, 4, 3, 3])))
    print(f"{a.level}: identical {ok}, differ {bad}; first bad (trial, x, decision, bytes): {firstbad[:8]}")


if __name__ == "__main__":
    main()
