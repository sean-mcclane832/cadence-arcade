"""Is every start playable on every level? A start = idle frames before the first decision (World 1's sets).
Steps one env per level with NOOP up to the largest start and records the first frame the game ends (an enemy
reaching idle Mario), so starts at or past it would crash the driver. Writes data/startcheck.json.
    python startcheck.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402


def main():
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    need = max(max(cs.SEL), max(cs.HELD), int(cs.POOL.max())) + 1
    out = {}
    for L in cs.ALL:
        env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=tuple(int(v) for v in L.split("-")))
        env.reset(); dead_at = None
        for f in range(1, need + 1):
            _, _, done, _ = env.step(0)
            if done:
                dead_at = f; break
        x0 = int(env._x_position) if dead_at is None else None
        env.close()
        bad = sorted(s for s in set(cs.SEL) | set(cs.HELD) | set(int(v) for v in cs.POOL) if dead_at is not None and s >= dead_at)
        out[L] = dict(idle_frames_checked=need, game_ends_at=dead_at, bad_starts=bad, x_after_idle=x0)
        print(L, out[L], flush=True)
    cs.DATA.mkdir(exist_ok=True)
    (cs.DATA / "startcheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
