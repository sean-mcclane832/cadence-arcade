"""Worlds 6-7: start check + first-sight baseline (greedy, act mode, 50 HELD starts) for the cold-start checkpoints.
No coach anywhere (cf.no_coach in every worker). Idle priority, 2 workers (1 if a game runs).
    python firstsight.py            -> runs/firstsight/startcheck.json, firstsight.json, firstsight.md
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf  # noqa: E402

OUT = HERE / "runs" / "firstsight"


def startcheck():
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    need = max(max(cf.SEL), max(cf.HELD), int(cf.POOL.max())) + 1
    out = {}
    for L in cf.DEV:
        env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=tuple(int(v) for v in L.split("-")))
        env.reset(); dead_at = None; x_start = int(env._x_position)
        for f in range(1, need + 1):
            _, _, done, _ = env.step(0)
            if done:
                dead_at = f; break
        x0 = int(env._x_position) if dead_at is None else None
        world, stage = int(env.ram[0x075F]) + 1, int(env.ram[0x075C]) + 1
        env.close()
        bad = sorted(s for s in set(cf.SEL) | set(cf.HELD) | set(int(v) for v in cf.POOL) if dead_at is not None and s >= dead_at)
        out[L] = dict(idle_frames_checked=need, game_ends_at=dead_at, bad_starts=bad, x_at_reset=x_start,
                      x_after_idle=x0, ram_world_stage=f"{world}-{stage}")
        print(L, out[L], flush=True)
    return out


def ep(task):
    L = cf.check_level(task["level"])
    pl = cf.mem._player(task["policy"], "greedy")
    offs = []
    r = cf.mem.play_attempt(pl, L, task["start"], cb=lambda g, o, a: offs.append(g.off))
    return dict(start=r["start"], max_x=r["max_x"], flag=r["flag"], end=r["end"], dec=r["decisions"],
                area_jumps=int(len(set(offs)) - 1), max_off=int(max(offs) if offs else 0))


def main():
    cf.idle(); OUT.mkdir(parents=True, exist_ok=True)
    sc = startcheck(); (OUT / "startcheck.json").write_text(json.dumps(sc, indent=1))
    nw = cf.workers(); print("workers", nw, flush=True)
    ex = ProcessPoolExecutor(nw, mp_context=get_context("spawn"), initializer=cf.worker_init)
    res = {}
    for name, path in cf.CKPTS.items():
        t = time.time()
        tasks = [dict(policy=str(path), level=L, start=s) for L in cf.DEV for s in cf.HELD]
        rr = list(ex.map(ep, tasks, chunksize=1))
        res[name] = {}
        for i, L in enumerate(cf.DEV):
            part = rr[i * len(cf.HELD):(i + 1) * len(cf.HELD)]
            xs = [r["max_x"] for r in part]
            res[name][L] = dict(mean_x=round(float(np.mean(xs)), 1), median_x=float(np.median(xs)), max_x=max(xs),
                                flags=sum(r["flag"] for r in part), ends={e: sum(r["end"] == e for r in part) for e in ("died", "stuck", "flag", "cap")},
                                area_jumps=sum(r["area_jumps"] > 0 for r in part), runs=part)
            print(name, L, {k: v for k, v in res[name][L].items() if k != "runs"}, flush=True)
        print(name, f"{time.time() - t:.0f}s", flush=True)
        (OUT / "firstsight.json").write_text(json.dumps(res, indent=1))
    lines = ["| level | " + " | ".join(f"{n} mean x / flags" for n in res) + " |", "|---|" + "---|" * len(res)]
    for L in cf.DEV:
        lines.append(f"| {L} | " + " | ".join(f"{res[n][L]['mean_x']} / {res[n][L]['flags']}" for n in res) + " |")
    (OUT / "firstsight.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    ex.shutdown()


if __name__ == "__main__":
    main()
