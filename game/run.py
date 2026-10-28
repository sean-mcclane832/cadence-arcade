"""Live one player through a whole Ember Field life and record it.

Each life is 45,000 moments in three seasons: amber feeds until 15,000, blue feeds until
30,000, amber again after. Per 500-moment window it records food caught, stings taken and
(for Cadence) the share of moments the brain was aroused. It also records six 400-moment
clips (frames, actions, pay, what the brain saw) for the replay page.

    python -m game.run --seeds 10-21 --out runs/main
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from game.field import EmberField, inputs

LIFE, TURNS, SPAWN, VIEW, WINDOW, CLIP = 45_000, (15_000, 30_000), 0.3, 3, 500, 400
CLIPS = {"newborn": 0, "season 1, late": 14_600, "after the 1st turn": 15_000,
         "season 2, late": 29_600, "after the 2nd turn": 30_000, "end of life": 44_600}
CADENCE = dict(modules=(32,), memory_choice=True, arousal_need_scale=2.0)


def make(kind: str, seed: int):
    from game.agents import CadencePlayer, MLPPlayer, RandomPlayer
    n = inputs(VIEW)
    if kind == "cadence":
        return CadencePlayer(n, seed, **CADENCE)
    if kind == "mlp":
        return MLPPlayer(n, seed)
    return RandomPlayer(n, seed)


def live(job):
    kind, seed = job
    env = EmberField(seed=seed, spawn=SPAWN, view=VIEW, turns=TURNS)
    player = make(kind, seed)
    starts = {v: k for k, v in CLIPS.items()}
    clips, current = {}, None
    food, sting, aroused = [], [], []
    wf = ws = wa = 0
    t0 = time.time()
    obs = env.observe()
    action = player.first(obs)
    for t in range(LIFE):
        if t in starts:
            current = clips.setdefault(starts[t], {"start": t, "frames": []})
        frame = env.frame()
        seen = "".join(str(int(round(v * 3))) for v in obs)
        reward, _ = env.step(action)
        mode = 0
        if kind == "cadence":
            mode = 1 if player.brain.last_arousal.get("mode") == "aroused" else 0
        if current is not None:
            current["frames"].append([frame[0], frame[1], action, reward, seen, mode, env.food()])
            if len(current["frames"]) >= CLIP:
                current = None
        wf += reward > 0
        ws += reward < 0
        wa += mode
        if (t + 1) % WINDOW == 0:
            food.append(int(wf)); sting.append(int(ws)); aroused.append(round(wa / WINDOW, 3))
            wf = ws = wa = 0
        obs = env.observe()
        action = player.act(obs, reward)
    return {"kind": kind, "seed": seed, "food": food, "sting": sting, "aroused": aroused,
            "clips": clips, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="10-21")
    ap.add_argument("--kinds", default="cadence,mlp,random")
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    jobs = [(k, s) for k in a.kinds.split(",") for s in range(lo, hi + 1)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import cadence
    meta = {"cadence": cadence.__version__, "python": platform.python_version(), "life": LIFE,
            "turns": TURNS, "spawn": SPAWN, "view": VIEW, "window": WINDOW, "cadence_settings": CADENCE,
            "mlp": "1-hidden-32 online Q-learning, lr .01, gamma .8, eps .1", "seeds": [lo, hi]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=list) + "\n")
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(live, jobs):
            (out / f"{r['kind']}-{r['seed']}.json").write_text(json.dumps(r, separators=(",", ":")) + "\n")
            print(r["kind"], r["seed"], "seasons net:",
                  [sum(r["food"][i:i + 30]) - sum(r["sting"][i:i + 30]) for i in (0, 30, 60)],
                  f"{r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
