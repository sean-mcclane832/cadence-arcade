"""Single-frame Breakout: does Cadence 0.89.0's recall position term help a brain play?

Run from this folder with the classic venv:

    ../.venv-classic/Scripts/python.exe run.py --seeds 10-21 --out runs/main

PRE-DECLARED (written before the main run; settings were picked on pilot seeds 0-2, which are
not reported)

Game. MinAtar Breakout, unchanged rules, sticky actions 0.1. The player sees one frame of 49
hand features (ball column, ball row, paddle column, ball-minus-paddle) and NOT MinAtar's trail,
so which way the ball is moving is only knowable from a moment ago.

Teaching. A coach that knows the ball's direction (it plays a copy of the game forward) labels
every training moment. All learners get the same kind of labels on their own trajectories
(DAgger style): Cadence via `Brain.lesson(obs, label)` then `Brain.act(obs, greedy=True)`, the
look-back benchmark's recipe; the MLPs via one online cross-entropy step. The position term is
trained only by lessons (rewards never move it), which is why this demo teaches instead of
paying: under reward-only `live` the maps stay zero and recall_position=32 reads exactly as
recall_position=0 (checked in `identity.py`).

Life. 30 blocks. Each block: 1,000 training moments (episodes end on a lost ball or at 300
moments; `reset()` between episodes empties the recall slots, learned maps stay), then a test
of 3 episodes (max 300 moments each) on a test game seeded by (seed, block), greedy, no lessons.
The first test episode of blocks 0, 9 and 29 is recorded for the replay page.

Arms (same seeds, same test games):
  pos32     Cadence, recall on, position term on (the 0.89.0 founder, recall_position=32)
  pos0      Cadence, recall on, recall_position=0 (0.88-style memory)
  norecall  Cadence, no recall memory
  mlp1      plain MLP, 1 hidden layer of 32, single frame
  mlp2      the same MLP given two frames (frame + MinAtar's trail): a reference for what
            knowing "a moment ago" is worth, not a fair competitor
  random

Cadence settings for all Cadence arms: modules=(32,), teaching="normalized", episodic=False,
everything else at the 0.89.0 founder (so recall_recency=0.25, recall_temperature=0.25,
recall_capacity=64). MLP lr 0.003 (best of 0.001/0.003/0.01/0.02 on pilot seeds).

Second condition, "recency off" (`--cadence '{"recall_recency": 0.0}' --arms pos32,pos0
--out runs/recency0`): the founder memory already prefers newer slots (recency 0.25 per moment
of age), which may do the position term's job for free. With recency off, as in upstream's
look-back benchmark, only the position term can learn to favour "a moment ago".

Seeds 10-21 (12 per arm, both conditions).

Pilot (seeds 0-2, 15 blocks, not reported): founder memory pos32 vs pos0 primary 5.5/5.8/3.6
vs 5.6/4.8/4.2, no clear gap; recency off 5.5/5.6/3.8 vs 2.4/2.3/2.7.

Metrics. Primary: bricks per test episode over the last 10 blocks (mean of the 30 test episodes
per seed). Secondary: test agreement with the coach, moments survived per test episode.

Predictions.
  P1  pos32 > pos0 on the primary metric, in at least 8 of 12 paired seeds, with a median paired
      difference above 0. If not, the position term did not help here, and that is the result.
  P2  pos0 >= norecall (a memory of the last moments is usable at all).
  P3  mlp2 > mlp1 by a wide margin (the trail is worth a lot: a lookup table with two frames
      scored ~3.7x one with one frame in a pilot).
  P4  every learner beats random.
  P5  recency off: pos32 > pos0 in at least 8 of 12 paired seeds (the term is the only way to
      look a moment back there).
  Expectation from the pilot: P1 null (no clear gain over the founder memory's recency), P5
  holds. Paired comparisons use the Wilcoxon signed-rank test on the 12 seed differences.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from breakout import Breakout

BLOCKS, TRAIN, TEST_EPS, CAP = 30, 1000, 3, 300
LAST = 10  # blocks in the primary metric
MLP_LR = 0.003
CADENCE = dict(modules=(32,), teaching="normalized", episodic=False)
ARMS = ("pos32", "pos0", "norecall", "mlp1", "mlp2", "random")


def make(arm: str, seed: int, cadence: dict, mlp_lr: float = 0.05):
    from players import CadenceStudent, MLPStudent, RandomPlayer
    if arm == "pos32":
        return CadenceStudent(seed, True, 32, **cadence)
    if arm == "pos0":
        return CadenceStudent(seed, True, 0, **cadence)
    if arm == "norecall":
        return CadenceStudent(seed, False, 0, **cadence)
    if arm == "mlp1":
        return MLPStudent(seed, two_frame=False, lr=mlp_lr)
    if arm == "mlp2":
        return MLPStudent(seed, two_frame=True, lr=mlp_lr)
    return RandomPlayer(seed)


def test(player, seed: int, block: int, record: bool):
    env = Breakout(seed=int(np.random.SeedSequence([seed, block, 7]).generate_state(1)[0]))
    scores, lengths, agree, n, clip = [], [], 0, 0, []
    for ep in range(TEST_EPS):
        env.reset(); player.new_episode()
        s = k = 0
        while True:
            obs, prev = env.observe(), env.previous()
            label = env.coach()
            a = player.play(obs, prev)
            if record and ep == 0:
                clip.append(env.frame() + [a, label])
            agree += a == label; n += 1
            r, done = env.step(a); s += r; k += 1
            if done or k >= CAP:
                if record and ep == 0:
                    clip.append(env.frame() + [-1, -1, int(done)])
                break
        scores.append(s); lengths.append(k)
    return scores, lengths, agree / n, clip


def life(job):
    arm, seed, cadence, blocks, mlp_lr = job
    player = make(arm, seed, cadence, mlp_lr)
    env = Breakout(seed=seed)
    t0 = time.time()
    out = {"arm": arm, "seed": seed, "score": [], "length": [], "agree": [], "train_agree": [],
           "train_eps": [], "position": [], "clips": {}}
    k = 0
    player.new_episode()
    clip_blocks = (0, blocks // 3 - 1, blocks - 1)
    for block in range(blocks):
        hits = eps = 0
        for _ in range(TRAIN):
            obs, prev = env.observe(), env.previous()
            label = env.coach()
            a = player.train(obs, prev, label)
            hits += a == label
            _, done = env.step(a); k += 1
            if done or k >= CAP:
                env.reset(); player.new_episode(); k = 0; eps += 1
        env.reset(); player.new_episode(); k = 0
        scores, lengths, agree, clip = test(player, seed, block, block in clip_blocks)
        player.new_episode()
        out["score"].append(scores); out["length"].append(lengths)
        out["agree"].append(round(agree, 4)); out["train_agree"].append(round(hits / TRAIN, 4))
        out["train_eps"].append(eps); out["position"].append(round(player.position_size(), 3))
        if clip:
            out["clips"][str(block)] = clip
    out["seconds"] = round(time.time() - t0, 1)
    out["primary"] = float(np.mean(out["score"][-LAST:]))
    out["warned"] = getattr(player, "warned", {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="10-21")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--mlp-lr", type=float, default=MLP_LR)
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--cadence", default=None, help="JSON overriding the Cadence settings")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    cadence = dict(CADENCE)
    if a.cadence:
        cadence.update(json.loads(a.cadence))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import cadence as cd
    meta = {"cadence": cd.__version__, "python": platform.python_version(), "blocks": a.blocks,
            "train": TRAIN, "test_eps": TEST_EPS, "cap": CAP, "last": LAST,
            "cadence_settings": cadence, "seeds": [lo, hi], "arms": a.arms.split(","),
            "mlp": f"1 hidden layer of 32, softmax, online cross-entropy, lr {a.mlp_lr}"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=list) + "\n")
    # slow arms first so the pool stays busy
    jobs = [(arm, s, cadence, a.blocks, a.mlp_lr) for arm in a.arms.split(",") for s in range(lo, hi + 1)]
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(life, jobs):
            (out / f"{r['arm']}-{r['seed']}.json").write_text(json.dumps(r, separators=(",", ":")) + "\n")
            print(r["arm"], r["seed"], "primary", round(r["primary"], 2), "agree", r["agree"][-1],
                  "pos", r["position"][-1], f"{r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
