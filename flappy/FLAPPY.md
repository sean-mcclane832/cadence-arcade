# Corvus: a Cadence brain that imitates a Flappy Bird teacher

A one-hour side build (2026-11-06). Corvus is a Cadence 0.94.0 brain that plays a Flappy Bird clone by
**imitating a scripted teacher** (DAgger: watch the teacher, then fly on its own while the teacher labels the states
it reaches, as in Cadence `docs/teach.md`). It does not learn from reward, and it does not see pixels: its senses are
four numbers about the next pipes. So this shows a Cadence brain copying a good controller well enough to play, not a
brain discovering Flappy Bird.

Named after Marcus Valerius Corvus, the Roman named for a raven.

## Setup
- Game: `game.py`, 288x512, gravity 0.45/frame, a flap sets the speed to -7, pipes 52 px wide with a 140 px gap,
  180 px apart, 3 px/frame. A new gap centre is at most 100 px from the previous one. Score = pipes passed.
- Senses (48 inputs): bird height vs the next gap's centre, vertical speed, distance to the next pipe, and the
  next-but-one gap's shift, each coded as 12 Gaussian bumps.
- Brain: `Brain.compose(inputs=48, actions=2, seed=0, teaching="normalized", learning_surprise=True)`,
  the default association region (64 neurons). One decision per frame: glide or flap.
- Teacher: flap when the bird is more than 35 px below the gap centre and not rising. Tuned on dev seed 7:
  200/200 pipes in all 40 games. (A first version reached a median of 13 because gap centres could jump 250 px;
  the 100 px limit matches the original game.)
- Training (`train.py`): round 0 watches the teacher on 32 games for 1500 frames (`lesson` only). Each later round,
  the brain acts greedily on 32 games and the teacher's label for the reached state is taught (`act(greedy=True)` then
  `lesson`). After each round, 20 dev games (seed 1000, cap 50, no learning) decide whether to stop.

| Round | Dev median (cap 50) | Dev mean | Dev min | Minutes |
|---|---|---|---|---|
| 0 (watched only) | 0 | 0.0 | 0 | 0.3 |
| 1 (own path) | 50 | 45.4 | 2 | 1.2 |

After one own-path round the dev median hit the 50 cap (mean 45.4, one game still ended at 2), so training stopped
there, 1.2 minutes after it started (the minutes column is time since the start).

## Test (registered in the plan before training)
20 fresh games (seed 5000, never used before), cap 100 pipes, brain greedy, no learning. "Plays" = median at least 20.

| Player | Median | Mean | Min | Max |
|---|---|---|---|---|
| Corvus | **70** | 68.5 | 2 | 100 |
| Teacher | 100 | 100 | 100 | 100 |
| Random (flaps 8% of frames) | 0 | 0.05 | 0 | 1 |

Passed (median 70 >= 20). Still below its teacher: some games end early (min 2). Raw numbers: `test_result.json`,
`train_log.json`.

Watching only gave a brain that scored 0. One round on its own path took its dev median to the cap. That matches the DAgger
point in Cadence's docs: a pupil that only watches never sees the states its own mistakes lead to.

## Run it
- `watch-flappy.bat` (or `python watch.py`): Corvus plays live, the game on the left and the brain on the right
  (the senses, the 64 association neurons, the two motor neurons, the flap vote, pipes per game).
  Keys: space pause, f fast, q quit.
- `python train.py`: retrain live in the same window. `python test.py`: the registered test.
- Venv: `uv venv .venv --python 3.11` then `uv pip install cadence-net==0.94.0 pygame numpy`.

## Next, if wanted
Pixels instead of the four numbers, reward learning on its own after imitation, and a teacher-free run.

## Learning without the teacher (2026-11-06, quick attempts, both failed)
Both start from the tested Corvus (dev median 81 on 20 dev games, seed 1000, cap 150, greedy).

- **Reward only** (`watch.py --reward`): `Brain.step` with +1 per pipe, -1 per death, +0.01 per frame, actor_eta 0.1.
  Corvus's two motor neurons are nearly tied, so sampled moves flip easily: at the composed temperature it scored 0,
  at 0.001 around 10. Over 43 games it drifted down from about 10 pipes to about 2.
- **Self-teaching** (`selfteach.py`, the lesson protocol with its own outcomes as labels): 5% of greedy moves flipped
  to explore; the moves since the last pipe passed become labels. Dev median 81 -> 1.5 after one round, 0 by round 3.
  A first version that also taught "the opposite of the last 6 moves" before a death went 81 -> 0 in one round.

Why the self-teaching broke: the brain uses `learning_surprise=True`, so only labels that differ from its current
answer teach anything. Its own greedy moves agree, so the only labels that taught were the random 5% flips: it was
being taught noise. A pipe passed 50 frames later says nothing about whether a flip helped. Same shape as the Mario
coach-free pilots (B1 self-imitation not distinguishable, B3 `Brain.live` forgot worlds 1-3).
