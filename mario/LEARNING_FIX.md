# Why the Cadence brain doesn't learn Mario, and a fix (2026-10-29 night)

Lab notebook. Every experiment is written down BEFORE it runs (hypothesis, prediction, what would
falsify it), then the result, dead ends included. Cadence 0.91.1, one brain per configuration.

## Facts going in (from PROGRESS.md and the run logs)
- Overnight (runs/main): pure coach labels + DAgger. Training agreement (sampled during
  lessons) ~0.62-0.73, while the "run" share of labels is 0.58. Solo peak median 1,417, final ~594.
- In-order (runs/inorder_blind): training agreement on the level's own rows only **0.35-0.42 on 1-1,
  0.23-0.31 on 1-2**. Its data file for 1-1 holds 28,500 rows, of which **25,128 (88%) are
  self-imitation rows** (the brain's own *sampled* actions during reward self-play, weight 0.5),
  2,899 coach rows of weight 1 and 473 critical coach rows of weight 10. By lesson mass the brain
  spends ~63% of its lessons copying its own sampled (temperature) actions.
- The predict / lesson path (`brain.predict`, `brain.learner.step(brain.stimulus(x, memory=False))`)
  settles cold (free phase from rest, `warm=None`), so the chat project's warm-start trap does not
  apply there. Self-play in levels.py uses `brain.step` (warm stream, working trace, reward learner),
  and the actor writes the SAME efficacies the imitation readout uses (ActorCritic wraps the Learner).

## E1. Can a learner fit the coach labels? (pre-registered 19:55)
Data: runs/main/data.npz (19,618 coach-labelled screens, 1,664 inputs, 7 buttons, 3,889 critical).
Split: random 80/20 (rng 0) and "late 20%" (last DAgger rounds held out).
- H1a: the labels are learnable from these inputs. Prediction: backprop MLP held-out >= 0.75,
  1-NN leave-one-out consistency >= 0.8. Falsified if MLP <= 0.65 (near the run share 0.58).
- H1b: Cadence as configured underfits. Prediction: Cadence (normalized + surprise, 256, weighted)
  stays <= 0.68 train after 10k lessons while the MLP reaches >= 0.8 train. Falsified if Cadence
  train >= 0.78.
- Arms: base (the overnight setting), nosurp, e1 (eta 0.001), rel (normalize_relative 0.1 +
  32 fixed nudged steps), founder (unnormalized eta 0.5), x2 (inputs doubled, clipped).

Result (backprop, diag/fit_main.json, GPU, weighted CE, 60 epochs):
| split | majority | 1-NN held-out | 1-NN LOO train | logreg test | MLP256 train | MLP256 test | MLP crit test |
|---|---|---|---|---|---|---|---|
| rand20 | 0.58 | 0.86 | 0.87 | 0.71 | 0.81 | 0.77 | 0.95 |
| late20 | 0.57 | 0.77 | 0.87 | 0.71 | 0.86 | 0.75 | 0.77 |
H1a holds: labels are ~87% consistent at the 1-NN level and an MLP gets 0.75-0.77 held out.
But the overnight MLP baseline that played from these labels was no better than random (596):
fitting 0.75 is not enough to play. Keep that in mind: the fix must also address the interface.


Partial Cadence fit (2,000 lessons of 32, rand20 split): base 0.69 -> 0.53 train (swings between
"run" and "run+jump" answers), nosurp 0.71 / 0.69 held-out, e1 0.69 / 0.66, rel 0.59, x2 0.63,
founder 0.66. All arms stopped at 20:25 when Sean started Minecraft (2-core limit). Last points:
base 0.53 / 0.52 at 2k (it was 0.69 at 1k: swings), nosurp 0.71 / 0.69 at 2k, e1 0.73 / 0.70 at 3k.
Against the MLP's 0.81 train / 0.77 held-out at matched data, Cadence fits less and less steadily,
but every arm is far above the 0.35-0.42 the in-order run reached on its own rows. H1b holds
loosely (Cadence <= 0.73 at <= 3k lessons). Not the main story: see E2.

## E2. How do solo attempts end? (diag/replay_eval.py, diag/alias_pipe.py) -- ROOT CAUSE 1
Replaying the stored actions of the last 60 overnight solo evals (rounds 10-21): **31 of 60 end
"stuck"**: Mario stands on the ground (float state 0) against a pipe at x 434 / 594 / 722 / 898,
pressing the SAME button every decision until the stuck limit: run+jump (holding A, which in SMB
does not start a new jump; a jump needs a fresh press) or plain run (into the pipe). The rest die,
most at the first Goomba (x 306-312).

Then the decisive check, at the stuck point of round 18 (x 434, holding run+jump): the coach says
**run, weight 10 (critical)**: let go of A. One decision later, same place, screen unchanged except
for the blinking ? block (max block change 0.127), A released: the coach says **run+jump**. Press A,
then it says run again. So the coach's label for one screen flips with a button the brain cannot
see. A memoryless policy that reads only the screen (predict, memory=False) must give one answer
there, and whatever it answers, it then repeats forever because the screen does not change.
DAgger makes it worse: the stuck attempt adds ~40 near-identical rows labelled "release" (critical
x10) one round, the next round's stuck attempt adds "press" rows, and the answer at the pipe flips
from round to round. That matches the overnight swings (r18 stuck holding run+jump at 434, r19
stuck pressing run at 594, ...). The MLP baseline played no better than random for the same reason:
any learner that sees only the screen is stuck here. It is the interface, not (only) Cadence.

## E3. The fix, tested on a fast learner first (pre-registered 19:50)
Give the brain what its own body feels: the buttons its thumb held during the last decision (A, B,
right, left) and Mario's proprioception (speeds, on ground / rising / falling, facing, power,
timers; game.proprio, no world RAM). Input 1,664 eyes + 336 body = 2,000. dagger.py, lean menu,
coach labels only (no self-imitation rows), 5 solo starts per round.
- H3: with the body sense a backprop MLP + DAgger clears 1-1 solo within ~1 h; without it (blind)
  it stays stuck at pipes. Prediction: body flags >= 2/5 by round 6; blind 0/5 with stuck endings.
  Falsified if body MLP also ends stuck at pipes, or blind clears as well as body.

Run notes: started 19:54 with 4 workers, stopped 20:25 (Minecraft, 2-core limit) before round 0
finished; restarted 20:30 with 2 workers and cheaper DAgger labelling: the learner plays alone,
then the attempt is replayed (deterministic emulator) and the coach labels every decision of the
last 60 before the end and every 3rd before that. Round 0: 6 coach demonstrations (1 clean + 5
with 12-25% random buttons... see log), saved per episode in parts/.

## E4. Why in-order training learned nothing and "stood still" on 2-1 -- ROOT CAUSE 2 (log reading)
- In-order rounds add ~1,600-1,850 **self-imitation rows** (the brain's own *sampled* actions from
  reward self-play, every decision that was not in the last 15 before a death) against ~110-190
  coach labels: >90% of the new rows each round are the brain copying its own dice rolls.
- The brain's agreement with its own level rows falls to 0.15-0.29 (2-1) and 0.23-0.31 (1-2): it
  is being taught a near-uniform action distribution, which it cannot (and should not) fit.
- 2-1, senses run: r4 solo x [40, 40, 40] (standing still), rollback to the r2 state, then r5
  taught on the same pile -> [40, 40, 40] again; r6 the same. The weights were restored, the
  collapse came back after one teaching pass: the **data pile** causes it, not the reward
  learner's drift. (Where argmax lands on a flat target is arbitrary; on 2-1 it was noop/left.)
- So the in-order design diluted the only consistent teacher (the coach) ~10:1 with noise. Dropped:
  the fix teaches coach labels only. Reward self-play is not used for the policy weights.

### E3 result (MLP control, runs/fix_mlp_body + fix_mlp_body2): H3 holds
Round 0 (11 coach attempts, 2,838 rows): training agreement **0.934** (critical 0.998), against
0.75-0.81 for the same MLP on the blind overnight labels: the body sense removes most of the label
conflict. Solo by round (5 starts, median x): r0 898 (all 5 hop at the tall pipe at 898: no longer
frozen, it now releases and re-presses, but its hops are 2 decisions long), r1 312, **r2 1,674 with
a solo flag (start 0)**, r3 2,018, **r4 2,472 with a solo flag (start 11)**. 31 min of DAgger after
the 31-min round 0, ~5k rows. Stopped at r4 to give the 2 cores to Cadence. The blind control is the
overnight run itself (same coach, same DAgger, no body sense): 0 flags in 21 rounds, 31/60 stuck.

## E5. The same fix on Cadence (pre-registered 21:37)
runs/fix_cad_body: Cadence 0.91.1, one region of 256, teaching="normalized" + learning_surprise,
teaching steps 0.001 / 0.0001 (the steadiest arm of E1), cold `predict` and cold lessons, coach labels
only, body sense, the SAME 2,838 round-0 rows the MLP started from, 3,000 lessons of 32 per round,
2 cores (Minecraft running).
- H5: the body sense lets Cadence fit the coach (training agreement >= 0.85, vs 0.62-0.73 blind) and
  DAgger then takes it to a solo flag on 1-1 within 4 h. Falsified if agreement stays <= 0.75, or
  if it reaches the MLP's agreement but never flags (then the remaining gap is Cadence's policy
  quality, not the interface).
E5 first try (arm e1 = surprise on): training agreement per round 0.85, 0.63, 0.73, 0.75, 0.61
while the MLP's was 0.93-0.96 on the same rows; solo medians 312, 680, 312, 898, 315. Paused at r4
(22:06) to find out why the fit swings. Falsifier branch 1 hit: agreement stays <= 0.75.

## E6. Why Cadence's fit swings (offline, diag/fit_body.py, MLP-run body data, 5,011 rows, rand 80/20) -- ROOT CAUSE 3
Pre-registered 22:07: H6: the swing comes from `learning_surprise` (lessons only on rows answered
wrongly) under normalized steps, not from unsettled free phases. Prediction: e1 with surprise peaks
then falls over 6k lessons; e1 without surprise rises monotonically to >= 0.88 train; the share of
lessons at the free_steps cap is ~0. Falsified if nosurp also falls, or if unsettled > 5%.

| arm (eta, surprise) | 1k | 2k | 3k | 4k | 5k | 6k (train / test / crit test) |
|---|---|---|---|---|---|---|
| 0.001, surprise (the run) | 0.82 | 0.85 | 0.84 | 0.78 | 0.80 | **0.66 / 0.61 / 0.35** |
| 0.0003, surprise | 0.84 | 0.88 | 0.82 | 0.85 | 0.89 | 0.90 / 0.86 / 0.90 |
| 0.001, every row | 0.82 | 0.89 | 0.91 | 0.92 | 0.92 | **0.92 / 0.90 / 0.93** |
Unsettled lessons (free phase at its 1,024-step cap): 0.0-0.2% in every arm. H6 holds. With
surprise on, once most rows are right only the conflicting/hard rows are taught, at full
normalized step size, and the readout drifts until it collapses (6k: critical labels 0.35). Lessons
on every row anchor it. The blind overnight fit (E1) showed the same: nosurp steadiest, base swung
0.69 -> 0.53. (Upstream advises surprise under the normalized point; on this task it is the
unstable choice. One seed per arm; it agrees across two datasets.)
Without surprise Cadence matches the MLP's fit (0.92 / 0.90 vs 0.93-0.96 train).

### E5 rerun without surprise (runs/fix_cad_body2, arm e1_nosurp, from the same 2,838 round-0 rows)
| round | min | rows | training agreement (crit) | solo x (5 starts) | median |
|---|---|---|---|---|---|
| 0 | 3 | 2,838 | 0.87 (0.99) | 2930, 1127, 305, 676, 1127 | 1,127 |
| 1 | 9 | 3,231 | 0.89 (0.99) | 1129, 1129, 1676, 1676, 2178 | 1,676 |
| 2 | 19 | 3,889 | 0.91 (0.98) | 315, 312, 312, 1788, 436 | 315 |
| 3 | 27 | 4,327 | 0.89 (0.98) | 1791, 1788, 1788, 1413, 1413 | 1,788 |
| 4 | 36 | 4,946 | 0.93 (0.99) | 1778, 1781, 1781, 2046, 1960 | 1,781 |
| 5 | 46 | 5,599 | 0.91 (0.98) | 2025, 2022, 898, 898, 898 | 898 |
The fit now holds at 0.87-0.93 (the MLP's level) every round. Round 0 already went further than
any blind brain ever did solo (2,930; the overnight best was 1,783 after 2.5 h). In round 6's DAgger
the round-5 brain, playing alone from the very start (no coach prefix; random start jitter, not recorded), **reached
the flag**: the first Cadence solo clear of 1-1 (not one of the 5 eval starts; logged in
runs/fix_cad_body2/log.txt 23:16). Continued as the same brain in runs/fix_cad_body3 (HOI4 started
23:18: 2 cores).

### E5 continued (runs/fix_cad_body3: the same brain, from its round-5 state, all its rows)
| round | min | rows | agree | solo x (starts 0/5/11/17/23) | flags |
|---|---|---|---|---|---|
| 0 | 3 | 6,254 | 0.91 | **3161**, 1961, 1961, 2016, 2016 | 1/5 |
| 1 | 11 | 6,770 | 0.92 | 2018, 2016, 2016, 2016, 2016 | 0/5 |
| 2 | 23 | 7,504 | 0.92 | **3161, 3161, 3161**, 1779, 1779 | **3/5** |
| 3 | 32 | 8,057 | 0.91 | 2763, 2471, **3161**, 2751, **3161** | 2/5 |
| 4 | 47 | 8,961 | 0.90 | 2474, **3161**, 2471, 2471, 2471 | 1/5 |
| 5 | 59 | 9,664 | - | **3161**, 2020, 2020, 2023, **3161** | 2/5 |
| 6 | 72 | 10,383 | 0.92 | **3161**, 2018, 2018, 2018, 2018 | 1/5 |
Round 2 clears starts 0, 5 and 11, i.e. **flag in 3 of 3 on the original 3-start rule**. Rechecked
on ten start jitters it never saw in an eval (1-4, 6-10, 12; runs/fix_eval_r02): **8 of 10 flags**
(the other two stuck at x 2,226). Not luck. The round-6 brain on the same ten: 4/10. More rounds
did not help after r2: the late deaths (diag/death_check.py) are single-decision timing slips in
critical moments where the coach says w10: letting go of A one decision early at a jump's apex
(x 1,951, dies at 2,020) and a jump pressed one decision early at the end staircase (dies at 2,471).
Total for the r2 brain: 31 min of coach demonstrations + ~70 min of DAgger, mostly on 2 cores.
Kept: runs/fix_cad_body3/policy_r02.npz = the "1-1 brain".

## E7. First sight with the 1-1 brain (starts 0/6/12, the baselines' starts; runs/fs_*)
| level | 1-1 brain (body) | untrained body brain | always run | run+jump | random (10 seeds) | fresh blind | old in-order blind |
|---|---|---|---|---|---|---|---|
| 1-2 | 260 | 60 | 196 | 196 | 338 | 40 | 183 |
| 1-3 | **402** | 289 | 321 | 317 | 300 | 705 | 311 |
| 1-4 | 229 | 40 | 234 | 229 | 331 | 551 | 234 |
| 2-1 | 339 | 40 | 308 | 306 | 384 | 120 | 322 |
It beats always-run, run+jump and the untrained body brain on 1-2, 1-3 and 2-1 and the old blind
in-order brain on every level, but random buttons beat it on 1-2, 1-4 and 2-1, and the fresh
blind brain's lucky 705 / 551 on 1-3 / 1-4 stand. Skill from one level does not transfer to unseen
levels at first sight: what it learned is 1-1's timing, not a general "jump over things".

## E8. Retention: practise 1-2, keep 1-1 (pre-registered 00:35)
runs/fix_cad_1-2: the 1-1 brain practises 1-2 (coach round 0 from runs/fix_coach_1-2, 2,519 rows,
then DAgger), each 32-row lesson = 16 rows of 1-2 + 16 replayed 1-1 coach rows (10,383), 1-1 solo
tested every round (starts 0/5/11/17/23).
- H8: with half of every lesson replaying old coach labels, 1-1 keeps >= 2/5 flags (the in-order
  run fell from 898 to ~300 with 10% replay of a noisy pile). Falsified if 1-1 drops below median
  2,000 or 0 flags for 2 rounds in a row.
