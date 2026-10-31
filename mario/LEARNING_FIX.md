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

### E8 result: H8 falsified as stated, but read E9
1-2 practice (runs/fix_cad_1-2, 8 rounds, 66 min): 1-2 solo medians 758, 883, 758, 1189, 879, 874,
747, 878 (best single 1,742; the old blind in-order best on 1-2 was 482; the coach itself flags only
6 of 11 round-0 attempts). 1-1 solo during it: flags 3, 1, 0, 1, 3, 1, 0, 0; medians 3161, 2020,
1435, 1504, 3161, 2021, 2033, 1786. Two rounds at 0 flags: falsified. Still far from the old
collapse (898 -> ~300).

## E9. Forgetting or fragility? (diag/consolidate.py, pre-registered 01:45)
H9: the 1-1 losses are not forgotten labels. Prediction: on the 1-2 round-7 brain, agreement with
the 1-1 coach labels is still ~0.9 (critical ~0.98) while 1-1 solo has 0 flags; and teaching on both
levels (50/50) moves solo results a lot while label agreement barely moves. Falsified if 1-1 label
agreement has dropped below 0.85.
| lessons (50/50 both levels, eta 0.001) | 1-1 labels (crit) | 1-2 labels (crit) | 1-1 solo x | 1-2 solo x |
|---|---|---|---|---|
| 0 (1-2 r7 brain) | 0.907 (0.976) | 0.906 (0.973) | 1784, 1786, 1786, 1129, 1416 | 1742, 185, 878, 871, 983 |
| 2,000 | 0.922 (0.989) | 0.895 (0.985) | 2473, 2020, 2020, **3161**, 2471 | 851, 884, **3161, 3161**, 171 |
| 4,000 | 0.917 (0.976) | 0.904 (0.973) | 898, 1792, 898, 898, 898 | 1186, 860, 851, 1646, 171 |
| 6,000 | 0.918 (0.989) | 0.906 (0.968) | 303, 306, 2226, 2020, 2020 | 874, 186, 191, 183, 172 |
H9 holds. **ROOT CAUSE 4: fragility, not forgetting.** Label agreement on 1-1 never left 0.91-0.92
(critical 0.98-0.99) through 1-2 practice and joint teaching, while 1-1 solo swung between a flag
and x 303. A level is a chain of ~30-60 critical decisions; at ~1-2% critical error each, a few
weight steps that move which few rows are wrong decide whether the chain breaks. The 2,000-lesson
point was the first Cadence solo clear of 1-2 (2 of 5 starts), and 4,000 lessons later it was gone.
(Those checkpoints were not saved; the next run saves every one.)

## E10. Held-out starts, and attacking fragility (2026-10-30, pre-registered 12:46, as arm A started)
New driver steady.py: teach in chunks of 250 lessons, save EVERY checkpoint, solo-test after every
chunk on SEL starts (idle frames 0,3,...,27) and keep the best per level (SEL flags, then mean x);
every new best is also played on TEST starts 32-41, which no training, DAgger or selection uses.
"Consistent" = TEST flags >= 8/10. DAgger jitters now come from 0-79 minus 32-41.

Correction first (measured 12:45, runs/st_eval0): the r02 "1-1 brain" scores SEL 6/10 but **TEST
1/10** (mean x 1,824; nine stop at ~1,676). The 8/10 on starts 1-12 (E6) does not carry to a few
more idle frames. r03: TEST 2/10, r06: 4/10, consol2 ck_00500: 2/10. 1-2: 0 flags anywhere.
So 1-1 is not yet consistent, and the start jitter (enemy phase) is a real part of the problem.

- H10a (selection + every-chunk checkpoints): with DAgger every 4 chunks on wider jitter, some
  checkpoint reaches SEL >= 8/10 within 24 chunks. Falsified if the best SEL stays <= 6/10.
- H10b (selection overfits): that best checkpoint's TEST is >= 2 flags below its SEL. Falsified if
  TEST >= SEL - 1.
- H10c (critical rows x3 + eta 0.001 -> 0.0001): fewer swings: the SD of SEL mean x over the last
  12 chunks is lower than the plain arm's, and best TEST flags >= plain's. Falsified if both worse.
- H10d (weight averaging, mean of the last 5 checkpoints): averaged brains swing less and reach
  best TEST >= the single checkpoints' best. Falsified if the average is worse than single in both arms.

### E10 first result (13:05): a consistent 1-1 brain, from checkpoint selection
Arm B (crit x3, eta 0.001 -> 0.0001), chunk 1 (250 lessons on the r02 brain's 10,383 rows, before
any new DAgger): **SEL 10/10, TEST 10/10**. Replayed fresh (runs/st_eval1, deterministic): SEL 10/10,
TEST 10/10, starts 80-99 **20/20**, but late starts 100-139 only **16/40** (mean x 2,687). Arm A's
chunk 1 (same plan, crit x1) is also SEL 10/10 but TEST 1/10 and 80-99 7/20. Saved as
brains/1-1_steady.npz. It meets the pre-registered gate (8/10 TEST); it is not robust to every
jitter. Both arms swing again right after (B chunk 2: SEL 2/10), so fragility is still there:
the gain came from saving and testing every 250 lessons, not from a steadier brain.
H10a holds (SEL 10/10 in one chunk). H10b: mixed (A: TEST 9 below SEL; B: TEST = SEL).
H10c/H10d results (13:40; A 7 chunks, B 15, killed to free cores for 1-2): SEL flags per chunk,
B single [10,2,4,7,8,5,7,7,9,9,8,6,4,10,5], B mean-of-last-5 [1,0,9,8,8,7,9,10,9,5,5,5]; SD of
SEL mean x: B single 227, B avg 214, A single 220, A avg 228. Over B's chunks 4-15, single
averages 7.1 flags and the weight average 7.0. **H10d falsified** (averaging weights does not
steady it). **H10c falsified** for swings: crit x3 with eta falling to ~0.0005 swings as much as plain.
What works is checkpoint selection, not a steadier learner.

- H10e (pre-registered 13:42): a majority VOTE of 5 consecutive checkpoints (ties go to the newest)
  beats single checkpoints: on B's windows, mean SEL+TEST flags >= the singles' mean + 2 (of 20),
  and fewer windows below 10/20. Falsified if the vote's mean is within 1 of the singles' mean.
H10e result (runs/st_vote, 13:58): SEL+TEST flags of 20. Votes of 5: 4-8 **7**, 6-10 **18**, 8-12 **19**,
10-14 **14**, 11-15 **6** (mean 12.8). Single checkpoints 4-15: 11, 18, 11, 10, 15, 16, 19, 18, 9, 8, 19, 9
(mean 13.6). **Falsified**: voting is no better than one checkpoint. The vote of a bad stretch is bad.
Side finding: after one more DAgger round on wider jitter, 5 of B's 12 single checkpoints pass TEST
>= 8/10 on 1-1 (ck 5, 8, 10, 11, 14). Consistency on 1-1 is now common, but still not every checkpoint.

## E11. Why 1-2 is slower (pre-registered 14:30)
Facts: the coach alone clears 1-2 from TEST starts 32/35/38/41 (4/4, ~300 decisions; runs/coach12.jsonl),
so the teacher is not the limit. But Cadence's fit to the 1-2 labels falls from 0.91 to 0.80-0.84 as
DAgger adds rows (st_12b, st_12c), while its 1-1 fit stays 0.91.
- H11: the 1-2 labels are less consistent for these senses (aliasing, as in ROOT CAUSE 1), not a
  Cadence capacity limit. Prediction: 1-NN leave-one-out and MLP train fit on runs/d12 (1-2) are
  >= 0.08 below the same on 1-1 (runs/fix_cad_body3). Falsified if the MLP fits 1-2 within 0.03 of 1-1
  (then it is Cadence: capacity or the shared region).
E11 result (14:45, diag/fit_torch.py, rand 80/20): 1-NN LOO 1-1 0.905 / 1-2 0.865; MLP256 60 epochs
train **0.950 / 0.950**, held-out 0.895 / 0.864. **H11 falsified**: the MLP fits 1-2 as well as 1-1.
The 1-2 labels are learnable from these senses; Cadence's 0.80-0.84 on them is a Cadence limit.

## E12. Cadence capacity on 1-2 (offline fits, diag/fit12.py, pre-registered 14:46)
- H12: one region of 256 shared with 1-1 is too small. Prediction after 8k lessons: a fresh 256 brain
  on 1-2 only fits 1-2 >= 0.88 train; a fresh 512 brain on both levels fits both >= 0.88. Falsified
  if fresh-256 on 1-2 alone also stays <= 0.85 (then it's not sharing, it's 1-2 itself or the teaching).
E12 result (14:54, 8k lessons, train/held-out on 1-2): fresh 256 on 1-2 only **0.856 / 0.833**; fresh 512
on both 0.79 at 5k (slow, still rising); the 1-2-practised r07 brain on 1-2 only 0.887 / 0.846 (its 1-1
fit falls 0.91 -> 0.85). **H12 falsified**: a dedicated fresh 256 brain fits 1-2 no better than the shared
one. Cadence tops out ~0.86-0.89 on 1-2 labels where the MLP reaches 0.95.
How 1-2 solos end (diag/fails.py, st_12c best, SEL starts): 2 flags, 6 deaths, 2 stuck at x ~850; most
end early (x 825-960) with the brain disagreeing with the coach on 7-10 of its last 12 decisions, mostly
weight-1 rows (noop / left: the coach waits for enemies). It is lost there, not one slip from success.
- H12b (pre-registered 14:55): DAgger focused on that early section (2/3 of attempts from the level
  start, 24 per round, every 2 chunks) on the r07 1-2-only brain, 90% 1-2 lessons, reaches SEL >= 5/10
  on 1-2 within 2 h. Falsified if best SEL <= 2/10 at 2 h.
E12 extra: fresh 512 on both levels after 8k lessons fits 1-1 0.83 / 1-2 0.80: a wider region is slower, not better.

## E13. 1-3 (pre-registered 15:26, run in parallel on 2 cores)
runs/fix_coach_1-3: 11 coach demonstrations, 1,926 rows (381 critical), coach flags 6/11 (5 of them
with 12-25% random buttons). runs/st_13a: brains/1-1_steady practises 1-3 (70% of lessons, 30% 1-1
replay), DAgger every 2 chunks, 12 attempts, half from the start.
- H13: 1-3 gets a first solo flag on SEL within 2 h while 1-1 keeps TEST >= 8/10 at its best
  joint checkpoint. Falsified if 0 SEL flags on 1-3 at 2 h.
H12b stopped at 1 h (15:47) to free cores for H14: best 1-2 SEL 2/10, **TEST 2/10** (ck_0009, the first
held-out 1-2 flags; that checkpoint also holds 1-1 TEST 9/10). Solo SEL still swings 0-2 flags per chunk.

## E14. A ratchet (pre-registered 15:50)
Keep a 250-lesson chunk only if the practised level's SEL (flags, then mean x) is not worse than the
kept line's; otherwise reload the last kept checkpoint (DAgger rows still accumulate, so the next
chunk differs). Solo play becomes an accept/reject signal on top of the coach's labels.
runs/st_12r: from st_12d ck_0009, its 19,581 1-2 rows (runs/d12b), 90% 1-2, 1-1 must keep SEL >= min(6, the kept line's 1-1 flags) (first launch had a fixed floor of 6 that rejected every chunk; restarted 15:59).
- H14: the ratchet climbs: SEL >= 6/10 and TEST >= 4/10 on 1-2 within 2 h. Falsified if TEST stays
  <= 2/10 (then it only overfits the 10 SEL starts) or SEL never passes 3/10.
H14 result (16:17, stopped after 8 chunks): every chunk was rejected (8/8 "back to ck_0000"). The line
started from a selected peak (SEL 2/10, mean 1,572) and no single 250-lesson step beat it; the best
tries reached 1/10. **Falsified** as run: a strict ratchet from a peak just freezes.

## E15. Under-teaching? (pre-registered 16:17)
Exposure check: the MLP needed ~60 passes over the 1-2 rows to fit 0.95; at 20 passes it fit 0.86.
Cadence fresh-256 at 6k lessons (~21 passes of 9k rows) fit 0.86: the same curve. The DAgger runs
teach only ~750 lessons (~1 pass of 20k rows) per round, so their 0.82-0.86 fit may be simple
under-teaching, not a Cadence ceiling.
runs/st_12L: st_12d ck_0009, 22,273 1-2 rows (runs/d12c), 90% 1-2 / 10% 1-1, NO DAgger, chunks of
1,000 lessons, 30 chunks (~3 passes per 3 chunks), solo test every chunk.
- H15: 1-2 label fit climbs to >= 0.92 train, and with it solo 1-2 rises to SEL >= 5/10 with smaller
  chunk-to-chunk swings. Falsified if fit plateaus <= 0.88, or fit reaches 0.92 but SEL stays <= 2/10.
- H15b (added 16:27, 8k lessons into st_12L: 1-2 fit flat at 0.84-0.86, SEL 0/10): the plateau is the
  step size. Normalized steps do not shrink near a minimum, so eta 0.001 holds the fit at a noise
  floor. runs/st_12L2: same start and rows, eta 0.0003 -> 0.00005 over 20 chunks of 1,000. Prediction:
  fit >= 0.90 by 10k lessons. Falsified if it stays <= 0.87.
H15 result (16:38, st_12L stopped at 18k lessons): 1-2 fit 0.843, 0.861, 0.858, 0.861, 0.847, 0.855,
..., 0.866, 0.853, 0.869, 0.856, 0.858; SEL 1-2 0/10 in 17 of 18 chunks. **Falsified**: the fit
plateaus at ~0.86 from 4k lessons on while the MLP keeps rising on the same rows. More teaching
alone does not do it. (st_12L2, the low-eta arm, keeps running for H15b.)

## E16. Cost-sensitive DAgger (pre-registered 16:39)
Look at the deaths (diag/endshots.py, runs/end12_s0.png): at x ~850 in 1-2 Mario runs into a Koopa
beside two Goombas under the brick columns. The coach says noop / left there with weight 1, because
another first button also survives its short look-ahead. But the button the BRAIN pressed dies. The
weight says how much a row matters to the coach, not to this brain.
Change (dagger.py, task costw): when the coach labels a DAgger row where the brain pressed a different
button, it also scores the brain's button; if that button's best plan ends > 50 px behind, the row
gets the critical weight. runs/st_12cw: from st_12d ck_0009, rows of runs/d12c, DAgger every 2
chunks, 18 attempts (2/3 from the start), 90% 1-2.
- H16: 1-2 TEST >= 4/10 at some checkpoint within 2 h. Falsified if best TEST <= 2/10.
H15b result (17:00, st_12L2 at 14k lessons, eta now ~0.0001): 1-2 fit 0.847-0.871, best SEL 3/10
(ck_0008) but TEST 0/10. **Falsified**: a smaller step does not lift the plateau either.
E13 paused (17:18, 1 h 52 min, 25 chunks, 10,774 1-3 rows): **H13 holds** (first SEL flag on 1-3 at
chunk 8, 29 min). Best 1-3: SEL 1/10, TEST 1/10 (ck_0021, which also holds 1-1 SEL 8/10, TEST 3/10).
1-3 fit 0.82-0.89. Many DAgger attempts end "stuck" (10 of 12 in one round): 1-3's lifts and gaps
need waiting, and the 40-decision stuck limit may cut those off. Paused to give 1-2 the cores.

## E17. Tie-aware labels (pre-registered 17:18)
The 1-2 labels: MLP held-out fit 0.864 = 1-NN consistency 0.865 = Cadence's train fit ~0.86. The coach
picks "the first plan within 2 px of the best", so where run and run+jump both survive, the label is a
coin flip of tiny differences, and the brain is taught to reproduce it. Change (dagger.py, tiew): on a
DAgger row where the brain pressed another button, if that button's best plan is within 2 px of the
coach's best, the label becomes the brain's own button (a tie, not a mistake). Combined with E16's
cost weights. runs/st_12t: from st_12d ck_0009, rows of runs/d12c.
- H17: new rows stop fighting the brain, fit rises to >= 0.90 on 1-2 and TEST >= 4/10 within 2 h.
  Falsified if fit stays <= 0.87 and TEST <= 2/10.

## E18. A broader held-out score (17:42)
TEST 32-41 turned out to be a narrow band: neighbouring jitters behave alike (results alternate with
odd/even idle frames). st_12cw ck_0011 is 1-2 TEST **6/10** but starts 80-99 **1/20** (1-1: TEST 8/10,
80-99 8/20). So from now on every best is also scored on HELD = TEST + 80-99 + 100-138 (even) = 50
starts, none of them in DAgger (pool 0-31, 42-79) or in selection. SEL is now 17 starts (0-76 step 4,
minus 32-40). Honest bar for "consistent": HELD >= 40/50 (80%), on top of the pre-registered TEST >= 8/10.
- brains/1-1_steady: SEL 17/17, TEST 10/10, **HELD 38/50** (76%). It passes the pre-registered gate
  and falls just short of the broader bar. The text to Sean said "10/10 held-out" with the late-start caveat.
H16 so far: 1-2 TEST 6/10 at ck_0011 meets the letter of the prediction, but 80-99 is 1/20: not real
progress. Verdict at 2 h.

## E19. A fovea (pre-registered 17:52)
E12 extra: a two-module brain (256, 256) fits 1-2 worse (0.80 at 6k lessons vs 0.86 for one region).
On 1-2 the MLP's held-out fit (0.864) equals 1-NN consistency (0.865): with these senses, ~14% of
held-out screens get a label their nearest neighbours disagree with. The coarse eyes are 8 x 8 px
blocks of the whole screen; the Koopa-vs-Goomba timing at x ~850 lives in a few blocks.
New sense (dagger.py senses "fovea"): 96 x 96 px around Mario in 4 x 4 px blocks, contrast-normalized,
present and previous decision (1,152 more inputs, 3,152 total). Data: runs/fov12, 1-2 rows labelled
by the coach on attempts of st_12cw ck_0011 (old senses drive the brain, both senses are stored).
- H19: the fovea makes labels more predictable. Prediction: 1-NN held-out and MLP held-out on
  old+fovea are each >= 0.03 above old. Falsified if both gain < 0.015 (then senses are not the limit).
E19 result (18:17, runs/fov12: 2,781 rows, 6 coach + 24 DAgger attempts, rand 80/20, diag/fit_fov.py):
| senses | 1-NN held-out | MLP train | MLP held-out | MLP crit held-out |
|---|---|---|---|---|
| old (eyes + body) | 0.848 | 0.974 | 0.832 | 0.828 |
| old + fovea | 0.834 | 0.993 | 0.839 | 0.852 |
| body + fovea (no coarse eyes) | 0.848 | 0.974 | 0.827 | 0.875 |
| fovea only | 0.821 | 0.969 | 0.834 | 0.859 |
**H19 falsified**: 1-NN -0.014, MLP +0.007. The ~15% held-out disagreement is not about what the eyes
can see. (Critical rows may gain a little, 0.83 -> 0.85-0.88, but on ~100 rows that is within noise.)
That points at the labels themselves (E17's ties) rather than at the senses. Fovea not adopted.
H16 result (18:25, st_12cw stopped at 19 chunks, 1 h 45 min): best 1-2 TEST 6/10 (ck_0011) but 80-99
1/20; SEL 1-2 0-2/10 every chunk; 1-2 fit fell 0.82 -> 0.81, critical 0.95 -> 0.90 (the new
cost-weighted rows are the hard ones). **Falsified** in substance.
H17 at 1 h (st_12t, 9 chunks, stopped 18:25): fit 0.81-0.83, SEL 0-1/10. The tie-relabelled rows are a
small minority in a 27k-row pile labelled the old way, so this was not a clean test. See E17b.

### 1-2 status at the 3-hour rule (18:25)
No real progress since 15:31 (first TEST 2/10). Best 1-2 checkpoints: st_12d ck_0009 (SEL 2/10, TEST
2/10), st_12cw ck_0011 (TEST 6/10, 80-99 1/20). Why it is stuck, as far as the probes go: Cadence fits
the 1-2 labels to ~0.82-0.86 (E12, E15) while 1-1 sits at ~0.90; the labels themselves are only ~85%
predictable from the screen even for an MLP or 1-NN on held-out screens (E11, E19), and critical rows
too (MLP 0.83 held-out). Not the teacher (coach clears 4/4), not the eyes (E19), not the step size
(E15b), not more teaching (E15), not capacity by width or depth (E12). A level needs dozens of critical
calls in a row; at ~5-9% critical error the chain breaks.

## E17b. A clean tie-aware pile for 1-2 (pre-registered 18:29)
runs/st_12tc: from st_12d ck_0009; 1-2 rows = only the 2,519 coach-demo rows (runs/fix_coach_1-2) plus
new DAgger rows labelled tie-aware (no old DAgger rows); a DAgger round EVERY chunk of 500 lessons.
- H17b: without the old coin-flip labels, 1-2 fit stays >= 0.90 as the pile grows past 10k rows, and
  SEL 1-2 reaches >= 4/17 within 2 h. Falsified if fit falls below 0.86 like the old piles.
Same recipe resumed on 1-3 (runs/st_13b, 2 cores): from st_13a ck_0021 (the 1-3 SEL/TEST 1/10 brain),
1-3 rows = the 1,926 coach-demo rows + tie-aware DAgger only.

## Milestone (19:31): one brain, 1-1 consistent on the strict bar, plus 1-3
runs/st_13b ck_0008 (saved as brains/joint_11_13_ck8.npz), picked by the joint SEL rule (never on
TEST or HELD): **1-1 SEL 12/17, TEST 10/10, HELD 44/50 (88%)**; **1-3 SEL 17/17, TEST 3/10, HELD 16/50**.
So 1-1 now passes both bars, in a brain that also clears 1-3 from every practice start. Held-out 1-3 is
weak: 7 of 10 TEST starts stop at x 1,371. Fit 1-1 0.88, 1-3 0.87. E17b recipe (coach-demo rows +
tie-aware DAgger only), 9 chunks of 250 lessons and 4 DAgger rounds after st_13a.
H17b result (19:44, st_12tc stopped at 6 chunks, 10,354 1-2 rows): 1-2 fit 0.882, 0.865, 0.850, 0.845,
0.856, 0.844 as the clean pile grew; SEL 1-2 at most 1/17. **Falsified** on 1-2: without the old rows
the fit still falls to ~0.85. (The same recipe on 1-3 gave the milestone above, so it is not useless,
but it does not explain 1-2.)

## E20. 1-3 held-out, from the joint brain (pre-registered 19:44)
runs/st_13c: from brains/joint_11_13_ck8, 1-3 rows = st_13b's pile so far (runs/d13), 30% 1-1 replay,
tie-aware DAgger every 2 chunks, 12 attempts, half from the start (starts from the 0-31/42-79 pool).
- H20: 1-3 TEST >= 6/10 at some joint-best checkpoint within 2 h, with 1-1 HELD >= 40/50 at that
  checkpoint. Falsified if the best joint checkpoint's 1-3 TEST stays <= 3/10.
Why tie-aware labels could not fix 1-2 (diag/tierate.py, st_12tc ck_0004 on its own solo paths,
starts 0 and 8, every decision scored by the coach): of 204 decisions the brain agrees on 142; of the
62 disagreements only **8 are ties** (<= 2 px), 31 are "small" (its button survives but ends 2-50 px
behind) and **23 lose** (> 50 px behind). On the coach's critical rows along the brain's own path it is
wrong on **9 of 33 (27%)**, while its fit on the stored critical rows is ~0.96. The gap is
generalization to the states its own play reaches, not label noise from ties.
Note (19:53): st_13b ck_0011 is 1-3 SEL 14/17 but TEST 0/10. TEST 32-41 sits in a hole of the DAgger
start pool (0-31, 42-79), so it tests enemy/lift phases the brain never practised near; HELD's 80-138
are beyond the pool. On 1-3 (moving lifts) that matters more than on 1-1.

## E21. 1-4 (pre-registered 19:53)
runs/fix_coach_1-4: 11 coach demonstrations. runs/st_14a: from brains/joint_11_13_ck8, 1-4 rows = coach
demos + tie-aware DAgger, 30% 1-1 replay, watched levels 1-1 and 1-4.
- H21: first solo flag on 1-4 within 2 h. Falsified if 0 SEL flags at 2 h.

## E22. Selection bug: SEL covered one frame phase (found 20:00)
A decision is 4 frames, so start jitter acts through its phase mod 4. E18's SEL (0, 4, ..., 76) were all
phase 0. The joint brain's 1-3 TEST flags are exactly starts 32, 36, 40 (phase 0); the other phases stop
at x 1,371. So its "1-3 SEL 17/17" means "1-3 from one phase", and the joint pick was biased toward
phase-0 skill. 1-1 is unaffected in substance (the joint brain's 1-1 TEST is 10/10, HELD 44/50, all
phases). Fix: SEL = 0, 3, ..., 78 minus 32-41 (24 starts, all four phases). Runs st_13b/13c/14a stopped
and restarted with it: st_13d (1-3, from the joint brain, rows runs/d13b = 6,821) and st_14b (1-4,
from the joint brain, rows runs/d14). H20 and H21 carry over unchanged to these runs.

## Milestone (20:46): 1-4 on the pre-registered bar
runs/st_14b ck_0010 (brains/joint_11_14_ck10.npz), picked by joint SEL: **1-4 SEL 17/24, TEST 8/10, HELD
38/50** (76%), about 50 min and 5 DAgger rounds after the 1-4 coach demos. It beats Bowser (clip from
TEST start 32). **H21 holds** (first SEL flag at chunk 5, 26 min). But the same checkpoint is 1-1 TEST
2/10, HELD 23/50 and 1-3 0/50: the joint pick traded 1-1 for 1-4. Earlier in the same run ck_0008 had
1-1 HELD 45/50 with 1-4 at 0. One brain per level so far; no checkpoint yet holds two levels at once
on the strict bar.

## E23. One brain for three levels (pre-registered 20:56)
runs/st_U: from brains/joint_11_14_ck10, rows 1-1 (runs/fix_cad_body3), 1-3 (runs/d13c, st_13d's pile)
and 1-4 (runs/d14b, st_14b's pile), lessons 1/3 each, a tie-aware DAgger round on EACH of the three
levels every 3 chunks, picked by the joint rule (weakest level's SEL flags first).
- H23: some joint-best checkpoint reaches TEST >= 6/10 on all three levels within 2.5 h. Falsified if
  none does (then per-level brains are what it takes, at this size and setup).

## Milestone (20:55): one brain holds 1-1 and 1-3 on the strict bar
runs/st_13d ck_0013 (brains/one_11_13.npz), picked by the joint SEL rule (SEL [15, 19] of 24), never on
TEST or HELD: **1-1 TEST 8/10, HELD 40/50; 1-3 TEST 6/10, HELD 42/50.** The same weights clear both
levels on >= 80% of held-out starts. On the pre-registered 10-start TEST bar, 1-3 misses (6/10).
The best 1-3-only checkpoint (ck_0012: 1-3 TEST 9/10, HELD 28/50) has 1-1 at 0/50, so the joint pick
mattered. **H20 holds** (1-3 TEST >= 6/10 with 1-1 HELD >= 40/50 at the same checkpoint).
