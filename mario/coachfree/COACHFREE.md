<!-- RESUME -->
## RESUME (11/05 20:50)
- **Worlds 6-7 load and play**: every start of SEL/HELD/POOL is playable on all 8 levels (no level ends during the
  idle frames; RAM world/stage matches). First sight, greedy, 50 HELD starts, mean x / flags (runs/firstsight/):
  CatoA ck91 / CatoB ck84 / CiceroB ck93: 6-1 578/512/752, 6-2 304/304/343, 6-3 383/390/367, 6-4 448/484/424,
  7-1 568/321/970, 7-2 162/385/162, 7-3 593/758/670, 7-4 347/326/332. **0 flags in 1,200 attempts.** Many levels are
  a deterministic wall (all 50 starts die at the same x: 6-2 at 303, 6-3 at 383, 7-4 at 347); 7-2 (underwater) CatoA
  and CiceroB stand at x 162 until stuck.
- **B1 pilot on 6-1 (CatoA ck91, 30 rounds x 16 sampled attempts, 3,000 lessons per arm): NOT DISTINGUISHABLE**
  by the pre-registered rule. D (HELD mean over rounds 26/28/30 minus start 578.1): b1 +107.4, anch +125.4.
  b1 - anch = -18.0, floor = 125.4 (|D(anch)|). Split-half gap: b1 26.0, anch 7.7.
- The headline finding is the drift: lessons on worlds 1-3 anchor rows ALONE swing greedy HELD on 6-1 between 502
  and 1,370 from one eval to the next (b1: 404 to 936). Any coach-free effect smaller than ~±400 x on a single
  eval, or ~±125 x on a 3-eval mean, is invisible on this metric.
- Practice (sampled, T 0.26): b1 last 10 rounds 875 vs first 10 864 (no climb); anch 794 / 801. 0 practice flags in
  480 attempts per arm. b1 - anch practice +81 is inside b1's own half gap (73) and not the pre-registered
  comparison (that is vs reload, still running).
- Forgetting (SEL 24 starts, 12 training levels): start 940.2 x / 17 flags (curve.csv chunk 91). b1 886.2 / 2,
  anch 974.2 / 1. b1's drop exceeds anch's by 88 x < 100: **not "forgot" by the rule**. But both arms lost 15-16 of 17
  flags (2-4 14 -> 1-2): the start checkpoint was the best-of-SEL pick, so part of that is regression from a lucky
  pick, and part is the same fragility.
- 22:00 reload arm done: frozen brain's practice last-10 = 831.4 (HELD at round 30 = 578.1, greedy is deterministic).
  Pre-registered secondary: b1 practice - reload = **+43.6 vs floor 38.5 (reload's split-half gap): just over**.
  anch - reload = -37.4. Weak: one seed of lesson order, a floor from a single 5-vs-5 round split, and the primary
  (greedy HELD) says nothing. Suggestive at most; it is the thing to re-test with more seeds.
- 22:52 **B3 (Brain.live, actor 0.02, reward dx/20) on 6-1: WORSE.** D -17.5 (b3 - anch = -142.9, floor 125.4);
  practice last 10 = 721.5 vs reload 831.4 (-109.9, floor 38.5), falling from 801.5 in the first 10.
  **Forgetting: SEL 12-level mean 940 -> 424.8, 0 flags (anch 974): B3 forgot badly**, the upstream #169 erosion
  (the reward path erodes the imitation skill), even at actor 0.02 and about a third of moments aroused.
- Compute note (Idle priority starved by the 8 plateau runs; the 6-1 b1/anch pair took 11 h for ~1 h of CPU work).
  6-1 `b3` done (Brain.live). 6-2 was dropped from the queue
  (runs/SKIP_6-2) to fit. When they finish: `python analyze.py 6-1` adds them to runs/pilot_6-1/score.json.
- NEXT: see "What to try next" at the bottom. In short: fix the metric before the method (greedy HELD swings are
  bigger than any plausible B1 effect), and test B1 where sampling actually finds progress greedy doesn't.
<!-- /RESUME -->

# Coach-free training on worlds 6-7 (2026-11-05)

Sean, 11/5: "add in worlds 6 and 7 to be trained coach free methods". Worlds 6 and 7 (6-1..6-4, 7-1..7-4) are the
coach-free development ground: methods are built, piloted and debugged here and these levels NEVER get coach labels
or coach demonstrations. World 4 stays the clean never-coached final test (nothing in this folder runs on it:
`cf.check_level` refuses World 4). Worlds 1-3 are the coached training set: the starting brains were trained there and
their piles are the anchor rows.

Design source: vault Research/Cadence Training Methods 2026-11-05 (Part B1, B3) and its review (B1 cost fix,
calibrated temperature, |w| <= 1, coach-row anchors, practice on POOL / score on HELD).

## Files
- `cf.py`: dev level list, Idle priority (SetPriorityClass), `no_coach()` (any coach call in this process raises),
  World 4 refusal, worker count (2, or 1 while a game runs).
- `firstsight.py`: start check + greedy first sight on 50 HELD starts for 3 cold-start checkpoints.
- `b1.py`: the B1 driver (`calib`, `run --arm b1|anch|rand|reload`).
- `brains0/`: copies of the cold-start checkpoints (CatoA ck91, CatoB ck84, CiceroB ck93 = each run's best joint SEL
  checkpoint). The running plateau-test runs are not touched.

## B1 design (as built)
- Practice: per round, 16 sampled attempts from POOL starts. Same starts and sampling seeds in every arm.
- Sampling: one greedy settle, then our own draw from the motor softmax at temperature T. T is calibrated per
  level so the start brain's mean P(greedy action) on its own greedy path (4 POOL attempts) is 0.85. Our own draw
  means no eligibility phases run (no actor learning; about 3x cheaper than `act(temperature=)`).
  Measured on 6-1 (CatoA): T 0.05 gives P(greedy) 0.986 (the old retry/gauntlet default barely explores), T 0.2
  gives 0.92, T 0.3 gives 0.79.
- Filter (self-imitation): baseline B = median max x of the round's 16 attempts. An attempt with max x M > B + 32
  is good, weight w = min(1, (M - B)/256) (so |w| <= 1). Its rows are the decisions with x >= B - 128 up to the one
  that first reached M, minus the last 3 if it died. Rows go into a FIFO buffer (4,000 rows).
- Lessons: 100 batches of 32 rows per round, via `Brain.lesson` (labels = the brain's own pressed buttons):
  8 own rows sampled with p ~ w (importance carried by sampling frequency, every lesson weight 1) + 24 anchor rows
  from the worlds 1-3 coach piles (fixed set of 3,000 rows per level drawn with p ~ W x crit 3, as ctrain).
  eta 0.001 (as ctrain), cold settles (as ctrain and act mode).
- Arms on one checkpoint (CatoA ck91):
  - `b1`: as above.
  - `anch`: same practice, same lesson count, all 32 rows anchors. Drift control: what lessons alone do to play on L
    (greedy play is fragile; any weight change moves it).
  - `reload`: practice only, frozen brain (the sampled-attempt curve of the unchanged brain; its greedy HELD score is
    the round-0 number).
  - `rand` (if time): own rows = every decision of every attempt, uniform. Tests the filter (the 10/29 E4 failure).
- Eval: greedy act mode on the 50 HELD starts of L every 2 rounds. Forgetting: SEL (24 starts) on the 12 training
  levels at the end of each arm vs the start brain.

## Pre-registered pilot metric (written before any pilot round ran, 11/05 09:20)
- Primary, per level: D = (mean greedy HELD max x over the last 3 evaluations, rounds 26, 28, 30) minus (the start
  brain's greedy HELD mean). B1 "moved something" on L if D(b1) - D(anch) > floor, where floor = max(|D(anch)|,
  split-half gap). Split-half gap = |mean per-start change over even-indexed HELD starts - same over odd-indexed| for
  the b1 arm (same 3-eval average). Otherwise: not distinguishable.
- Secondary: practice mean max x over the last 10 rounds, b1 vs reload; split-half floor = |mean of rounds 21-25 -
  mean of rounds 26-30| of the reload arm (the frozen brain's round-to-round noise). Flags in practice and HELD.
- Forgetting: SEL 12-level mean x at the end vs start. "Forgot" if b1's drop exceeds anch's drop by more than 100 x
  (the order of the cold-start night's chunk-to-chunk SEL swings, SD 98-184 x).
- Levels: 6-1, then 6-2 (every first-sight attempt dies at x 303-306: a clean wall).
- Added 09:40, before any result: a `b3` arm (b3.py, Brain.live) on both levels, judged by the same rule
  (D(b3) - D(anch) > floor, floor from the b3 arm's own split-half gap) and the same forgetting rule.

## B3 design (as built)
- The checkpoints have no arousal genes and `retune` can't add them, so b3.py recomposes Cato (seed 0) with
  arousal=True (same connectome, pre/post checked) and copies every learned array in. Greedy parity with the original
  checkpoint is asserted on 3 starts before each run (passed on 6-1 in the smoke test).
- Every practice decision goes through `live`: reward clip(dx/20, -1, 1), death -1, flag +1, stuck 0 (all done).
  actor_eta 0.02 (composed 1.0; upstream #169 erosion), youth 0 (a trained brain), critic untrained at the start.
  Same practice starts as b1. Smoke test: 6-1 attempts ran about half aroused (28/59, 19/91 moments), 1.5 s each.

## Results: 6-1 pilot (11/05, CatoA ck91, T 0.2626)
Greedy HELD mean x (50 starts) by round; start 578.1:

| round | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 404 | 404 | 620 | 870 | 404 | 466 | 936 | 676 | 637 | 404 | 404 | 404 | 857 | 477 | 724 |
| anch | 938 | 835 | 586 | 769 | 547 | 592 | 1370 | 536 | 853 | 1216 | 1078 | 872 | 502 | 1095 | 514 |

- b1 kept 65-891 own rows per round (median ~560); the buffer was full (4,000) from round 8.
- x 404 recurs in b1 (6 of 15 evals): one specific greedy death that the brain falls back into. anch never shows it.
- Score file: runs/pilot_6-1/score.json (analyze.py).
- Verdict: not distinguishable. Forgetting: not "forgot" by the rule (b1 -54 x, anch +34 x, start 940).

## What this says, and what to try next
1. **The metric is the bottleneck, not (yet) the method.** 3,000 anchor-only lessons move 6-1 greedy play by up to
   +790 x on one eval. That is the fragility finding (E9) again, now measured on an unseen level with a clean control.
   A single-checkpoint greedy HELD number can't judge a coach-free method. Options: average more evals (every round,
   last 10), score sampled play at T (smoother, and it's what the method trains), or freeze the anchor drift by
   comparing b1 against several anch seeds (lesson-order seeds) to get a real drift distribution.
2. **B1 had little to imitate.** Practice never flagged and never climbed; the "good" segments are the top half of
   16 attempts that mostly die in the same few places. Self-imitation can only sharpen what sampling already finds.
   Test it where sampling finds something greedy doesn't: the deterministic walls (6-2 x 303, 6-3 x 383, 7-4 x 347).
   Pre-register "share of sampled attempts past the wall" and "does greedy pass the wall" (binary, no drift problem
   below the wall). That's the 4-4 lava-pit question on a dev level.
3. **Keep the anchor share but lower lesson volume per round.** 100 lessons per 16 attempts is ~6x more lesson
   mass than evidence; the drift is driven by lessons, not by the own rows. Try 20-30 lessons per round.
4. B3 lost on every count and wrecked worlds 1-3 (940 -> 425). Drop the reward path as tried; revisit only with
   much lower actor rates plus anchor lessons interleaved, or macro-actions (review open question 5).
5. Compute: Idle priority under 8 busy plateau runs gives ~0.1-0.2 of a core. Pilots should run after the plateau
   test frees the CPU, or with 1-2 cores reserved.
