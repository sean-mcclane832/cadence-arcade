<!-- RESUME -->
## RESUME (11/05, in progress)
- Pilot running. Results below as they land.
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
