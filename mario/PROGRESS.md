# mario/ progress (overnight 2026-10-29)

Goal: ONE Cadence 0.91.1 brain playing Super Mario Bros. World 1-1 from the screen, as far as
possible (ideally the flag), with a replay page.

## Stages
- [x] Feasibility: nes-py 8.2.1 + gym-super-mario-bros 7.4.0 build on Windows with MSVC 2019
  Build Tools (vcvars64 + DISTUTILS_USE_SDK=1). ~776 emulator frames/s on one core. ROM ships in
  the pip package; nothing ROM-related is in git.
- [x] Cadence 0.91.1 speed: 6,656 inputs, one region of 256, 32-row lesson ~0.15 s on CPU after a
  one-off ~35 s numba compile.
- [x] Coach (emulator look-ahead planner, game.py) clears 1-1 from every unjittered start
  (6/6, ~270 decisions); ~1.2 s per label per worker on the shared machine.
- [x] Baselines (random 594 median x, always-run 312, jump rhythm 311; 0 flags).
- [x] Round 0 (3,259 coach screens, 306 lessons): brain only runs, dies at first Goomba (x 312).
  Raised lessons to 2,000-4,000 per round, resumed with --resume --extra-teach at 02:11.
- [x] Attempt 1 (grey 4x4 blocks, 6,656 inputs, 7 buttons, rare buttons up-weighted): training
  agreement stuck at ~0.65 = the "run" share; brain always runs, dies at the first Goomba (x 312)
  after 2 DAgger rounds. A backprop MLP on the same screens: 0.73 held-out. Kept in
  runs/attempt1_grey7.
- [x] Two-button coach (run / run+jump only) tested: cannot clear 1-1 (dies x 1221 / 2386, gets
  stuck at 435 with a longer horizon), because Mario sometimes has to wait or back off. Dropped.
- [x] Attempt 2 (colour 8x8 blocks): readout swung between "all run" and "all run+jump"
  (agreement 0.70 <-> 0.23). Stopped; its coach screens were lost (train.py saved data only after
  teaching; fixed: data is now saved right after collection).
- [x] Sweep (sweep.py, attempt-1 screens, held-out = later coach runs): with raw [0,1] pixels every
  teaching setting answers ONE button for every screen (input-independent). Sky-distance input
  (|grey - sky grey|, sky = 0) at 8x8 blocks makes answers screen-dependent; best was
  normalized + learning_surprise: 0.62 held-out (majority 0.58), train 0.74, ~0.12 s per lesson on
  one thread (7x faster).
- [x] Attempt 3 (since 03:15, runs/main): 8x8 grey sky-distance, 1,664 inputs, 7 buttons,
  teaching="normalized" + learning_surprise, uniform sampling, 3,000 lessons per round, 20 coach
  runs then DAgger rounds of 15 episodes. Rounds 0-4: eval stuck at x 312.
- [x] 03:54 resumed with cost-sensitive labels: the coach marks a label critical (weight 10) when
  every other first button ends >50 px behind (usually: dies); lessons drawn by weight. 8 attempts
  per round from the start, 7 from points along the coach path. Brain fits critical screens 88%.
- [x] BUG found 04:12: pool workers cached the brain by path only, so every eval and DAgger attempt
  used the first brain each worker loaded (eval flat at 306-312 for that reason). Fixed (reload on
  mtime/size). Old evals flagged `eval_stale` in history.json; checkpoints re-measured fresh with
  evalckpt.py: r03 median 312, r06 median 722 (best 898). First honest live eval, r07: 594 median.
  DAgger labels collected before the fix are still valid coach labels, just less on-policy.
  predict() is deterministic (same buttons on a rerun).
- [x] Training ended 05:56 after round 21 (deadline), 1,664 inputs, ~20k labelled screens.
  Honest evals (median x of 5 starts): r03 312, r06 722, r07 594, r08 1417 (best 1783), r09 596,
  r10 725, r11-21 between 306 and 722, r21 594. Peak at r08, then a plateau with big swings.
- [x] Baselines (runs/baselines.json): random 594 (best 898), always-run 312, jump rhythm 311,
  backprop MLP on the final screens 596 (best 1561). The final-round brain is no better than random
  by median; the best round is ~2.4x random.
- [x] Assisted runs (final.py, round-10 brain): flag on starts 0/5/11/23, brain pressed 59/66/69/60%
  of buttons, 11/8/7/11 rescues. Start 17 looped on rescues and was stopped. Round-4 brain: 41%.
- [x] Page: docs/mario/ (brain-alone replay = round 8 start 0, x 1783; assisted = round 10 start 11),
  index entry, README section. Real NES frames replayed from stored buttons, checked to end at the
  same x.

## Next steps
- Steady the brain: the plateau after round 8 suggests the label pile keeps growing with conflicting
  coach choices. Try a 2-3 button coach that can still wait (run / run+jump / left), more lessons on
  recent DAgger rounds, or lower teaching rates late in training.
- Keep the best round: checkpoint every round from the start (the watcher only began at round 9).
- Try the continuing stream (act with the working trace) instead of predict, as upstream advises.

## Doom precedent
Upstream muellerberndt/cadence #118 "Build and qualify the full native Doom expert" is an open
roadmap epic (M5), with no posted playing results. Its first skill is Basic combat, it allows only
rendered information to enter the player, and it says a frame/action classifier or host planner is
a component control, not the complete expert. No public Cadence Doom demo was found.

## Notes
- Venv: `P:/projects/cadence-arcade/.venv-mario` (uv, py3.11, cadence-net 0.91.1 from the local
  clone, non-editable, numpy<2 for old gym).
- Build: see README for the vcvars command.

## 2026-10-29 in-order run (runs/inorder, levels.py)
One brain plays 1-1, 1-2, 1-3, ... in order; first sight (3 solo attempts before practice) is the
headline. Page: docs/mario/levels/ (built by build_levels_page.py, clips by levels_update.py).
- Start brain: runs/main/brain_e10.npz (overnight round 10; round 8's brain was never saved, the
  checkpoint watcher began at round 9). So 1-1 is not a real first sight.
- Fixed before launch: (1) the smoke run's WinError 5 in DuplicateHandle came from an outside stop:
  taskkill took the workers first, multiprocessing.Pool spawned replacements, the parent was gone
  when they unpickled its pipe handles. Pool also hangs forever if a worker dies mid-task. Now a
  ProcessPoolExecutor that rebuilds itself and retries (then runs the task in the main process).
  (2) Coach stuck in 1-2 at x 1,218 (low ceiling over a gap: every running jump hit the bricks and
  fell). A brute-force search found run 7-8 then a 1-2 decision hop; added run-then-hop plans.
  Then stuck on top of the exit pipe at x 2,674: pipe entry (RAM 0x0E = 2/3) now scores as
  progress and x carries across the area change. Coach clears 1-2 (ct_1-2_lean_v2/v3).
  (3) Caps: 38 min practice per level, rounds ~5 min. (4) Added old-level PLAY: 10% of each
  round's self-play decisions are spent on a random earlier level (reward learning + own good
  stretches added to that level's data), on top of the 3-in-32 replay rows.
- Launched 12:09, until 19:25, 5 workers + main, detached (runs/inorder/run.sh).
- 1-1 (pre-trained): first sight [594, 899, 434]; 9 rounds, 40 min, capped; best solo r6 [898, 1128, 598] (median 898; overnight r10 had 725). Coach labels 3,372, own rows 18,632.
- 1-2: first sight [183, 184, 183]; 9 rounds, capped, 2 rollbacks; best solo r2 [482, 482, 177]. Retention after 1-2: 1-1 fell to [303, 306, 302] (from 898 median): strong forgetting with 10% old play.
- 1-3: first sight [311, 311, 311]; 14 rounds, capped; best solo r14 [626, 746, 776] (median 746). Retention after 1-3: 1-1 [2020, 594, 594] (2,020 = furthest solo 1-1 yet), 1-2 [198, 196, 184].
- 1-4: first sight [234, 234, 234]; capped; best solo r11 [551, 551, 551]. Retention after 1-4: 1-1 [695, 723, 696], 1-2 [198, 196, 189], 1-3 [420, 420, 420]. 2-1 first sight [322, 322, 322].
- 2-1: first sight [322, 322, 322]; capped; best solo r7 [457x3]. Retention after 2-1: 1-1 [700x3], 1-2 [322, 354, 291], 1-3 [419x3], 1-4 [437x3]. 2-2 (water) first sight [514, 514, 514], best first sight so far, but a swim level with no pits at the start.
- 2-2 (blind): first sight [514x3]; best solo r2 [2225, 1496, 1194] (median 1,496); stopped 16:01 after r6 for
  the senses change (no retention after 2-2). Blind run saved in runs/inorder_blind/.
- Bug found 16:01: set_rates' late teaching-rate retune (round 3+) always failed (cadence 0.91.1
  refuses learning_eta under teaching="normalized"). Effect: teaching rates were never lowered late;
  actor rates set at rounds 1-2 persisted. Kept the same behaviour in run 2, minus the failing call.

### Per-level first-sight baselines (baselines_levels.py, median x, starts 0/6/12; random 10 seeds)
| level | blind first sight | run | run+jump | random | fresh blind | fresh senses | beats best? |
|---|---|---|---|---|---|---|---|
| 1-1 | 594 (pre-trained) | 315 | 314 | 450 | 309 | see json | yes, but trained overnight |
| 1-2 | 183 | 196 | 196 | 338 | 40 | | no |
| 1-3 | 311 | 321 | 317 | 300 | 705 | | no |
| 1-4 | 234 | 234 | 229 | 331 | 551 | | no |
| 2-1 | 322 | 308 | 306 | 384 | 120 | | no |
| 2-2 | 514 | 162 | 514 | 1,234 | 1,199 | | no |
Also measured for 2-3 to 3-2 (runs/baselines_levels.json, *_senses.json). Blind first sight beats
the best baseline on no new level.

### Run 2: new senses (fresh brain, runs/inorder, launched 16:03, until 19:25)
- Input 6,128: eyes 4x4 px blocks of the present frame (3,328) + 3 signed frame differences at
  8x8 (3 x 832), each frame normalized to its own background grey and 95th-percentile contrast;
  proprioception from RAM about Mario only (x speed, y speed, float state, rising, facing, power,
  hurt/star timers, time left; 38 population-coded values x 8 copies = 304). No enemy/tile/pit RAM.
- Speed: game step + obs 9 ms, lesson 0.12 s, brain.step 68 ms (blind ~27 ms). Same caps,
  same 10% old-level play.
- 1-1 (senses, fresh): first sight [40x3]; 6 rounds, 40 min, capped; solo medians 295, 315, 434, 315, 315, 594. Blind continuing brain over its 40 min: 594 -> best 898 (but ~2.5 h of overnight 1-1 training first). 1-2 first sight [198, 196, 189].
- 1-2 (senses): first sight [198, 196, 189]; 6 rounds capped, never beat round 0 (best state = the one it arrived with); retention 1-1 [594, 594, 596] kept.
- 1-3 (senses): first sight [321x3] (= always-run); capped; best r5 [419x3]. Retention: 1-1 fell to [316x3] (from 594), 1-2 [198, 258, 260]. 1-4 first sight [234x3].
- 1-4 (senses): first sight [234x3]; capped, never beat round 0. 2-1 first sight [324x3].
- 2-1 (senses): first sight [324x3]; 6 rounds to 19:28 deadline; best r2 [338x3]; from r4 the
  brain stood still (x 40) even after a rollback. Job ended 19:28 (EXIT 0, no worker failures all day).

### Summary 2026-10-29
First sight never beat the best baseline on a new level, blind or with new senses. No solo
clears. Retention poor in both. The new senses did not help within 40 min per level from scratch.

## 2026-10-29/30 night: the learning fix (details and every experiment: LEARNING_FIX.md)
Root causes, each with its decisive evidence:
1. **Interface aliasing.** SMB starts a jump only on a fresh A press. The brain saw only the screen,
   so it could not tell whether its thumb was still on A: the coach's label for one screen flipped
   between "let go" (critical) and "jump", and on a static screen (against a pipe) the greedy brain
   repeated one button forever. 31 of the last 60 overnight solo runs ended that way; DAgger made
   the answer at the pipe flip from round to round. A backprop MLP had the same problem.
2. **Noise in the teaching pile.** In-order rounds added ~1,700 rows of the brain copying its own
   *sampled* actions per ~130 coach labels; training agreement 0.15-0.4; the 2-1 "stand still"
   collapse came back after a rollback, i.e. from the data.
3. **learning_surprise drift.** With lessons only on wrong rows, Cadence's fit peaked (0.85) then
   collapsed (0.66, critical 0.35 at 6k lessons); lessons on every row rise steadily to 0.92 / 0.90
   held-out, the MLP's level.
Recipe (dagger.py): eyes as before (8x8 grey blocks, 2 frames) + body sense (buttons held last
decision + Mario's own proprioception, no world RAM), lean 4-button coach, coach labels only,
DAgger where the brain plays alone and the attempt is then replayed for the coach's labels (last 60
decisions + every 3rd), Cadence normalized teaching at eta 0.001 WITHOUT surprise, cold predict.
Results: MLP control clears 1-1 solo at DAgger round 2. **Cadence clears 1-1 solo**: round-2 brain
of runs/fix_cad_body3 flags on starts 0, 5, 11 (3/5 eval starts) and 8 of 10 unseen start jitters.
First sight on 1-2..2-1 with that brain: better than always-run / run+jump / untrained, not better
than random buttons on 1-2, 1-4, 2-1 (E7). Retention while practising 1-2 with 50% replay: E8.
