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
- [ ] Training continues to ~05:52. assist.py (brain plays, coach rescues after a death/stall, exact
  rewind by replaying buttons) on the r04 brain: flag, brain pressed 41.3%, 13 rescues.
- [ ] Record the brain's run, build page, publish to docs/mario/.

## Doom precedent
Upstream muellerberndt/cadence #118 "Build and qualify the full native Doom expert" is an open
roadmap epic (M5), with no posted playing results. Its first skill is Basic combat, it allows only
rendered information to enter the player, and it says a frame/action classifier or host planner is
a component control, not the complete expert. No public Cadence Doom demo was found.

## Notes
- Venv: `P:/projects/cadence-arcade/.venv-mario` (uv, py3.11, cadence-net 0.91.1 from the local
  clone, non-editable, numpy<2 for old gym).
- Build: see README for the vcvars command.
