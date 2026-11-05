# A deeper Mario coach (2026-11-03 to 11-05)

Fix #1 from the roadmap review (vault Research/Cadence Roadmap Review 2026-11-01; Plans/Cadence Roadmap 2026-11-02,
"Order" item 4): the brains learn by imitating the coach (DAgger), and the coach could not clear 2-1 or 3-4, so no
brain could learn to. World 4 is the held-out test: the coach was never run there (harness.py and plug.py refuse it).

## Files
| file | what |
|---|---|
| smartcoach.py | coach variants (base, verify, smart). `install(g, "smart")` swaps the coach on one Game |
| plug.py | `worker_init(deep, name, opts)`: patches game.Game in a worker so every coach call uses a variant; World 4 games refuse coach calls |
| harness.py | coverage harness: clean coach runs from coachcollect's 6 starts (0, 3, 8, 13, 20, 27), stuck limit 40, 900 decisions, seconds per call. 1 worker while a game runs, else up to 4, rechecked every minute, BelowNormal priority |
| trace.py | one run with x / y / button / call cost per decision and the last screens (`--prefix` replays a recorded run to start at a hard spot) |
| springprobe.py, springrate.py | the 2-1 springboard probes |
| ramtest.py | is the 2 KB RAM a full save state? (no, see below) |
| runs/ | traces and coverage runs (runs/cov_final = the table below) |

## What was wrong (traces, base coach)
The base coach (game.py) simulates each of 26 plans for 16 decisions from a saved state and takes the first button of
the plan that ends furthest right without dying. Traces of the failures:
- **3-4 x 1,434 (Bowser's castle):** at x 1,376 it starts a jump that survives exactly the 16 decisions of the
  horizon; one decision later every plan dies (it hits a wall and drops into the lava). The plan "survived" only
  because the horizon ended just before the death.
- **2-1 x 1,760:** same shape: jumps a piranha pipe, ends the horizon mid-air over the gap behind it, falls in.
- **2-1 x 3,026 (stuck):** the tall wall near the end needs the springboard: land on it and press jump at the right
  moment. No menu plan does that, so the coach runs into the wall for 40 decisions. Random button sequences clear it
  rarely (springrate.py from a coached state at the spring: 5 of 600 with short holds, 1 of 600 with long holds).
- **3-3 x 795:** the same horizon-ends-mid-air death on the lifts.
- Why a deeper horizon made 2-1 worse (COLDSTART.md): scoring by end x at 24 decisions makes every plan that jumps
  the pipe at x 723 die to the Koopa behind it inside the horizon, so "press against the pipe" scored best.
  A longer horizon makes the coach both more timid and still blind just past its new horizon.

## What I tried (numbers)
1. **verify (doom check):** a plan counts as alive only if one of 8 short tails (run, wait, left, run+jump, and
   wait/back-up then jump) survives 8 more decisions from where it ends. From the 3-4 failure spot it then clears
   3-4 (Bowser included); from the 3-3 spot it clears 3-3. From 2-1's spring it is still stuck (no tail bounces).
2. **stall search (random sticky shooting):** when the best plan gains < 24 px for 3 calls in a row or all plans die,
   try 48-120 random sequences of 32-40 decisions (each button held for a random 1-10, later 1-4 or 1-10,
   decisions) and keep the best verified one. With N=120 and long holds it found the spring after 20 tries (21 s
   per call); mixing short holds made it find it within a few tries.
3. First 12-level pass (verify + shooting, commitment only for long shot sequences): 2-1 stuck at x 530 on 2 starts
   (small Mario jumping in place against the stairs for 40 decisions: dithering between near-equal plans).
4. **commitment:** the rest of the chosen plan is offered again next decision and wins ties. 2-1 6/6, 3-4 6/6 with
   no cache (runs/cov_smart2_t).
5. Second pass on the other 10 levels (runs/cov_smart2_rest): 3-3 4/6, 1-3 6/6, 1-4 6/6, 2-2 6/6, but 1-1 4/6 and
   2-4 5/6, below base (6/6 each). Traces: the coach waited (button "wait") for 40 decisions at 1-1 x 2,370 and 2-4
   x 1,714. Running on is judged "doomed" there (the danger past plan + tail needs a jump no tail has), while
   "wait, then run" pushes the same danger past the check, so waiting always looks safer.
6. **fallback against procrastination:** if x has not passed its best for 8 decisions, shoot; if no shot helps, label
   with base scoring for that decision. Retests from the 4 failing runs (1-1 s13, 2-4 s3, 3-3 s3, 3-3 s27): all
   4 flags.
7. Dead end: **RAM as an extra save state** (ramtest.py). nes-py has one backup slot, so every branch replays from
   the root. Writing the 2 KB RAM back is not a full state: the frame counter (0x09) and the RNG (0x7A7-) differ
   from the first decision on. So the search stays replay-from-root.

Side finding: a coach call is not invisible. After `_backup / _frame_advance / _restore` the next real frame differs
from never simulating (2-1 drifts 1 px at decision 72). One dummy simulated frame per decision reproduces a coached
run exactly (300/300 decisions), so harness `--prefix` replays do that. Worth knowing for DAgger, whose labelling
pass replays a brain's buttons with coach calls only at labelled decisions.

## Result: coverage, 12 training levels, clean runs from 6 starts
Before = base coach at HORIZON 16 (coldstart/data/h16/<L>/coverage.json, same starts and stuck limit; my base traces
reproduce those runs: 2-1 s0 1,760, 2-1 s3 3,026, 3-4 s0 1,434, 3-3 s0 795). After = smart (runs/cov_final).
f flag, d died, s stuck, in start order 0, 3, 8, 13, 20, 27.

| level | before clears | before ends | before max x | after clears | after ends | after max x | after s/call |
|---|---|---|---|---|---|---|---|
| 1-1 | 6/6 | ffffff | 3161, 3161, 3161, 3161, 3161, 3161 | 6/6 | ffffff | 3161, 3161, 3161, 3161, 3161, 3161 | 1.85 |
| 1-2 | 6/6 | ffffff | 3161, 3161, 3161, 3161, 3161, 3161 | 6/6 | ffffff | 3161, 3161, 3161, 3161, 3161, 3161 | 1.90 |
| 1-3 | **5/6** | fffdff | 2425, 2425, 2425, 1727, 2425, 2425 | 6/6 | ffffff | 2425, 2425, 2425, 2425, 2425, 2425 | 3.64 |
| 1-4 | **5/6** | fffffd | 2246, 2251, 2246, 2246, 2267, 1591 | 6/6 | ffffff | 2244, 2245, 2245, 2244, 2267, 2267 | 1.46 |
| 2-1 | **0/6** | dsddds | 1760, 3026, 1760, 1554, 1760, 3026 | 6/6 | ffffff | 3193, 3193, 3193, 3193, 3193, 3193 | 2.35 |
| 2-2 | **5/6** | fffffs | 3161, 3161, 3161, 3161, 3161, 2482 | 6/6 | ffffff | 3161, 3161, 3161, 3161, 3161, 3161 | 2.14 |
| 2-3 | 6/6 | ffffff | 3593, 3593, 3593, 3593, 3593, 3593 | 6/6 | ffffff | 3593, 3593, 3593, 3593, 3593, 3593 | 2.32 |
| 2-4 | 6/6 | ffffff | 2264, 2245, 2264, 2249, 2256, 2268 | 6/6 | ffffff | 2267, 2266, 2248, 2251, 2263, 2267 | 2.00 |
| 3-1 | 6/6 | ffffff | 3193, 3193, 3193, 3193, 3193, 3193 | 6/6 | ffffff | 3193, 3193, 3193, 3193, 3193, 3193 | 1.80 |
| 3-2 | 6/6 | ffffff | 3337, 3337, 3337, 3337, 3337, 3337 | 6/6 | ffffff | 3337, 3337, 3337, 3337, 3337, 3337 | 1.95 |
| 3-3 | **3/6** | dfdfdf | 795, 2409, 795, 2409, 795, 2409 | 6/6 | ffffff | 2409, 2409, 2409, 2409, 2409, 2409 | 3.33 |
| 3-4 | **0/6** | dddddd | 1434, 1434, 1431, 1433, 1695, 434 | 6/6 | ffffff | 2244, 2266, 2268, 2245, 2267, 2267 | 2.66 |
| all | 54/72 | | | 72/72 | | | 2.25 |

## Cost per coach call
Same machine, 4 workers at once, BelowNormal (runs/cov_base_cost vs runs/cov_final, start 0):
| | base s/call | smart s/call | ratio |
|---|---|---|---|
| 1-1 | 1.14 | 1.78 | 1.6x |
| 2-1 | 1.15 (dies at 1,760) | 2.38 | 2.1x |
| 3-1 | 1.41 | 1.87 | 1.3x |
| 3-4 | 0.98 (dies at 1,434) | 2.69 | 2.7x |
| all 4 | 1.22 | (all 72 runs) 2.25 | **1.8x** |
- Per-call p95: 4.5-8 s on smart (stall searches: 48 sequences x 40 decisions ~5-8 s each), vs ~2.2 s on base.
- Where it goes: commitment and the doom check add a little to every non-trivial call; the fast path ("run is
  safe and fast") costs ~0.2 s instead of ~0.08 s because it also runs one tail; stall searches are rare but slow.
  3-3 and 1-3 (lifts, waiting) are the dearest at 3.3-3.6 s/call.
- DAgger-shaped check (the real use): a trained brain (coldstart/runs/CatoB/best_joint.npz) plays 2-1 and the coach
  labels every 3rd decision + the last 60, as mem.episode does in ctrain: 75 labels in 243 s with smart vs 160 s
  with base (brain play included), **1.5x**. Labels agree on most rows (first 20: 18 the same).
- Whole-pass wall time for this table: 3.7 h on 4 workers for 72 runs (base needed ~1.7 h on 6 workers for 132 runs
  with screens).

## How to use it
- In ctrain.py (opt-in, default unchanged): `python ctrain.py ... --coach smart` (optionally
  `--coach-opts '{"patience": 8}'`). DAgger labels then come from the smart coach; coach labels on World 4 raise.
- Anywhere else: `plug.worker_init(deep, "smart", {})` as a ProcessPoolExecutor initializer, or
  `smartcoach.install(game_obj, "smart")` on one Game.
- New coach demonstrations (piles) for 2-1 and 3-4: coachcollect.py builds them with the base coach. To rebuild with
  smart, run it with `initializer=plug.worker_init, initargs=(deep, "smart", {})` (not done here; the brief was the
  coach only, and the existing piles are what the 11/2 runs trained on).
- Labels: same 7-button output, chosen from {run, run+jump, wait, left}; the critical weight rule is game.py's
  (x10 when the best other first button ends > 50 px behind; shot sequences are always critical).
- Optional escape cache: `data/escapes.json` ({level: [[x, y, sequence, area], ...]}) is tried before shooting.
  Not built: the coverage above is without it. Harness results carry each run's found escapes (`found`) if a cache
  is wanted to cut DAgger cost later.

## Not solved / caveats
- Coverage is coach-from-start only (6 clean starts per level). DAgger asks from brains' off-route states; the smart
  coach gives a label from any state (it never needs a route), but whether its labels there are as good is untested
  beyond the 2-1 check above. The procrastination fallback is what keeps it from waiting forever there.
- It is still a search with a horizon. A spot that needs > 40 decisions of exact play before any x progress shows
  could still beat it; none of the 12 training levels does at these starts.
- Not deterministic across code changes in the way base is: shooting is seeded by (level, decision, x), so runs
  repeat exactly, but any change to the menu or the tails changes which sequences are found.
- Waits count against the harness's stuck limit of 40 decisions (no new max x). 3-3's lifts come close.
- The piles (coach demonstrations) in coldstart/data/h16 are still base-coach piles: 2-1 and 3-4 piles stop where the
  base coach died. Rebuilding them with smart is the obvious next step before a training run that uses --coach smart.
- Not done: escape cache (data/escapes.json), an ablation of each piece on all 12 levels (the numbers above are
  each step's measured effect on the failing runs, not a full ablation).
