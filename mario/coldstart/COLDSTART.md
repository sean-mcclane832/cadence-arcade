<!-- RESUME -->
## RESUME (11/03 09:30 EST, night over, scored, page pushed)
- **Verdict (pre-registered rule): NOT DISTINGUISHABLE for both memory arms.** Kc = 74 (LivyB, suspended 18:40-23:07,
  sets the common range; 37,000 lesson batches). AUC / END (mean SEL x over 12 training levels):
  CatoA 783.4 / 979.3, CatoB 800.5 / 1090.9, CiceroA 790.5 / 1076.6, CiceroB 734.7 / 926.9, LivyA 719.5 / 1118.9,
  LivyB 738.0 / 966.0. Arms: Cato 792.0 / 1035.1, Cicero 762.6 / 1001.8, Livy 728.8 / 1042.5.
  Floor (largest within-arm gap): AUC 55.8, END 152.9. Cicero vs Cato: AUC +7.1 / -65.8, END +97.3 / -164.0.
  Livy vs Cato: AUC -63.9 / -62.5 (both below by more than the floor), END +139.6 / -124.9 -> not "hurts" (rule
  needs both metrics). runs/morning/curve.json, morning.md.
- HELD 50 (greedy, act mode) at ck74 and each last ck: 12 training levels 1-18 flags of 600 (mostly 1-4, 2-4, 3-2),
  mean x 850-1074. **World 4 first sight: 0 flags on 4-1..4-3 for all 11 checkpoints** (mean x 344-751, best CatoB
  ck84, furthest 1,859 on 4-1). **4-4: every attempt of every brain ends at exactly x 138** (550/550). Checked 11/3:
  it is a real death, the lava pit right after the start. Holding right dies at x 136-138; a scripted run+jump
  pattern gets past it (x 1,060). So no harness bug: no brain ever jumps that first pit. runs/morning/held.json.
- Gauntlets (demos, 48 = 6 runs x greedy/sampled x from 1-1/2-1/3-1/4-1, 5 lives, last ck): one level cleared in
  total, CiceroA ck108 SAMPLED (temp 0.05) 1-1 on its 4th life, then 1-2 deaths at 918, 886. Every other run died out
  on its first level. Best World 4: CatoB ck84 greedy 4-1 deaths 346, 1842, 1842, 1320, 344. Clips copied to
  docs/mario/gauntlet/c01_CiceroA.mp4, c02_CatoB_w4.mp4. runs/morning/gauntlets.json.
- Page docs/mario/ updated (Cold-start section, scoreboard row 4-1..4-3, gauntlet rows, Latest). Full report in the
  vault: Research/Cadence Report 2026-10-29 to 11-03.
- 11/3 14:50: .venv-concepts upgraded 0.91.3 -> 0.92.1 (non-editable, --no-deps; old copy in .cadence-0.91.3-backup/).
  Parity: 3 brains (CatoA ck91, CiceroA ck104, LivyA ck83) x greedy/sampled x 1-1/2-4, action hashes identical (12/12).
- Sean's hypothesis (11/3): memory and no-memory learn at about the same pace, but memory plateaus much higher. Tonight's
  ~20 h is the early part of the curve; testing it needs the runs continued until the curves flatten.
- NEXT: deeper/smarter coach for 2-1 and 3-4;
  no-coach learning curve on a held-out level. 4-4 checked (real lava-pit death, valid test).
<!-- /RESUME -->

# Cold-start night (2026-11-02): Cato / Cicero / Livy from scratch, worlds 1-3, World 4 held out

Design: vault Plans/Mario Memory Brains 2026-11-01, section "From-scratch pair: running night of 11/2" (approved by
Sean 11/2). Methods from vault Research/Cadence Roadmap Review 2026-11-01. Code and specs reused from ../memory
(MEMORY.md: mem.py specs, burn-in lessons, cold settles in act mode, stationary memory guess for pile rows, echo
synapse_rate 0.125). Nothing in ../memory was changed; this folder copies and extends.

Sean, 11/2: "The issue isn't that the memory didn't help. Guarantee. We are going to do the cold starts tonight..."
"Add up to level 16. Include world 2 in the gauntlet fully too." "Train on world 1, 2 and 3. Let 4 be the test."

## Files
| file | what |
|---|---|
| cs.py | level lists, start sets, per-level coach horizon patch (game._lean_plans(h)) |
| startcheck.py | every start playable on all 16 levels -> data/startcheck.json |
| coachcollect.py | coach demonstrations (World 1's recipe) -> pile, coach paths, ordered rows, coverage per level |
| mkbrains.py | 6 fresh connectomes (brains0/) + act-mode sanity play |
| ctrain.py | night driver (mtrain.py copy: 12 levels, equal lessons, SEL curve, FIFO throttle, no auto-stop) |
| cgauntlet.py | gauntletm.py copy: explicit --mode greedy / sampled, --first / --last level |
| morning.py | morning step: pre-registered metric, fixed-checkpoint HELD (16 levels), gauntlets + clips |

## Start sets
World 1's sets (concepts/cplay.py), the same idle-frame offsets on every level: SEL 24 (0..78 step 3 minus 32-41;
curve + selection), HELD 50 (32-41, 80-99, 100-138 even; never trained or picked on), POOL (0-79 minus 32-41; DAgger).
startcheck.py: on all 16 levels the game survives 139 idle frames, so every start is playable (no level ends before
the first decision; the 11/1 retry crash came from offsets of 300+ frames).

## Coach coverage (step 2)
coachcollect.py, World 1's recipe on every level (6 clean coach runs from starts 0/3/8/13/20/27 + 3 with 12% random
buttons + 2 with 25%), HORIZON 16, every decision labelled (data/h16/<L>/coverage.json). Clean clear rate and where
the clean runs end (f flag, d died, s stuck):
| level | clean clears | clean max x | ends | noisy clears | rows (critical) | used |
|---|---|---|---|---|---|---|
| 1-1 | 6/6 | 3161 x6 | ffffff | 2/5 | 2,838 (713) | h16 |
| 1-2 | 6/6 | 3161 x6 | ffffff | 0/5 | 2,519 (561) | h16 |
| 1-3 | 5/6 | 2425, one death at 1727 | fffdff | 1/5 | 1,926 (381) | h16 |
| 1-4 | 5/6 | ~2250, one death at 1591 | fffffd | 1/5 | 1,619 (325) | h16 |
| 2-1 | **0/6** | 1760, 3026, 1760, 1554, 1760, 3026 | dsddds | 0/5 | 2,069 (518) | h16, labels up to where it reaches |
| 2-2 (water) | 5/6 | 3161 x5, 2482 | fffffs | 0/5 | 4,576 (73) | h16 |
| 2-3 | 6/6 | 3593 x6 | ffffff | 0/5 | 2,605 (329) | h16 |
| 2-4 | 6/6 | ~2260 | ffffff | 1/5 | 1,701 (323) | h16 |
| 3-1 | 6/6 | 3193 x6 | ffffff | 0/5 | 2,487 (596) | h16 |
| 3-2 | 6/6 | 3337 x6 | ffffff | 1/5 | 2,709 (428) | h16 |
| 3-3 | 3/6 | 2409 x3, dies at 795 x3 | dfdfdf | 0/5 | 1,358 (235) | h16 |
| 3-4 (castle) | **0/6** | ~1434 (one 1695, one 434) | dddddd | 0/5 | 1,292 (414) | h16, labels up to where it reaches |
- 2-1, deeper horizon (clean runs only): HORIZON 24 -> 0/6, all stuck at x 722-723 (pressed against the pipe with a
  Koopa behind it); HORIZON 24 + stuck limit 150 (coach allowed to wait) -> 0/6, stuck at 723 or 869 (standing under
  the ? blocks, every plan judged fatal). HORIZON 32 + stuck 150: started, stopped after 25 min to launch (would not
  finish before ~12:45; 24 was already worse than 16). So deeper search did NOT help here: HORIZON 16 reaches furthest
  (mean 2,106, two runs to x 3,026, near the end), and its labels (up to where each run ends) are what 2-1 trains on.
  (The old "0/11 at HORIZON-16" was the same 0 clears; with today's level-agnostic background it gets much further.)
- 3-4 (Bowser's castle, firebars) 0/6, deaths at ~1,434: no deeper probe (out of time; 2-1 showed deeper isn't
  automatically better). 3-3 (mushroom platforms) clears half; the other half falls at x 795.
- No level dropped. data/levels.json = data/h16/<L> for all 12; no data/horizon.json (coach horizon 16 everywhere,
  DAgger labels too).
- Cost: 6,133 s on 6 workers for 132 coach runs.

## Arms (step 4)
Six fresh connectomes (mkbrains.py, brains0/, nothing transplanted), mem.SPECS compose + tune, 2,000 inputs, 7 buttons,
modules (256,), teaching normalized, learning_surprise False. Seed A = 0, seed B = 1.
| run | spec | memory parts |
|---|---|---|
| CatoA, CatoB | Cato | none (working_memory_amplitude 0) |
| CiceroA, CiceroB | Cicero | thought (working trace amplitude 3.0) + button (efference 1.0, decay 0.8) |
| LivyA, LivyB | Livy | Cicero's + echo (history decay 0.9, synapse_rate 0.125 on the echo synapses) + place (memory_choice; store never written in this recipe, as on 11/1) |
Sanity (brains0/parity.json): all six play an act-mode attempt (cold settles, greedy) on 1-1 and 2-1 without errors.
Seed A of every spec stands still (all NOOP, stuck at x 40); seed B moves (x 281-450). Untrained, so expected.

## Driver (step 5): ctrain.py
memory/mtrain.py's recipe, unchanged where possible: lessons of 32 rows, half pile rows (stationary memory guess) and
half ordered rows primed with K=3 burn-in acts + exact heard history / efference; eta 0.001; critical rows x3; act-mode
play with cold settles. Changes: 12 training levels with equal lesson shares; chunk = 500 lesson batches; SEL (24
starts) on all 12 levels after every chunk -> runs/<run>/curve.csv; DAgger every 6 chunks, 1 attempt per level from
POOL, labelled every 3rd decision + the last 60 (coach horizon per level from data/horizon.json); no HELD during the
night; no stall rule; stop at 08:00 (checked after each chunk); 1 play worker, 1 numba thread per run (6 runs on 8
cores). Throttle while a game runs (memory/gamecheck.py: hoi4, factorio, Minecraft java): 2 shared slots, FIFO tickets,
game recheck every 60 s inside the wait. Smoke test (runs/_test/ct, LivyB, 2-1, 2 tiny chunks + a DAgger round): ran
clean.

## Pre-registered (written before launch, before any training result)
Hypothesis (Sean's): trained from scratch with the memory parts live from the first lesson, a memory brain learns the
12 training levels faster than the same recipe without memory (the 11/1 night lost on warm starts, where W1_F2ck110
was trained without memory and the memory synapses started at zero).
- **Primary metric:** the learning curve of SEL mean max x averaged over all 12 training levels, vs lessons.
  Common lesson range Kc = the smallest last chunk any of the 6 runs reaches by 08:00 (equal lessons, not hours).
  AUC = trapezoid area of that curve over chunks 0..Kc divided by Kc (a mean x); END = the curve's value at chunk Kc.
  Arm value = mean of seeds A and B.
- **Noise floor:** the gap between seed A and seed B within an arm; the floor per metric = the largest of the three
  arms' within-arm gaps.
- **Memory "helps"** (per memory arm, Cicero and Livy) only if BOTH seeds beat Cato's same-letter seed (A vs CatoA,
  B vs CatoB) by more than the floor, on AUC AND on END. "Hurts" by the mirror rule. Anything else: not distinguishable.
- Secondary (reported, no verdict): SEL clears; HELD 50 on all 16 levels at chunk Kc and at the last checkpoint;
  World 4 first sight (4-1..4-3 mean x and flags; 4-4 maze castle separately); 5-life gauntlets on each run's last
  checkpoint, greedy and sampled (temperature 0.05) reported separately, from 1-1 through World 2 fully then World 3
  and World 4, plus world-start gauntlets (from 2-1, 3-1, 4-1) so every world is played. Gauntlets are demos.
- Expectation (honest): fresh brains learned ~2/200 clears in 14 worker-hours before, so clears will be near zero;
  the curves are the result.
- Morning script: morning.py (not run tonight). `morning.py --steps curve` gives the verdict in seconds.

### Ease-up (18:40, Sean using the PC)
All 6 runs at Idle priority; seed-B runs (CatoB, CiceroB, LivyB) SUSPENDED (frozen, not killed) to free cores. Seeds A keep going. Restore planned 23:07. The pre-registered metric uses the lesson range all 6 reach, so B seeds being behind shrinks that range; report it.
- 23:07: restored (seed-B runs resumed, all Normal priority).

## Results (morning 11/03, morning.py)
See RESUME for the numbers. Notes:
- Runs stopped 08:04-08:17 (CiceroA and LivyA finished a chunk past 08:00). Last chunks: CatoA 104, CatoB 84,
  CiceroA 108, CiceroB 93, LivyA 90, LivyB 74. Seed-B runs were frozen 18:40-23:07 for Sean's PC use, so Kc = 74.
- Livy costs more per lesson but the metric is at equal lessons, so that's fair by construction.
- The within-arm seed gaps (56 AUC, 153 END) are as big as any between-arm difference: with 2 seeds this design can
  only catch a large effect. None showed up.
- Morning step cost: HELD ~25 min, gauntlets ~5 min on 8 workers.
