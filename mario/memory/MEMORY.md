<!-- RESUME -->
## RESUME (11/02 07:30 EST, night over, all training stopped)
- **Verdict: memory lost every pre-registered test.** Details: "End of night (11/02)" at the bottom.
  - Short memory helps (Cicero/Livy beat Cato by 20+ HELD/200 or +150 2-1 mean x, AND memory-off costs 10+): LOST.
    Top-3 HELD W1 means: Cato 74, AgrippaX 70, Cicero 82 (+8), Livy 55 (-19). 2-1 mean x: Cato 671, Cicero 524,
    Livy 572. Cicero memory off at play GAINS 18 (88 -> 106); Livy all off -28 (43 -> 15): its weights lean on the
    memories, but it plays worse than Cato.
  - Place memory works (retry +300 by life 10, or a level only KEEP clears): LOST. Livy life 10: 1-2 904 vs 1,320,
    clears 10 vs 18 of 80; 2-1 501 vs 530. (Agrippa, AgrippaX lost 11/1.)
  - Fallback (SEL mean x +15%): LOST (Cicero +3%, Livy -7%).
- **Noise floor (A/A = AgrippaX vs Cato):** top-3 mean gap 4 (74 vs 70), best-pick gap 34 (119 vs 85), one line's
  checkpoints span 33-119/200; rough SE of a top-3-mean difference ~25 clears. 2-1 mean gap 82 x. Every memory
  difference is inside it. AgrippaX ablations (place off, x zeroed, both) identical to on (85): store empty, as built.
- **Correction (11/02):** `gauntletm.py --mode act` is NOT greedy. MemPlayer only special-cases 'greedy'/'retry', so
  'act' samples at temperature 0.05 and calls learn() mid-attempt (weights byte-identical after, runs/_actcheck.py,
  runs/eon/actcheck.out; it reproduces m06 exactly, greedy does not). m05-m08 are sampled play; HELD scores are greedy.
  Labeled on the page and in the gauntlet table. Fix next time: an explicit --mode greedy/sampled, reject unknown modes.
- Gauntlets (demos): m05 Cato ck20 1-1, 1-2; m06 Cicero r2 ck1 all of World 1; m07 Livy ck57 1-1; m08 AgrippaX ck18 1-1.
- Brain-cam: viz/record_brain.py --act (records settles DURING act-mode play) + brain_video.py memory panels.
  Cicero r2 ck1 on 1-2/38 and 1-1/0 (docs/mario/gauntlet/brain_cicero-*.mp4).
- Page docs/mario/ updated (Memory night section, roster, retry chart, verdicts, noise floor, m06 demo, 1-3 46/50
  Cato_ck20 labeled best of many). Pushed.
- **NEXT** (vault Plans/Cadence Roadmap 2026-11-02, in order): (1) done tonight: noise floor published. (2) Offline
  checks: MLP/CNN clone on the ~179k W1 pile rows scored on World 2; linear probe of W1_F2ck110 for RAM labels on W1 vs
  W2 screens; 2-1 transfer vs number of training levels. (3) Methods fix: fresh start set, 3 seeds, equal lessons
  (not wall time), FIFO throttle slots, one primary metric (share of HELD 2-1 starts past x 723), checkpoint rule
  fixed in advance; also fix the gauntletm mode bug. (4) Deeper coach + 8-12 SMB1 stages, held-out stages
  (World 2 never coached). (5) No-coach 2-1 curve: 100-200 own attempts, hindsight goals, Brain.live as the memory arm.
<!-- /RESUME -->

# Memory brains (2026-11-01 night)

Lab notebook for the vault plan Plans/Mario Memory Brains 2026-11-01 (approved by Sean, revised after review).
Same habits as ../concepts/CONCEPTS.md: hypothesis + falsifier BEFORE each run, then the result, corrections in place.
Cadence 0.91.3 (../../.venv-concepts). Start sets as before: SEL 24 (selection only), HELD 50 (never trained or picked on).
Starting brain: ../brains/W1_F2ck110.npz (HELD 50 / 11 / 6 / 25 on 1-1..1-4), never modified.

| Brain | Memory parts (page names) | Cadence settings |
|---|---|---|
| Cato | none (control: W1_F2ck110 continued, same driver) | working_memory_amplitude=0 |
| Agrippa | place memory | memory_choice=True, memory_context=0, memory_demonstrations=0, memory_fan_in=32, working_memory_amplitude=0 |
| Cicero | thought memory + button memory | working trace amplitude 3.0, efference_amplitude=1.0, efference_decay=0.8 |
| Livy | thought + button + echo + place | Cicero's + history_decays=(0.9,) + memory_choice=True (memory_context 0.5), memory_demonstrations=0, memory_fan_in=32 |

Choices made before any result (13:10): echo = ONE history block at decay 0.9 (about 10 decisions = 0.7 s of game; two
blocks would almost triple Livy's synapses). Button memory decay 0.8 ("what it pressed and for how long"; the body sense
already holds the buttons of the last decision). New memory projections start at 0.1x their composed size ("quiet"),
so the warm brain starts close to W1_F2ck110 and learning grows them.

Code: mem.py (specs, transplant, act-mode player, retry lives, OUR blame window), evalm.py (act-mode scores),
p0_transplant.py, p0_blame.py, gamecheck.py.

Game check (13:12): the javaw.exe running since 10/31 17:23 is the ATLauncher launcher window itself (classpath
ATLauncher.exe, 25 s of CPU in 20 h), not Minecraft. gamecheck.py throttles on hoi4.exe or a Java process running
Minecraft, not on the idle launcher.

## Phase 0

### 0.1 Transplant + parity (13:05)
How: compose the memory connectome (seed 0, same as W1's line), match every old synapse by its (pre, post) pair
(Cadence appends the efference / history regions after the motor neurons, so old neurons keep their indices), copy
efficacy, momentum state, biases, gains and the critic. Checked: every old pair exists with the same sign, count and
plasticity.
- Predict parity: **0 of 3,000 stored screens answer differently**, for all four brains (runs/p0_transplant.json).
- Act-mode parity, 1-2 HELD (memory parts off): Cato 11/50 mean 1,598, Cicero with thought + button off 11/50 mean
  1,598, identical to each other. W1_F2ck110 in predict mode: 11/50 mean 1,580. So act mode (warm settles from the
  last moment) is not bit-identical to predict mode (cold settles), independent of the transplant. Cicero with its
  memory ON (quiet 0.1, untrained): 8/50 mean 1,405. Full four-level parity: below.
- Full HELD, four levels (runs/p0_parity.json, act mode as Cadence does it, warm settles): the four brains with their
  memory parts off are identical to each other on every start of every level (transplant is exact). But act mode vs
  W1's predict mode: **1-1 38/50 vs 50/50** (0 of 50 action sequences identical), 1-2 11 vs 11, 1-3 3 vs 6, 1-4 25 vs 25.
  Cause: `act` settles warm from the last moment's activity, `predict` and every lesson settle from rest; the brain was
  trained cold. That would have handed every memory brain a 15-clear handicap that has nothing to do with memory.
- **Fix (driver-side, ours): act-mode players settle every decision from rest** (`mem.make_cold`: the brain's warm start
  is set to None; memories still enter the drive). Re-check (runs/p0_parity_cold.json): Cato in cold act mode = W1 in
  predict mode exactly: 1-1 50/50 mean 3,161, 1-3 6/50 mean 1,627. Training (prime burn-in acts) uses the same cold
  settles, so play and lessons match.
- Quiet start: with the memory projections at 0.1x, Cicero and Livy (memory on, untrained) lost clears on 1-1 (34, 35 of
  50 in cold mode) and Livy 1-3 0/50. **Changed to 0.0x** (memory projections start at zero, learning grows them): every
  memory brain then starts exactly at W1_F2ck110's play with its memory on, so any later difference, and any ablation
  loss, comes from what the memory synapses learned. (EP steps on a zero synapse are pre x post contrasts, so they
  can grow.)

### 0.4 Burn-in K (runs/p0_burnin.json, 13:20, on the 0.1x brains)
Real attempts in act mode (1-2/32, 1-3/80, 1-1/90), true memory state before each decision vs rebuilt from K burn-in
acts. Heard history and button memory are computed exactly from the stored screens and pressed buttons (cosine 1.000
at every K, no burn-in needed). Thought memory (working trace): K=0 0.0, **K=1 0.9995 (min 0.989), K=2 0.9999 (min
0.998)**, same for Livy. Pass (>= 0.95) at K=1. Using **K=3** for margin, since the trace's dependence on the past grows
as the prefrontal synapses learn (they start at zero now). Re-check K on the trained brains at the end.

### 0.2 / 0.3 Blame test, key overlap, records-only (pre-registered 13:35)
Agrippa (place memory only), one retry life per start on 1-2 and 2-1 (8 held starts each), sampled at temperature 0.05
(about 5% of decisions not the greedy choice), learn(r, done, next_obs) after each, death = -1, flag = +1, stuck = 0,
actor rates 0. Decisive decision = the last critical coach call (weight 10) in the 40 before death where the brain's
button differed.
- Records-only: weights + critic byte-identical (hash) before and after every life. Any difference = bug, stop.
- Blame rule fixed now: **Cadence's own write is enough if, on at least half the deaths, the place vote for the taken
  action at the decisive decision is <= -0.1** (a fifth of what a fresh record of -1 reads after one write at rate 0.5).
  Otherwise the retry curves add OUR blame window (labeled ours): on a death, write -1 into the place records for the
  buttons pressed 5, 10, 15, 20, 25 and 30 decisions before it (6 writes, one per decision, batch 1).

**Result (13:33, runs/p0_blame.json).** 16 lives: 12 deaths, 1 flag (1-2/38), 3 stuck.
- Blame: a decisive decision was found in 10 deaths. Place vote for the taken button there <= -0.1 in **3 of 10** (1-2/41
  -0.22 at 16 back, 1-2/95 -0.20 at 19 back, 2-1/80 -0.23 at 5 back); the other 7 read exactly 0 (the decisive button was
  a different one from the dying button, so the -1 on the dying button says nothing about it). **Below half: the retry
  curves use OUR blame window** as fixed above. Cadence's own write itself works as documented: the dying decision reads
  -0.525 for its button.
- Key overlap (cosine of separated place keys, mean): **0.59 at 1 tile, 0.53 at 5 tiles, 0.50 at 20 tiles**, 0.71
  between neighbouring decisions. The place keys are only weakly local: a death writes about half its value into every
  screen of the level for that button. Seen directly: vote at 5 decisions back is -0.26 to -0.36 for the dying button
  even where nothing went wrong. Expect place memory to act more like "this button is risky on this level" than "here".
- Records-only: **weight VALUES identical after every life** (efficacy, biases, gains, critic). The byte hash differed:
  32,918 memory synapses flipped from -0.0 to +0.0 (zero-step on a negative zero from the quiet scaling). Fixed in the
  transplant (no signed zeros); after the fix a full retry life leaves weights byte-identical (runs/_dbg.py).
- Exploration at temperature 0.05: 3-6% of decisions not the greedy choice on 1-2, 7-19% on 2-1 (flatter readout on an
  unseen level).

### Night launched (13:37)
Cato / Cicero / Livy, runs/night/<name>, mtrain.py, 2 workers + 2 threads each, K=3, lessons 50% pile rows (empty
memory) / 50% ordered rows (coach paths + tonight's DAgger, primed), F2's level shares (1-1 .15, 1-2 .25, 1-3 .40, 1-4
.20), DAgger on all four every 6 chunks (8 attempts per level, act mode), SEL every chunk, HELD on each new per-level
best, stop 05:45 (leaves time for end-of-night retry curves, ablations, gauntlets, page). Ordered coach rows
(runs/seq, 13:34): 1-1 776, 1-2 703, 1-3 565, 1-4 471 labelled rows in 22 replayed coach paths.
Agrippa retry curves running alongside (3 workers): blame window keep vs control, then keep without blame (exploratory).

Correction (13:55): both retry runs CRASHED (ValueError "cannot step in a done environment" while building the game).
Cause: life 10's offset of 229 idle frames on top of held starts 90 and 95 on 1-2 (319 / 324 idle frames) ends the game
before the first decision. Checked every start x offset on 1-2 and 2-1: only those two fail. Fix: life 10's offset is 19
(all offsets now 0, 37, 83, 110, 131, 151, 173, 197, 211, 19, every combination checked). Reran 13:57, KEEP + blame vs
CONTROL first, then the no-blame KEEP arm. Thanks to the main session for catching it.

### 0.5 Agrippa retry curves vs reload control (pre-registered 13:50, before any retry result)
Agrippa (W1_F2ck110's weights + place memory, nothing trained), 10 lives in a row from each of 8 held starts
(32, 35, 38, 41, 80, 85, 90, 95) on 1-2 and on 2-1. Life i starts after start + (0, 37, 83, 110, 131, 151, 173, 197,
211, 229)[i] idle frames. Every life: sampled acts at temperature 0.05 + learn() (actor rates 0, weights checked
unchanged), death -1, flag +1, stuck 0. Arm KEEP: the same brain all 10 lives, place records kept. Arm CONTROL: the same
brain loaded fresh before every life (records wiped), same offsets, same sampling seeds. Reported: max x per life (not a
running best), median over starts, clears per life.
- Win (the plan's bar): KEEP's median max x at life 10 beats CONTROL's by 300+ on 1-2 or 2-1, or KEEP clears a level
  within 10 lives on a start where CONTROL never clears it.
- Lose: neither. A flat or worse KEEP curve gets reported as flat.
- Blame variant per 0.2's rule (Cadence's own write, or + OUR window 5..30).
Lesson-speed note (13:55): Livy (history block of 2,000 neurons) costs ~3x Cato per lesson (0.38 s vs 0.14 s per
primed batch of 32 on 2 threads). Arms get equal workers and equal hours, so Livy gets fewer lessons. That is the
honest price of the echo, and it goes in the scoreboard.

### Night, first chunks (13:55)
All three arms play chunk 0 identically (SEL 20 / 7 / 9 / 11, HELD 50 / 11 / 6 / 25 = W1_F2ck110), the parity check
in the live driver. After chunk 1: Cato 7 / 2 / 2 / 20, Cicero 0 / 0 / 10 / 7, Livy 0 / 1 / 0 / 0 (SEL of 24). Read with
care: W1_F2ck110 is a SEL-picked peak of a line that swings wildly chunk to chunk (F2 chunks 100-111 on 1-1 SEL: 12, 14,
4, 12, 24, 21, 5, 5, 8, 4, 20, 6). So falling off the peak is expected for every arm, and the plan's "Cato itself gets
worse" branch (fallback: SEL mean x +15%) may well apply. Livy's all-level drop is the biggest; watching it.

### Correction (13:58): night restarted as runs/night2 (v1 in runs/night, stopped after 2-3 chunks)
Diagnostic on v1's Cicero ck3 (1-1 SEL, runs/diag_cicero3_1-1.json): memory on 0/24 mean 1,247; button memory off
0/24 mean 1,216; **thought memory off 11/24 mean 2,735**. So the thought-memory synapses had already learned something
that breaks play. Likely cause: half of every lesson batch was pile rows taught with ALL memories empty (no order known),
a state that never occurs in play after the first decision, while the thought synapses grew from the few ordered rows.
Fix (all arms, same driver): pile rows get a stationary guess instead of empty memory: thought memory = K burn-in acts
on that same screen, echo = that screen, button memory = the button the body sense says was pressed last (the held
bits are already an input, so nothing new leaks). Ordered rows unchanged (exact). All three arms restarted from ck0 at
13:58 in runs/night2. (An attempt to restart into runs/night appended ~1 min of log there before I stopped it; ignore
the tail of runs/night/*.out.)

### 0.5 Result: Agrippa retry curves (14:13, runs/p0_retry_blame.json). **Place memory loses its pre-registered test.**
Median max x per life over 8 held starts (lives 1..10), KEEP (records kept, + OUR blame window) vs CONTROL (reloaded):
| level | arm | median max x, lives 1-10 | clears per life | starts cleared at least once |
|---|---|---|---|---|
| 1-2 | KEEP | 1197, 989, 863, 863, 870, 745, 861, 376, 862, **1271** | 1,1,0,2,0,0,0,0,1,1 (6) | 3 of 8 |
| 1-2 | CONTROL | 1197, 1152, 1110, 1097, 1320, 1333, 459, 865, 70, **1109** | 1,2,0,1,0,2,0,1,0,0 (7) | 5 of 8 |
| 2-1 | KEEP | 396, 335, 397, 338, 341, 343, 338, 338, 340, **464** | 0 | 0 |
| 2-1 | CONTROL | 396, 531, 530, 399, 402, 396, 341, 530, 530, **337** | 0 | 0 |
- Life 10: +162 on 1-2, +127 on 2-1 (bar: +300). One start (1-2/85) KEEP cleared and CONTROL never did, but CONTROL
  cleared three starts KEEP never did (41, 80, 95): that clause reads as noise, not a win. Over lives 2-9 KEEP is mostly
  BELOW the control on 1-2 (mean x 776 / 745 / 770 / 555 vs 1,108 / 1,456 / 560 / 921 at lives 5-8).
- Read: with keys this overlapping (0.5 across the level), every death says "the button I died on is bad" nearly
  everywhere, mostly run / run+jump, so the records push the brain off its good habits as often as off the mistake.
- Offset artifact, both arms: lives 7-9 (172-211 idle frames) often die at x 60-80 right after the start (an enemy
  reaches an idle Mario). Same for KEEP and CONTROL, so the comparison stands; the absolute numbers there are low.
- Texted Sean 14:14.
Exploratory (labelled, not pre-registered): AgrippaSparse = same but memory_expansion 4096, memory_winners 8 (key cosine
at 1 / 5 / 20 / 60 decisions apart: 0.61 / 0.41 / 0.37 / 0.31 vs Agrippa's 0.75 / 0.58 / 0.53 / 0.48), and Agrippa KEEP
without our blame window. Running.

### Correction (14:20): Livy restarted (runs/night2/Livy; first try in runs/night2/Livy_v1)
Livy ck1 played 0/24 on every level (dies at x 234 on 1-4). Part-by-part switch-off on 8 held starts
(runs/diag_livy1.json): echo off -> 1-1 1/8 mean 2,337, 1-4 3/8 mean 2,171; thought, button or place off -> no change
(still dead). **The echo broke it.** Likely cause: the echo projection is 512,000 synapses (2,000 inputs x 256), all
starting at zero with fresh normalized-step statistics, so 250 lessons of near full-size steps on every one of them add
up to a huge drive into the association region. Fix: the echo synapses take 1/8 of the normal step (Cadence's
per-synapse `synapse_rate`, 0.125 = 256 / 2,000, so its total push is about the size of the thought memory's 256-wide
projection). Livy restarted from ck0 at 14:20, about 20 min behind the others; same workers and stop time.
Cicero v2 at chunk 1 played fine (18 / 5 / 11 / 18), so the stationary pile guess fixed the thought-memory problem.
Exploratory results (14:30, runs/p0_retry_sparse.json, runs/p0_retry_noblame.json), same 8 starts, same CONTROL:
| arm | 1-2 life-10 median | 1-2 clears in 80 lives | 2-1 life-10 median |
|---|---|---|---|
| CONTROL (reloaded every life) | 1,109 | 7 | 337 |
| Agrippa KEEP + our blame window (pre-registered) | 1,271 | 6 | 464 |
| AgrippaSparse KEEP + blame (keys 4,096 / 8 winners) | 963 | 2 | 461 |
| Agrippa KEEP, Cadence's own writes only | 861 | 1 | 341 |
Sparser keys did not rescue it, and without our blame window the records only hurt. So locality is not the only problem.
Every variant is at or below simply reloading the brain. The 2-1 life-10 "gains" (+124 to +127) sit inside the swing
the CONTROL itself shows from life to life (337 to 531). Place memory, as Cadence builds it on these screens, does
not learn from dying here.

## Gauntlets (5 lives, gauntletm.py, every attempt clipped; runs/gauntlet/)
| # | time | brain | mode | cleared | deaths (level: x) | clip |
|---|---|---|---|---|---|---|
| m01 | 14:30 | Agrippa (ck0 = W1_F2ck110 weights + place memory) | remembers across lives (temp 0.05, + our blame window) | 1-1 | 1-2: 874, 978 (stuck), 2,150, 675, 825 (stuck) | m01_Agrippa.mp4 |
| m02 | 14:57 | Cato best_joint (= ck0) | act mode, nothing kept | 1-1 | 1-2: 872, 883, 1,320, 468, 1,321 | m02_Cato.mp4 |
| m03 | 14:57 | Cicero best_joint (= ck0, memory synapses still 0) | act mode, nothing kept | 1-1 | 1-2: 872, 883, 1,320, 468, 1,321 | m03_Cicero.mp4 |
| m04 | 14:57 | Livy best_joint (= ck0, memory synapses still 0) | act mode, nothing kept | 1-1 | 1-2: 872, 883, 1,320, 468, 1,321 | m04_Livy.mp4 |
| m05 | 23:47 | Cato best_joint (ck0020) | act mode, nothing kept, "one brain" (demo, not evidence) | 1-1, 1-2 | 1-2: 978 (stuck); 1-3: 1,256, 1,140, 1,942, 1,910 | runs/gauntlet_night/m05_Cato.mp4 |
| m06 | 01:03 | Cicero r2 best_joint (r2 ck0001 = leg-1 ck0025) | act mode, nothing kept, "one brain" (demo, not evidence) | 1-1, 1-2, 1-3, 1-4 | 1-2: 1,344; 1-3: 936; 2-1: 471, 722, 330 | runs/gauntlet_night/m06_Cicero.mp4 |
For comparison, W1_F2ck110 without memory (g16, last night): 1-1, then 1-2: 872, 883, 1,320, 468, 1,321.
m02-m04: at 14:57 every night arm's joint pick (best SEL sum over the four levels) was still chunk 0, i.e. W1_F2ck110's
weights with the memory synapses at zero. So the three gauntlets are the same game, frame for frame, and match g16
exactly: a gauntlet-level parity check, not a test of memory. Per-level bests have moved (Cicero 1-4 HELD 44/50 at ck5,
Cato 1-2 HELD 9/50 at ck5), but no later checkpoint beats ck0 on all four levels together yet. Rerun when a joint pick
moves (the end-of-night agent).

### Exploratory: AgrippaX, place memory keyed on x from game RAM (Sean's idea, 14:55; where am I only, decisions still from the screen)
Not pre-registered; read with the same +300 bar. Same as Agrippa (W1_F2ck110 weights, place memory only, memory_context=0,
memory_demonstrations=0, fan-in 32, same retry protocol / starts / offsets / our blame window), but the observation gets 112
extra inputs: Mario's level x from RAM (0x6D * 256 + 0x86) as bumps every 2 tiles, about 4 tiles wide, height 8. Cadence
keys the place records on the observation code, and I found no API to give the episodic key a separate code, so the x
block is extra SENSORY input whose synapses into the cortex are all zero (transplant maps the shifted neuron indices;
only the initial `sign` array differs, because a wider sensory region shifts the random draws, and the settle uses the
copied efficacies). So only the episodic key sees x.
(Source, 0.91.3: the place records key on `Brain._placed(x)`, generic.py ~4227, which without a learned code is the
observation code, then the fixed `PatternSeparator`, stream.py ~2106. No separate-key option.)
- Parity: AgrippaX plays the same buttons as Agrippa (= W1_F2ck110) on 1-1/32, 1-2/35, 1-2/80, 1-3/85 (every action
  identical).
- Key cosine on a real 1-2 attempt, decisions 1 / 5 / 20 / 60 apart: Agrippa 0.69 / 0.58 / 0.56 / 0.52, AgrippaSparse
  0.56 / 0.42 / 0.39 / 0.39, **AgrippaX 0.80 / 0.29 / 0.11 / 0.07**: now local.
- **Result (14:47, runs/p0_retry_x.json, same protocol as 0.5: 8 held starts, 10 lives, temp 0.05, + our blame window).
  Not a win.** CONTROL replayed Agrippa's CONTROL exactly (16 of 16 series identical), so the comparison is clean.
  | level | arm | median max x, lives 1-10 | clears per life | starts cleared at least once |
  |---|---|---|---|---|
  | 1-2 | KEEP (x key) | 1197, 1261, **1875**, 861, 860, 876, 272, 898, 372, **1089** | 1,3,1,1,0,0,0,0,0,2 (8) | 4 of 8 (32, 38, 41, 90) |
  | 1-2 | CONTROL | 1197, 1152, 1110, 1097, 1320, 1333, 459, 865, 70, **1109** | 1,2,0,1,0,2,0,1,0,0 (7) | 5 of 8 (35, 38, 41, 80, 95) |
  | 2-1 | KEEP (x key) | 396, 339, 413, 398, 344, 340, 338, 399, 339, **437** | 0 | 0 |
  | 2-1 | CONTROL | 396, 531, 530, 399, 402, 396, 341, 530, 530, **337** | 0 | 0 |
  - Life 10: **-20 on 1-2, +100 on 2-1** (bar +300). KEEP cleared two starts CONTROL never did (32, 90), CONTROL cleared
    three KEEP never did (35, 80, 95): same reading as Agrippa's, a swap, not a win.
  - The one hint: lives 2-4 on 1-2 KEEP mean 1,600 vs CONTROL 1,365 (life 3 median 1,875 vs 1,110, 4 clears vs 3 in
    lives 2-4). Then it fades: lives 5-10 KEEP is at or below CONTROL. Pooled lives 2-10: 1-2 1,137 vs 1,104, 2-1 425
    vs 472. Small sample (8 starts), so the early bump may be noise; it is NOT a result.
  - vs Agrippa (same blame window): life-10 1-2 1,089 vs 1,271, 2-1 437 vs 464. Local keys did not beat overlapping
    ones. So key locality was not the bottleneck. Best guess: the write itself. Our blame window puts -1 on whatever
    button was pressed 5..30 decisions back, which says "not that" but never "do this instead"; with only 7 buttons and
    a cortex that already prefers run / run+jump, a few -1s mostly push it onto a worse second choice. Positive records
    only come from a flag (+1), which ends the attempt.
  - Label: "place memory keyed on x from game RAM (where am I only; decisions still from the screen)".
  - Records-only (14:50, runs/_xcheck.py -> runs/x_weightcheck.out): 10 KEEP lives on 1-2/32 with the blame window
    (69 place writes) replay the json's series exactly, and efficacy / bias / gain / critic are byte-identical after.
  - Exploratory, KEEP with Cadence's own writes only (no blame window; 14:56, runs/p0_retry_x_noblame.json), vs the
    same CONTROL: 1-2 median by life 1197, 925, 1664, 672, 1118, 897, 267, 458, 473, **952** (9 clears in 80 lives,
    starts 35, 38, 41, 85, 90, 95 = 6 of 8); 2-1 396, 722, 402, 626, 722, 530, 338, 335, 337, **333**, 0 clears. Life 10:
    **-157 on 1-2, -4 on 2-1**. It cleared two starts CONTROL never did (85, 90) and missed one CONTROL cleared (80).
    Read: 9 vs 7 clears and 6 vs 5 starts on 8 starts is inside the noise, and the life-10 medians are below CONTROL.
    Not a win; not run as a gauntlet. Same early shape as the blame arm: lives 2-4 mean 1,436 vs 1,365, then down.
    (Contrast: Agrippa's no-blame KEEP was clearly worse, 1 clear, life-10 861. With x keys, Cadence's own -1 on the
    dying decision no longer smears over the whole level, so it stops hurting; it just doesn't help.)
  - Not done: the AgrippaX gauntlet (only on a win, per the brief).
- AgrippaX build notes (for whoever picks this up): `mem.xblock(ram)` = 8.0 * exp(-((x - 32k) / 32)^2) for k = 0..111
  (XN, XSTEP, XWIDTH, XAMP in mem.py). Spec `SPECS["AgrippaX"]` has `xinputs=112`. Brain: brains0/AgrippaX_ck0.npz
  (built by p0_xkey.py: transplant + parity + key cosine). `MemPlayer.xkey` is set when the sensory width is > 2000, and
  `play_attempt` then appends xblock to every observation (also the final one passed to `end`). `keep_obs` stores only
  the 2,000 screen/body values. Height 8 was chosen so position dominates the separated key (the screen part has a norm
  of roughly 15-20; with fan-in 32 most cells read 0-2 x inputs, and those win the top-24). Not tuned further.
  gauntletm.py did NOT append the x block (it calls player.act directly); fixed 14:40 (o and the end observation get
  mem.xblock when player.xkey). evalm.py untested with it.
### AgrippaX training arm (Sean 15:10: "the exact same pipeline as the other brains"; written 15:10, BEFORE launch)
Same mtrain.py recipe and args as Cato / Cicero / Livy (--workers 2 --K 3, chunk 250, DAgger every 6), init
brains0/AgrippaX_ck0.npz, out runs/night2/AgrippaX. Sean 15:15: same wall-clock budget as Cato / Cicero (13:58 -> 05:45 =
15 h 47 min), so its --stop-at is its launch time + 15:47, not 05:45.
- Catch: stored lesson rows (W1's piles, runs/seq coach paths, DAgger files) hold only the 2,000 screen values; AgrippaX's
  input is 2,112 wide. Piles have no RAM and no order, so x can't be recovered for them (rules out a and b for half of
  every batch); re-collecting would cover only the ordered rows.
- **Choice: (c) zero x block on lesson and burn-in inputs; play (SEL, HELD, DAgger) gets the real x** (mem.play_attempt).
  mtrain.py change: when the brain is wider than 2,000 inputs, `b.act` / `b.lesson` zero-pad 2,000-wide inputs. No-op for
  the other three (they are 2,000 wide; their running processes have the old file loaded anyway).
- Why it is faithful, checked in source and by test (scratch xcheck.py, 15:05):
  1. Place records are never written by this recipe. Cadence 0.91.3 writes the place store from a lesson only at
     memory_demonstrations (0 here, generic.py _teach ~3611), and from an act only when an outcome arrives via learn();
     mtrain never calls learn. Test: place vote max |v| = 0.0 before and after 6 burn-in + lesson batches (zero or real x).
  2. With a zero x block, the x->cortex synapses (28,672, efficacy 0 at ck0) stay exactly 0.0 (efficacy and velocity),
     because every EP step on them is pre x post with pre = 0. With a real x block they grow (0.0046 after 6 batches):
     the cortex would start reading x, i.e. the RAM-cheat experiment, not "where am I only, decisions from the screen".
  So under (c) the lessons see what they would see with real x in every way that matters (the cortex never sees x, the
  store is empty either way), and only play sees x, where the key is built.
- **Consequence, say it plainly:** under mtrain's recipe AgrippaX's place memory is inert all night (empty store, zero
  vote), so its training should track Cato's: same weights in the cortex, plus an x key that is empty. It is the
  pipeline Sean asked for, but it will NOT test "weights trained with the vote live" (Next night in the vault plan);
  that needs a recipe change (memory_demonstrations > 0, or learn() on DAgger/retry lives). The same holds for Livy's
  place part (memory_choice with memory_demonstrations=0): its place/context records are also never written in
  training. End-of-night: AgrippaX's checkpoints are where the retry curves (store live at play) get read.
- Parity check: AgrippaX's chunk 0 line (mtrain's own SEL eval path, real x at play) must read 1-1 20/24 m2970, 1-2 7/24
  m1730, 1-3 9/24 m1812, 1-4 11/24 m2159, as Cato / Cicero / Livy did. **PASS (15:16): identical on all four levels; HELD
  50 / 11 / 6 / 25 = W1_F2ck110.**
- Note for the new owner (14:46): the first night2 DAgger round took 1,110 s for 1-1 alone (8 attempts, 2 workers;
  coach labelling is the slow part). Four levels means ~70 min of DAgger every 6 chunks per arm. Consider fewer episodes
  or less frequent rounds if the arms look starved of lessons.

### No-progress note (17:45, session check-in)
Cicero: no new SEL best since 14:32 (3h+), the plan's no-progress rule. Noted only, not killed (Sean: hands-on, keep it running). Cato joint best moved to ck0020 at ~17:45; its gauntlet is deferred until after 01:00 (Sean on Minecraft, no extra CPU side jobs before then). Livy ck0018 1-4 HELD 50/50.

### Cicero stopped itself (18:14)
mtrain.py's built-in --stall-hours 3.0 stopped Cicero at chunk 24 (no new SEL best since 14:32). Not a crash. The other arms run the same rule, so any of them can stop early the same way. Cicero trained ~4h16m vs the others' planned 15h47m: say so on every comparison. Asked Sean whether to resume it.

### Throttle note (22:46)
Game detected from ~21:45 (Sean on HOI4/Minecraft): all arms throttled to 2 shared worker slots, 1 thread. Compute per hour is much lower from here, roughly equal across arms. The 3 h stall clock keeps running in wall time, so arms may auto-stop during throttle; resume them like Cicero (r2 pattern).

### Slot starvation (23:05)
Throttle has a fairness hole: an arm releases its slot at the end of a chunk and grabs it again at once, while a waiting arm polls every 30 s. Cicero_r2 waited for a slot from 22:02 (no progress for 1h+) while Livy and AgrippaX held both through long 1-thread DAgger rounds. No game running at 23:02 (gamecheck NONE), so it self-heals: each arm rechecks the game at its next chunk, turns throttle OFF and stops taking slots. Not intervening. Cicero loses roughly 1-1.5 h more of compute: note on comparisons. Fix for next time: a FIFO ticket queue for slots, and recheck the game inside slot_acquire.

### Old RESUME (11/01 15:15 to 11/02 03:47), kept for the record
- Phase 0: transplant exact; act mode settles cold (= W1 predict); memory synapses start at 0, so every memory brain
  starts as W1_F2ck110. Burn-in K=3. Blame: Cadence's write reaches the decisive move in 3 of 10 deaths -> OUR blame
  window (5..30 back) in the retry curves.
- RUNNING: night arms Cato / Cicero / Livy (runs/night2, mtrain.py, stop 05:45). Livy restarted 14:20 (echo synapses at
  1/8 step; stop still 05:45, so Livy gets 22 min less than Cato / Cicero's 15 h 47 min, not changed).
  AgrippaX (4th arm, same mtrain recipe, runs/night2/AgrippaX, log.txt) started LATE at 15:10, --stop-at 06:57 (= 15 h
  47 min, same wall-clock budget as Cato / Cicero, Sean 15:15). Lesson rows get a zero x block, play gets real x; its
  place store stays empty in training (see "AgrippaX training arm"), so expect it to track Cato. Its end-of-night
  scoring runs AFTER 06:57, later than the others: top-3 SEL checkpoints on HELD, World 2 first sight, ablation with the
  place vote off (x block / memory_amplitude 0), retry curves vs reload control. Check-ins: read its log.txt too.
  Per-level bests so far: Cicero 1-4 HELD 44/50 (ck5), Cato 1-2 HELD 9/50 (ck5). Joint picks still ck0.
- Place memory retry test (bar: +300 median max x at life 10 vs reload control):
  - Agrippa LOST (life 10 +162 on 1-2, +127 on 2-1). Sparse keys and no-blame variants worse.
  - AgrippaX (Sean's idea, x from RAM in the place key; "where am I only, decisions still from the screen") 14:47:
    keys now local (cosine 0.80 / 0.29 / 0.11 / 0.07 at 1 / 5 / 20 / 60 decisions vs Agrippa 0.69 / 0.58 / 0.56 / 0.52),
    cortex parity exact, weights byte-identical after retry lives. Retry: **not a win**, life 10 -20 on 1-2, +100 on 2-1
    (no-blame: -157, -4). Hint only: lives 2-4 on 1-2 a bit above control, then fades. Texted Sean 14:49.
    Read: locality was not the bottleneck; a -1 on a past button never says what to do instead.
- Gauntlets: m01 Agrippa (remembers across lives) 1-1 then dies on 1-2 x5; m02-m04 Cato / Cicero / Livy joint = ck0,
  identical to g16 (1-1, then 1-2 x5).
- STILL NOT DONE (end-of-night agent, after 05:45): top-3 SEL checkpoints scored on HELD, play-time ablations (one
  memory part off at a time on Cicero, Livy, Agrippa), end-of-night retry curves (Livy, Agrippa), gauntlets on moved
  joint picks, brain-cam panels, roster cards, retry-curve chart, page update.
- Cicero RESUMED 21:05 (Sean: "Continue Cicero"): runs/night2/Cicero_r2, init = Cicero leg-1 ck0024 (r2 ck0000 = leg-1 ck0024), --stall-hours 99, --more-seq loads leg-1's DAgger attempts (ordered rows 4816/3811/2777/3587 = leg-1's), stop 05:45. Leg 1 ran 13:58-18:14 (auto-stopped by the 3 h rule); total ~12h55m vs Cato's 15h47m. End-of-night: score Cicero over BOTH dirs (leg-1 bests + r2 bests). If another arm auto-stops, resume it the same way (Sean wants equal time).
- END-OF-NIGHT NOTE (roadmap review 11/1 23:40): AgrippaX trains as a near-clone of Cato (empty store, zero x in lessons, same seed), so report AgrippaX vs Cato as the NOISE FLOOR (A/A spread; by chunk 47 SEL-best HELD totals were Cato 154 vs AgrippaX 113). Memory arms count only above that gap plus a 10+ clear ablation. Single-checkpoint records (e.g. Cato 1-3 46/50) are best-of-many picks: label them so. Note unequal compute (Cicero ~12h55m + slot starvation; Livy ~3x cost per lesson).
- m06 (01:03): Cicero (thought + button memory) r2 ck0001 beat all of World 1 in one 5-life gauntlet (deaths 1-2 x1,344, 1-3 x936; 2-1 x471/722/330). Second brain ever, first memory brain. A demo (one run of one checkpoint), not evidence that memory helps: page it as such, with the clip runs/gauntlet_night/m06_Cicero.mp4.
- Cato auto-stopped 02:44 (3 h rule, chunk 78); RESUMED 02:46 as runs/night2/Cato_r2 from ck0078 (r2 ck0000 = leg-1 ck0078), stall off, leg-1 DAgger rows carried, stop 05:45. Score Cato over both dirs. Lost ~2 min.
- AgrippaX auto-stopped 02:46 (3 h rule, chunk 66); RESUMED ~02:50 as runs/night2/AgrippaX_r2 from ck0066, stall off, leg-1 DAgger rows carried, stop 06:57 (equal duration kept). Score over both dirs; scoring still after 06:57 (cron 07:07).
- Livy auto-stopped 03:23 (3 h rule, chunk 73); RESUMED 03:47 as runs/night2/Livy_r2 from ck0073, stall off, DAgger rows carried, stop 05:45. Lost ~23 min (on top of its 22 min late start). Score over both dirs.

## End of night (11/02, 05:51-07:30)
Training stopped: Cato_r2 05:49, Livy_r2 05:47, Cicero_r2 06:02 (its last chunk began with a DAgger round before
05:45; the deadline check runs after a chunk, same rule as the others), AgrippaX_r2 06:57.

**Picks (written before any HELD score, eon_select.py -> runs/eon/picks.json):** per arm, top 3 checkpoints by mtrain's
joint SEL key [min flags, sum flags, sum mean], both legs pooled. Chunk 0 left out: leg-1 ck0 is W1_F2ck110 in every
arm (it was in every arm's top 3, which would compare the start with itself), scored once as "Start"; r2 ck0 = leg-1
last. Candidates: Cato 96, Cicero 71, Livy 85, AgrippaX 93. First pick = the arm's chosen checkpoint (ablations,
gauntlet), fixed by SEL rank, not by HELD.

**HELD (50 starts per level, greedy act mode, evalm.py; runs/eon/held_*.json, eon_table.py):**
| brain | 1-1 | 1-2 | 1-3 | 1-4 | W1 /200 | 2-1 mean x | 2-1 max | 2-1 past x 723 | 2-2 mean / max |
|---|---|---|---|---|---|---|---|---|---|
| Start (W1_F2ck110) | 50 | 11 | 6 | 25 | 92 | 604 | 735 | 5/50 (all x 735 deaths at the pipe) | 162 / 163 |
| Cato L1 ck20 | 33 | 10 | 46 | 30 | 119 | 664 | 722 | 0 | 163 / 163 |
| Cato L1 ck68 | 18 | 11 | 11 | 30 | 70 | 722 | 723 | 0 | 138 / 162 |
| Cato L1 ck8 | 6 | 1 | 12 | 14 | 33 | 626 | 722 | 0 | 142 / 163 |
| AgrippaX L1 ck18 | 29 | 14 | 4 | 38 | 85 | 544 | 723 | 0 | 163 / 163 |
| AgrippaX L1 ck49 | 13 | 7 | 27 | 17 | 64 | 678 | 735 | 5/50 (x 735) | 163 / 163 |
| AgrippaX L1 ck34 | 24 | 12 | 3 | 21 | 60 | 545 | 722 | 0 | 162 / 163 |
| Cicero r2 ck1 | 19 | 11 | 13 | 45 | 88 | 479 | 722 | 0 | 162 / 163 |
| Cicero L1 ck1 | 38 | 12 | 20 | 27 | 97 | 515 | 723 | 0 | 162 / 163 |
| Cicero L1 ck6 | 17 | 17 | 3 | 24 | 61 | 578 | 1,021 | 7/50 (x 914-1,021) | 151 / 163 |
| Livy L1 ck57 | 5 | 18 | 3 | 17 | 43 | 530 | 532 | 0 | 163 / 163 |
| Livy L1 ck27 | 37 | 9 | 1 | 18 | 65 | 464 | 465 | 0 | 163 / 163 |
| Livy L1 ck11 | 21 | 7 | 4 | 24 | 56 | 722 | 723 | 0 | 211 / 514 |
Top-3 means: Cato 74, AgrippaX 70, Cicero 82, Livy 55; 2-1 mean x (avg of 3): 671, 589, 524, 572. No World 2 clears.
Every arm's average pick is BELOW the start (92): training from a SEL-picked peak regresses, as the review said.
Cato ck20 1-3 46/50 is a new 1-3 high (beats U3ck83 45), but it is the best of 12 HELD-scored picks from ~350 saved
checkpoints: best of many, labeled so on the page.

**Noise floor (A/A):** Cato vs AgrippaX, same recipe (empty store, zero x into the cortex). Top-3 mean gap 4; best-pick
gap 34; within-arm range 33-119 (Cato) and 60-85 (AgrippaX). Pooled checkpoint SD ~32 -> SE of a 3-vs-3 mean difference
~26. 2-1 mean gap 82 x (bar was 150). Memory arms: Cicero +8, Livy -19 vs Cato: inside the floor either way.

**Ablations (first picks, HELD W1; runs/eon/abl_*.json):**
| brain | on | thought off | button off | echo off | place off | all off |
|---|---|---|---|---|---|---|
| Cicero r2 ck1 | 19/11/13/45 = 88 | 28/7/24/45 = 104 (+16) | 20/8/13/45 = 86 (-2) | - | - | 26/11/24/45 = 106 (+18) |
| Livy ck57 | 5/18/3/17 = 43 | 0/2/6/29 = 37 (-6) | 6/20/3/27 = 56 (+13) | 8/1/0/15 = 24 (-19) | identical (43) | 0/11/1/3 = 15 (-28) |
| AgrippaX ck18 | 29/14/4/38 = 85 | - | - | - | identical | x zeroed: identical; both: identical |
Read: Cicero's memory costs it clears (mostly 1-1 and 1-3). Livy's weights depend on thought memory and echo (1-2
falls 18 -> 2 or 1 without them), but the whole brain is still the worst arm. Place off changes nothing in greedy play
for Livy or AgrippaX: greedy acts write no place records, so the vote is zero (expected, a parity check).

**Retry curves, Livy ck57 (retry.py, phase-0 protocol: 8 held starts, 10 lives, temp 0.05, our blame window 5..30,
keep vs reload control; runs/eon/retry_Livy.json):**
| level | arm | median max x, lives 1-10 | clears per life (total) | starts cleared any |
|---|---|---|---|---|
| 1-2 | KEEP | 1646, 863, 870, 861, 879, 851, 69, 817, 446, **904** | 2,1,1,1,1,2,0,0,2,0 (10) | 8 of 8 |
| 1-2 | CONTROL | 1646, 1531, 1670, 1198, 1392, 859, 929, 1049, 67, **1320** | 2,3,3,1,3,1,2,1,0,2 (18) | 8 of 8 |
| 2-1 | KEEP | 322, 731, 417, 465, 436, 398, 626, 461, 450, **501** | 0 | 0 |
| 2-1 | CONTROL | 322, 399, 530, 348, 401, 530, 461, 498, 530, **530** | 0 | 0 |
Life 10: -416 on 1-2, -29 on 2-1 (bar +300). No level cleared by KEEP that CONTROL never cleared. LOST. KEEP is below
CONTROL from life 2 on 1-2, the same shape as Agrippa's.
AgrippaX retry: SKIPPED. Its place memory lost the same retry test on 11/1 (frozen W1 weights), and tonight's arm
never wrote its store, so its weights are Cato-like with an empty notebook: a retry run would repeat the 11/1 test on
different weights with no reason to expect a different answer (Sean's "it never learned to use it" needs the
live-store recipe, now folded into roadmap step 5).

**Gauntlets (demos, sampled play, see correction):** m05 Cato ck20 (23:47) 1-1, 1-2; m06 Cicero r2 ck1 (01:03) all of
World 1; m07 Livy ck57 (06:44) 1-1 (third try), then 1-2 x 1,743, 867; m08 AgrippaX ck18 (07:05) 1-1, then 1-2 873,
873, 850, 871, 1,595. Cato's and Cicero's chosen checkpoints are m05's and m06's brains, so no rerun.

**Correction: gauntletm --mode act is sampled, not greedy.** MemPlayer handles 'greedy' and 'retry'; any other mode
(incl. 'act', used for m05-m08 per the brief) falls into the sampled branch (temperature 0.05) and calls learn() every
decision without the frozen actor rates. runs/_actcheck.py: weights byte-identical after two attempts (learn() with
reward 0 mid-attempt and no final learn changes nothing hashed), mode act reproduces m06's log exactly (1-1 flag in
306 decisions, 1-2 death at x 1,344), greedy does not (312 decisions; 1-2 dies at x 921). So m06 is one sampled draw.
Labeled on the page; HELD/SEL scores were always greedy.

**Brain-cam:** record_brain.py --act --parts (plays with mem.MemPlayer greedy cold, records each act's settle during
play; act-mode recorded choice = pressed button 100% on 1-2/38, 99.7% on 1-1/0) and brain_video.py panels for thought
memory (prefrontal), button memory (efference) and echo (history eye block). Cicero r2 ck1: 1-2/38 clear, 1-1/0 clear.
No place panel: greedy play has an empty store.

**Code changed tonight:** mem.py (ablation part "x": x block zeroed via MemPlayer.xamp; retune skipped when empty),
eon_select.py, eon_table.py, runs/_actcheck.py, viz/record_brain.py, viz/brain_video.py.
