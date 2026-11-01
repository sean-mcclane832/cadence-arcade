<!-- RESUME -->
## RESUME (11/01 13:40 EST)
- Phase 0: transplant exact (0/3,000); act mode needed cold settles to match W1 (fixed); memory synapses start at 0, so
  every memory brain starts as W1_F2ck110. Burn-in K=3 (K=1 already 0.9995). Blame: Cadence's write reaches the decisive
  move in 3 of 10 deaths -> OUR blame window (5..30 back) in the retry curves. Place keys overlap ~0.5 even 20 tiles apart.
- RUNNING: night arms Cato / Cicero / Livy (runs/night, stop 05:45). Agrippa retry curves (runs/p0_retry_*.json).
- Next: retry-curve answer -> text Sean; gauntlets every ~90 min.
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
