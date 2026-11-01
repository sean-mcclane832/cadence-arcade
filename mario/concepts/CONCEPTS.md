<!-- RESUME -->
## RESUME
(filled in at the end of the run)
<!-- /RESUME -->

# Concepts, goals, learning from dying (2026-10-31 night)

Lab notebook for the plan in the vault (Plans/Mario Concepts Goals Dying 2026-10-31). Same habits as
../LEARNING_FIX.md: hypothesis + falsifier BEFORE each run, then the result, corrections in place.
Code here: cplay.py (play/label, RAM concept map), csteady.py (training driver), evalset.py (scores),
critpath.py (own-path critical error), gauntlet.py (5-life gauntlet + clip), collect.py (coach demos).
Python: ../../.venv-concepts (Cadence 0.91.3). Start sets as in LEARNING_FIX: SEL 24 (selection only),
HELD 50 (TEST 32-41 + 80-99 + 100-138 even; never trained or selected on).

**Anything marked RAM is a CHEAT (ceiling test): the brain reads a map built from game memory.**

## Phase 0. Control on Cadence 0.91.3 (17:05)

### 0.91.2 / 0.91.3 changelog
Both releases are about the recall memory: 0.91.2 binds reused recall products to the bank within one
operation (bit-identical answers, faster), 0.91.3 makes several recall heads compete for slots
(`recall_compete`, only matters with 2+ heads). Our Mario brains are one region of 256 with no recall
memory, and every lesson/predict uses `memory=False`. So nothing here should change, and nothing about
how recall is configured needs to change for this setup. If a later phase uses recall heads (more
than one), note that 0.91.2 and earlier cannot load a several-head file saved by 0.91.3.

### Do 0.91.1 brains act the same under 0.91.3? (equiv.py, 17:10)
3,000 stored screens (750 each from the 1-1, 1-2, 1-3, 1-4 piles), predict() in both venvs, six brains
(U3ck83, one_11_13, 1-1_U3ck3, 1-2_12n59, U3ck28, st_U3 ck_0044): **0 of 18,000 answers differ.**
The saved brains load and act identically.

## Phase 1. RAM concept map, ceiling test (CHEAT) (pre-registered 17:20)
The map (cplay.concept_map, checked by printing it at known spots on 1-1 and 1-2, runs/mapshow.png):
13 rows x 16 columns of 16 px tiles around Mario (4 behind, 11 ahead, the 13 block rows of the screen),
channels solid / enemy / pit / powerup / mario / empty, plus last decision's enemy channel (motion).
From RAM: the block buffer 0x500-0x69F, the 5 enemy slots, the powerup slot, Mario's position.

**Pre-registered choice: the RAM brain reads body + map, NO screen** (336 + 1,456 = 1,792 inputs). That is
the clean ceiling question: if the only world it sees is "what is where", does it play better and carry
over to World 2? With the screen in, a gain or loss could come from either.

Setup (csteady.py): FRESH Cadence brain (one region of 256, normalized teaching, no surprise, eta 0.001,
critical rows x3 in sampling), coach demonstrations on 1-1/1-2/1-3/1-4 (collect.py, 11 per level, the
dagger.py round-0 recipe, stored with every sense so arms can share them), lesson shares 1-1 0.2, 1-2 0.4,
1-3 0.2, 1-4 0.2, tie-aware DAgger on all four every 6 chunks (12 attempts each, 2/3 from the start),
every checkpoint SEL-tested on all four, joint pick as before.

Matched control (added on purpose): the showcase brains had ~10 h each and started from earlier brains,
so they are not a fair comparison for a fresh 3 h brain. Arm B is the SAME recipe, same coach rows,
fresh brain, reading the normal senses (eyes + body, 2,000). Arms A (ram) and B (body) split the PC.

- H1: the concept map helps. Prediction: arm A beats arm B on 1-2 HELD and on 2-1 first-sight max x.
- Win (plan): 1-2 HELD >= 20/50 at A's best 1-2 or joint checkpoint, or 2-1 first-sight max x > 1,000
  (first sight = HELD starts on 2-1 / 2-2, measured ONLY on A's joint best and A's best 1-2 checkpoint, never
  used for picking).
- Lose (plan): no better than the control on both: 1-2 HELD <= the control's AND 2-1 max x <= the control's.
  Controls: showcase (1-2_12n59 HELD, U3ck83 2-1 max 723) and arm B. If A beats B but not the showcase,
  I'll say exactly that.
- 3 h no-progress rule; budget ends ~21:00.

### Phase 0 result: showcase brains re-scored on 0.91.3 (evalset.py, runs/p0/held_0913.json, 17:18)
HELD = solo clears of 50 held-out starts; SEL in brackets (24 selection starts). Every 0.91.1 number
reproduces exactly (deterministic emulator + identical answers).
| brain | 1-1 | 1-2 | 1-3 | 1-4 |
|---|---|---|---|---|
| U3ck83 (st_U3 ck_0083) | 15 (17) | **0 (0)**, max x 896 | 45 (23) | 33 (20) |
| U3ck44 (st_U3 ck_0044) | 25 (13) | **0 (0)**, max x 900 | 15 (15) | 37 (14) |
| one_11_13 | 40 (15) | 0 (0) | 42 (19) | 0 (0) |
| 1-1_U3ck3 | **46** (23) | 0 (0) | 4 (0) | 8 (10) |
| 1-2_12n59 | 14 (7) | **29** (13) | 0 (0) | 0 (0) |
| U3ck28 | 30 (14) | 0 (0) | 11 (7) | **50** (24) |
**1-2 test of the three-level brains: U3ck83 and ck_0044 flag 1-2 on 0 of 74 starts** (HELD + SEL; both stop
around x 900, the Koopa/brick-column section). So no brain has cleared every World 1 level; the headline stays
"1-1, 1-3 and 1-4 with the same weights". Side findings: 1-1_U3ck3 also clears 1-3 (4/50) and 1-4 (8/50), so it
is a three-level brain too, just lopsided; 1-2_12n59 clears 1-1 14/50, the only brain that does 1-2 plus anything.

### Gauntlet rule (Sean, updated during phase 0)
5 lives, a death restarts the current level, flag -> next level, game over on the 5th death. The emulator is
deterministic, so an identical restart replays the identical death (the first try at it, run with 0 idle
frames every life, died 5 times at exactly x 322 on 1-2: kept as runs/gauntlet/old/). Retries now start
after 37 / 83 / 110 / 131 idle frames (all held-out starts), the first try of each level after 0.

| # | time | brain | type | cleared | deaths (level: x) | lives left | clip |
|---|---|---|---|---|---|---|---|
| g01 | 17:11 | U3ck83 | one brain | 1-1 | 1-2: 322 (stuck), 470, 322 (stuck), 893, 354 (stuck) | 0 | runs/gauntlet/g01_U3ck83.mp4 |
| g02 | 17:48 | U3ck44 | one brain | 1-1 | 1-1: 2762, 1784; 1-2: 323, 354, 354 (all stuck) | 0 | runs/gauntlet/g02_U3ck44.mp4 |
| g03 | 17:48 | 1-1_U3ck3 / 1-2_12n59 / U3ck83 / U3ck28, U3ck83 after | **relay of specialists** | **1-1, 1-2, 1-3, 1-4** | 1-2: 1595, 852, 1336; 2-1: 531, 723 (stuck) | 0 | runs/gauntlet/g03_relay.mp4 |
| g04 | 19:20 | A_ram ck_0013 (phase 1 joint best, 1.5 h) | one brain, **RAM CHEAT** | none | 1-1: 2850 (stuck), 1820, 1820, 2850 (stuck), 1820 | 0 | runs/gauntlet/g04_Aram.mp4 |
| g05 | 19:20 | A_body ck_0016 (phase 1 control joint best) | one brain | none | 1-1: 1126 x5 | 0 | runs/gauntlet/g05_Abody.mp4 |

Phase 1 run notes: coach demos (collect.py, 17:19-17:44, 26 min on 8 cores) reproduce the earlier round-0
row counts exactly (1-1 2,838, 1-2 2,519, 1-3 1,926; 1-4 1,619), coach flags 8/6/6/6 of 11. Arms launched
17:47 (runs/p1/A_ram, runs/p1/A_body, 4 workers each). Minecraft (javaw) was open; the coordinator relayed
Sean's "use it all up" (he is out), so the javaw throttle is off for this run (--ignore-games), logged here.

Phase 3 API check (done early): Cadence's `Brain.compose(..., slots=(7, 2, 2, ...))` gives one brain several
motor groups, "each read as its own softmax", in the same settle. So a second readout of the SAME brain
exists: slot 0 = buttons, further slots = concept questions (enemy close ahead? gap ahead? on the ground?).
No separate module needed. Only used if phase 1 wins.
(U3ck83 is not re-run every 90 min: the gauntlet is deterministic for a fixed brain, so g01 IS its result
until the brain changes. Same for the relay.)

Phase 1 at 1.5 h (19:20): both fresh brains 0 HELD flags anywhere. Body arm: first 1-2 SEL flag (ck_0016,
HELD 0), 1-1 HELD 2/50 at ck_0015. RAM arm: 0 SEL flags but goes further on 1-1 in the gauntlet (x 2,850 vs
1,126). DAgger rounds are the slow part (~35-40 min for four levels on 4 workers vs ~1 min per chunk).
Side fix: critpath.py did two plan searches per decision and would have taken ~3 h on 2 workers; now one
search per decision (same label and weight), 8 held-out starts, to be run on the full PC after phase 1.

### Phase 1 result (stopped 21:16 after 3.5 h; runs/p1/score.json, HELD = 50 held-out starts)
| checkpoint (picked on SEL) | 1-1 | 1-2 | 1-3 | 1-4 | 2-1 first sight max x (mean) | 2-2 max x |
|---|---|---|---|---|---|---|
| A ram (CHEAT), joint best ck_0034 (pre-registered) | 10 | **0** | 0 | 36 | **429** (428) | 162 |
| A ram (CHEAT), best 1-2 ck_0032 (pre-registered) | 0 | **0**, max 1,330 | 0 | 20 | **474** (472) | 162 |
| A ram (CHEAT), best 1-4 ck_0028 (not pre-registered) | 0 | 0 | 0 | **49** | 1,021 (962) | 162 |
| B body (control), joint best ck_0020 | 2 | 0 | 0 | 0 | 340 | 162 |
| B body (control), best 1-1 ck_0029 | 0 | 0 | 0 | 0 | 530 (472) | 163 |
| showcase control (10+ h each) | 46 | 29 | 45 | 50 | 723 (U3ck83) | 514 |
**Verdict as pre-registered: phase 1 does not win.** 1-2 HELD 0/50 (needed 20), 2-1 first sight 429 / 474
on the two pre-registered checkpoints (needed > 1,000, and below the showcase's 723). By the lose rule
("no better than control on both 1-2 and 2-1") it **loses**: on 1-2 it ties arm B at 0, on 2-1 the
pre-registered checkpoints are no better than the showcase control.
What it did show, clearly: **learning speed**. Same recipe, same coach rows, same 3.5 h, same PC share:
the RAM brain learned 1-4 to 49/50 and a 1-1 + 1-4 brain (10 + 36 of 50), while the screen brain got
1-1 2/50 and nothing else. With perfect "what is where", imitation learns World 1 levels several times
faster. It did not make 1-2 or first sight come in 3.5 h. One exploratory number, flagged because I did not
pre-register it and it is the best of five brains scored: the 1-4 specialist ck_0028 reaches x ~960-1,021 on
2-1 on every held-out start (the screen brains: 340-723). Not a win; worth a look on a later night.
Phase 3 (learned concepts) is skipped: it was conditional on a phase 1 win.

## Phase 2. Death clinic + keep-only-survivals (pre-registered 21:20)
Best input from phase 1 for 1-2: phase 1 lost on its 1-2 / first-sight test, so the screen input stays
(and that keeps the result honest for the showcase, no RAM). Starting brain: 1-2_12n59 (1-2 HELD 29/50), the
brain line where the 27% own-path critical error was measured. A/B, same everything else:
- arm C (clinic): every DAgger round = 6 normal attempts + the death clinic: up to 4 of the current
  checkpoint's failed SEL attempts on 1-2, each rewound to 1 / 2 / 4 s (15 / 30 / 60 decisions) before the
  end, the brain plays on, the coach labels (12 attempts) + survivals: its own buttons from those retries
  when it gets >= 16 px past where it died, weight 0.5, capped at 20% of each lesson.
- arm N (no clinic): 18 normal attempts per round (same coach budget).
Both: rows = st_12n's 1-2 pile (52,445) + 1-1 replay (fix_cad_body3), lessons 90% 1-2 / 10% 1-1, crit x3,
eta 0.001, DAgger on 1-2 every 10 chunks, tie-aware, SEL on 1-1 and 1-2 every chunk, 4 workers each, 3 h.
- Metric (critpath.py): 1-2 own-path critical error on 8 held-out starts, coach scoring every decision of
  the brain's own solo path. Measured on 1-2_12n59 (start) and each arm's best 1-2 checkpoint at the end.
- Win: C's best 1-2 checkpoint < 15% own-path critical error (from 27%). Lose: still >= 25%.
- Secondary: C beats N on own-path critical error and on 1-2 HELD.
