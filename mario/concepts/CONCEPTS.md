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

Phase 1 run notes: coach demos (collect.py, 17:19-17:44, 26 min on 8 cores) reproduce the earlier round-0
row counts exactly (1-1 2,838, 1-2 2,519, 1-3 1,926; 1-4 1,619), coach flags 8/6/6/6 of 11. Arms launched
17:47 (runs/p1/A_ram, runs/p1/A_body, 4 workers each). Minecraft (javaw) was open; the coordinator relayed
Sean's "use it all up" (he is out), so the javaw throttle is off for this run (--ignore-games), logged here.

Phase 3 API check (done early): Cadence's `Brain.compose(..., slots=(7, 2, 2, ...))` gives one brain several
motor groups, "each read as its own softmax", in the same settle. So a second readout of the SAME brain
exists: slot 0 = buttons, further slots = concept questions (enemy close ahead? gap ahead? on the ground?).
No separate module needed. Only used if phase 1 wins.
