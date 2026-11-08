# Fabius tiny tests (2026-11-08)

Before the Fabius run: does the brain still learn its buttons with the recall memory and 104 prediction slots?
`tiny.py`, lessons as ctrain with `--p-seq 1.0` (ordered coach rows, burn-in K 8 = recall capacity 8), trained on
the smoke piles of 1-1 and 4-1 (570 rows, one smart-coach run each), scored on 160 of those rows ("seen") and on 6-4
("held", 198 rows, never taught). Button agreement with the coach, greedy act. Files in `runs/tiny_*.json`.

## Result at 150 lessons (Cato2: 300)

| arm | brain | mask | seen | seen, critical rows | held | slots seen / held (always "same": 0.869 / 0.868) |
|---|---|---|---|---|---|---|
| fabius (file `tiny_fabius_0_brains_k8`) | Fabius | none in effect, see below | 0.706 | 0.333 | 0.783 | 0.869 / 0.865 |
| fabius-nofix | Fabius | none | 0.706 | 0.333 | 0.783 | 0.869 / 0.865 |
| slots | slots, no recall | none | 0.706 | 0.333 | 0.783 | 0.869 / 0.868 |
| recall | recall, 7 buttons | none | 0.931 | 0.958 | 0.439 | - |
| fabiusbtn | Fabius | buttons x105 | 0.925 | 0.917 | 0.348 | 0.869 / 0.856 |
| fabiusall | Fabius | all outputs x105 | 0.750 | 0.625 | 0.763 | 0.889 / 0.760 |
| fabiusmix4 | Fabius | buttons x105, slots x4 | 0.919 | 0.917 | 0.364 | 0.871 / 0.817 |
| **fabiusmix10** | Fabius | buttons x105, slots x10 | **0.925** | **0.958** | **0.525** | 0.874 / 0.798 |
| cato2 seed 0 / 1 (300 lessons) | Cato2 | - | 0.919 / 0.912 | 0.958 / 0.958 | 0.621 / 0.682 | - |

Majority button: 0.706 seen, 0.783 held. 0.706 / 0.783 means the brain presses the most common button everywhere.

## What it showed
- **The prediction slots stop button learning, not the recall memory.** Slots-only collapses to the majority button,
  recall-only learns. Cause: with 105 slots each slot's nudge is beta / 105 = 0.00095, below the settle tolerance
  (0.003), so the button contrast is noise.
- **The first "fabius" run had no mask in effect.** The mask was set once after loading, but the learner is rebuilt at
  the first lesson after a config change, which drops it. That is why `tiny_fabius_0_brains_k8` and
  `tiny_fabius-nofix_0_brains_k8` are identical to the last digit. tiny.py and ctrain.py now set the mask before every lesson.
- **mix10 was picked**: buttons as good as Cato2's on seen rows, best on critical rows, best held of the masked arms.
  The slots are barely above always-"same" on seen rows and below it on held: whether prediction learns at all is
  open until the real run (150 lessons on 570 rows is tiny).
- **Held-out 6-4 falls for every arm that learns**, Cato2 included (its held agreement peaks at 0.848 at lesson 25,
  then 0.566 at 200, 0.621 at 300): overfitting to two coach runs. The held column compares arms, it does not say
  what a full pile does.
- Cost (machine shared with the pile rebuild): seconds per lesson Cato2 ~0.24-0.39, Fabius K8 ~0.37-0.6, Fabius K16 ~0.87.
  K 8 kept.
