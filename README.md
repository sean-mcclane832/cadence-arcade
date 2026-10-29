# cadence-arcade

**Ember Field**, an original catching game for a continuing [Cadence](https://github.com/muellerberndt/cadence)
brain, and a replay page that shows one brain learning it.

- `game/field.py`: the game. 9 x 7 field, the lantern walks the bottom row (left / stay / right),
  amber and blue embers fall. Food colour +1, sting colour -1, miss 0. The colours swap at each
  season turn (15,000 and 30,000 moments into a 45,000-moment life). The player sees 16 numbers:
  how close the lowest ember of each colour is in the 7 columns around it, plus walls.
- `game/agents.py`: the players. A Cadence brain through `Brain.live`
  (`modules=(32,), memory_choice=True, arousal_need_scale=2.0`), a plain online MLP Q-learner
  (1 hidden layer of 32, eps 0.1), and random.
- `game/run.py`: lives every player through a whole life, records food / stings per 500 moments and
  six 400-moment clips. `python -m game.run --seeds 10-21 --out runs/main` (3 processes, about
  10 minutes on the desktop).
- `page/build.py` + `page/template.html`: build the single-file replay page.
  `python page/build.py runs/main page/ember-field.html`. The showcase life is the median Cadence
  seed by season-three score.

Settings were picked in a short sweep on seeds 0-2 (scratch scripts not kept); the reported runs use
fresh seeds 10-21. Published page: https://claude.ai/artifact/QG35Mk2iEMEr6eXrNDsUrH (private).

## Results (12 lives each, food / stings per 500 moments, median)

| player | first 2,000 | end season 1 | 2,000 after turn 1 | end season 2 | end season 3 |
| --- | --- | --- | --- | --- | --- |
| Cadence | 12.3 / 6.6 | 15.8 / 3.8 | 5.6 / 8.9 | 11.1 / 4.1 | 10.6 / 3.9 |
| plain MLP | 9.9 / 5.4 | 25.3 / 2.4 | 5.8 / 17.9 | 12.2 / 2.9 | 19.8 / 2.8 |
| random | 8.1 / 8.3 | 8.0 / 8.2 | 8.3 / 8.6 | 8.4 / 8.0 | 7.8 / 7.8 |

The Cadence brain learns from pay alone (2x random food, about half the stings) and relearns after each
turn. The plain MLP is the stronger player overall; it charges into the new sting after a turn (twice
random), and its old season-one skill returns when amber feeds again, where the Cadence brain's does not.

# classic/: One-Frame Breakout (Cadence 0.89.0 recall position term)

An existing game, **MinAtar Breakout** (Young & Tian 2019, GPL-3.0, `pip install minatar`), with
one change: players see one frame of hand features (ball column/row, paddle column, their
difference) and not MinAtar's trail, so the ball's direction is only knowable by remembering a
moment ago. A coach that knows the direction labels every training moment; Cadence learns by
`Brain.lesson` then `act`, because the 0.89.0 position term is trained only by lessons (under
reward-only `live` it never moves; `identity.py` shows identical actions with and without it).

Protocol, arms and predictions are pre-declared in `classic/run.py` (committed before the run).
Published page: https://claude.ai/artifact/81ZMYFEZbsVPrT3HX5P5FJ (private).

## Results (seeds 10-21, bricks per test game over the last 10 of 30 tests, median [IQR])

| player | bricks / game | agrees w/ coach |
| --- | --- | --- |
| Cadence, recall + position term (founder memory) | 5.65 [5.06-7.06] | 84% |
| Cadence, recall, `recall_position=0` | 5.57 [5.45-6.24] | 83% |
| Cadence, no recall | 6.97 [6.82-7.62] | 87% |
| plain MLP, one frame | 5.10 [5.00-5.23] | 84% |
| plain MLP, given two frames (reference) | 6.72 [6.29-6.95] | 89% |
| random | 0.35 | 31% |
| recency off: recall + position term | 5.87 | 89% |
| recency off: recall, `recall_position=0` | 4.20 | 84% |

- Founder memory: position term vs none 6 wins / 6 losses, median +0.42, Wilcoxon p = 0.68. **Null.**
- Recency off (`recall_recency=0`): position term better in 12/12 seeds, +1.52, p = 0.0005; it
  replaces the fixed recency nudge but does not beat it.
- No recall beat recall (11/12 vs plain recall, 8/12 vs position term): the settle already carries the
  previous moment. MLP two frames > one frame 12/12.

## Long run: 10x the training (300 blocks)

Same protocol, `--blocks 300`, arms pos32/pos0/norecall/mlp1/mlp2, seeds 10-21. Page:
https://sean-mcclane832.github.io/cadence-arcade/breakout-long/ (built by `classic/long_page.py`, raw lives
in `classic/runs/long10x`, summary in `classic/runs/long10x/summary.json`).

| player | 30 blocks | 300 blocks, median [IQR] |
| --- | --- | --- |
| Cadence, recall + position term | 5.65 | 13.15 [11.82-13.74] |
| Cadence, recall, `recall_position=0` | 5.57 | 10.38 [7.98-12.33] |
| Cadence, no recall | 6.97 | 13.35 [10.50-14.95] |
| plain MLP, one frame | 5.10 | 6.02 [5.84-6.18] |
| plain MLP, two frames (reference) | 6.72 | 19.20 [17.70-20.38] |

- Every Cadence arm roughly doubled after block 30 and was still rising at 300.
- The one-frame MLP plateaus near 6 (it can't see the ball's direction); every Cadence arm beats it 12/12
  by about 7. The settle carries the previous moment.
- Given two frames, the MLP pulls clear in the long run: beats the position-term brain 12/12 (+6.2) and
  no-recall 11/12 (+5.9). At 30 blocks they were level.
- Position term vs none: +2.97 median, 7/12 seeds, p = 0.09 on seeds 10-21, so 24 more seeds (22-45,
  `runs/long10x-more`) were run. Pooled 36 seeds: 17/36 wins, median -1.02, Wilcoxon p = 0.68. **Null**, as at
  30 blocks; the lean was noise. No recall is as good as recall + position term (6/12, p = 1.0).

```bash
cd classic
$PY run.py --seeds 10-21 --arms pos32,pos0,norecall,mlp1,mlp2 --blocks 300 --procs 6 --out runs/long10x   # ~1.5 h, 6 procs
$PY long_page.py runs/long10x ../docs/breakout-long/index.html
```

## Rerun

```bash
cd P:/projects/cadence-arcade
export UV_CACHE_DIR=P:/projects/.uv-cache UV_LINK_MODE=copy
uv venv --python 3.11 .venv-classic
uv pip install --python .venv-classic/Scripts/python.exe minatar==1.0.15 cadence-net==0.89.0
cd classic
PY=../.venv-classic/Scripts/python.exe
$PY run.py --seeds 10-21 --out runs/main                      # 6 arms x 12 seeds, 3 processes, ~45 min
$PY run.py --seeds 10-21 --arms pos32,pos0 --cadence '{"recall_recency": 0.0}' --out runs/recency0   # ~15 min
$PY identity.py                                               # reward-only check, ~5 min
$PY summarize.py                                              # table + paired tests -> runs/summary.json
$PY build_page.py page/one-frame-breakout.html                # single-file replay page (prose in page_text.json)
```

# mario/: Super Mario Bros. World 1-1, one Cadence brain

One Cadence brain (cadence-net 0.91.1, founder seed 0, one region of 256 neurons) learns to play
World 1-1 of the real NES game from the screen: 26 x 32 colour blocks of the present frame and of the
frame one decision ago, 7 buttons (noop, right, right+jump, run, run+jump, jump, left), 4 frames
per decision. It learns by imitation from a look-ahead coach that tries button plans in a saved
emulator, then by DAgger: the brain plays, its own buttons are executed, and the coach labels every
screen it reached (the recipe in Cadence's [teach guide](https://github.com/muellerberndt/cadence/blob/main/docs/teach.md#teach-on-the-brains-own-path)).
It is one brain, trained once and continued round after round, not an average over seeds. Page:
https://sean-mcclane832.github.io/cadence-arcade/mario/ . Progress notes in `mario/PROGRESS.md`.

- `mario/game.py`: the body (nes-py), the brain's view, and the coach.
- `mario/train.py`: coach demos, then DAgger rounds; checkpoints and `history.json` in `runs/main`.
- `mario/baselines.py`: random, always-run, jump rhythm, and an MLP on the brain's final data set.
- `mario/record.py` + `mario/build_page.py` + `mario/page_template.html` + `mario/page_text.json`:
  real frames of the brain's best eval attempt (replayed from its stored buttons, checked to end at
  the same x) and the page.

### Rerun (Windows; nes-py needs a C++ compiler)

```bat
:: once, in a "x64 Native Tools" prompt or after vcvars64.bat (MSVC 2019 Build Tools work):
set DISTUTILS_USE_SDK=1 & set MSSdk=1
uv venv --python 3.11 .venv-mario
uv pip install --python .venv-mario\Scripts\python.exe "cadence-net[fast]==0.91.1" "numpy<2" pillow setuptools wheel
uv pip install --python .venv-mario\Scripts\python.exe --no-build-isolation-package nes-py nes-py gym-super-mario-bros
```

```bash
cd mario; PY=../.venv-mario/Scripts/python.exe
$PY train.py --out runs/main --hours 3 --workers 5     # one brain; --resume continues it
$PY baselines.py --out runs/baselines.json --mlp runs/main
$PY record.py --run runs/main --out ../docs/mario --brain-live
$PY build_page.py --run runs/main --baselines runs/baselines.json --out ../docs/mario
```

No ROM is in this repo: gym-super-mario-bros installs its own copy, and `*.nes` is git-ignored.
