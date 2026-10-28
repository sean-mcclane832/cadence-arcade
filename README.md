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
