"""Assisted run: the brain plays World 1-1 from the start; when it dies or gets stuck, the game
rewinds and the coach drives briefly past the trouble, then hands control back to the brain.

Every decision is tagged with who pressed the button. The share of brain decisions and the number
of rescues measure how much of the level the brain plays by itself. Rewinding is exact: the
emulator is deterministic, so a fresh game replaying the logged buttons reaches the same state.

    python assist.py --run runs/main --start 0 --out runs/main/assist_s0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from game import Game  # noqa: E402

REWIND = 12      # decisions to go back after a death (~0.8 s of game time)
COACH_FOR = 14   # decisions the coach drives after a rewind
STUCK = 25
MAX_DECISIONS = 1200


def replay(start: int, acts: list[int]) -> Game:
    g = Game(noop_start=start)
    for a in acts:
        g.step(a)
    return g


def assisted(brain, start: int, log=print) -> dict:
    acts: list[int] = []
    who: list[str] = []
    rescues = []
    g = Game(noop_start=start)
    rewind = REWIND
    while len(acts) < MAX_DECISIONS:
        a = int(np.asarray(brain.predict(g.observation()[None])).reshape(-1)[0])
        acts.append(a); who.append("brain")
        g.step(a)
        if g.flag:
            break
        if g.done or g.stuck >= STUCK:
            why = "died" if g.done else "stuck"
            at_x = g.max_x
            back = min(len(acts), rewind + (STUCK if why == "stuck" else 0))
            del acts[-back:], who[-back:]
            g.close()
            g = replay(start, acts)
            passed, n_coach = False, 0
            for _ in range(COACH_FOR + (back - rewind)):
                c = g.coach()
                acts.append(c); who.append("coach"); n_coach += 1
                g.step(c)
                if g.done:
                    break
                if g.x > at_x + 24 and n_coach >= 4:
                    passed = True
                    break
            rescues.append(dict(at_decision=len(acts), x=at_x, why=why, coach_passed=passed or g.flag,
                                coach_decisions=n_coach))
            log(f"  rescue {len(rescues)} at x {at_x} ({why}), coach passed: {passed}, {n_coach} coach decisions")
            if g.flag:
                break
            if g.done:  # the coach could not save it from there: go further back next time
                rewind += REWIND
                del acts[-n_coach:], who[-n_coach:]
                rescues.pop()
                g.close()
                g = replay(start, acts)
            else:
                rewind = REWIND
            if len(rescues) > 60:
                break
    out = dict(start=start, actions=acts, who=who, rescues=rescues, flag=g.flag, max_x=g.max_x,
               brain_share=float(np.mean([w == "brain" for w in who])), decisions=len(acts))
    g.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from cadence import Brain
    brain = Brain.load(HERE / args.run / "brain.npz")
    res = assisted(brain, args.start)
    print(f"start {args.start}: flag {res['flag']} x {res['max_x']}, {res['decisions']} decisions, "
          f"brain pressed {100 * res['brain_share']:.1f}%, rescues {len(res['rescues'])}")
    out = Path(args.out) if args.out else HERE / args.run / f"assist_s{args.start}.json"
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
