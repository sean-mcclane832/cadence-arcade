"""Record a brain's eval attempt as real NES frames for the replay page.

The eval attempts in history.json store the buttons the brain pressed. The emulator is
deterministic, so replaying those buttons from the same start reproduces the attempt frame for
frame; this script checks that it ends at the same x and flag, then writes every 2nd frame
(30 fps) into PNG sprite sheets of 8 x 8 frames plus a JSON of per-frame button and x.

    python record.py --run runs/main --out ../docs/mario [--round -1] [--start 0]

`--brain-live` instead loads the saved brain and lets it play again (it must agree with the
stored buttons), which proves the stored buttons are the brain's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from game import ACTION_NAMES, Game  # noqa: E402
from train import MAX_DECISIONS, STUCK  # noqa: E402

SHEET = 8  # frames per sheet side
EVERY = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="../docs/mario")
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--brain-live", action="store_true")
    ap.add_argument("--name", default="replay")
    ap.add_argument("--assist", default=None, help="an assist.py JSON instead of an eval attempt")
    args = ap.parse_args()
    run = HERE / args.run
    hist = json.loads((run / "history.json").read_text())
    rnd = hist[args.round]
    if "eval" not in rnd:
        rnd = hist[args.round - 1]
    evals = rnd["eval"]
    if args.assist:
        pick = json.loads((HERE / args.assist).read_text())
        pick["round"] = rnd["round"]
    elif args.start is None:
        pick = max(evals, key=lambda e: (e["flag"], e["max_x"], -e["start"]))
    else:
        pick = next(e for e in evals if e["start"] == args.start)
    acts = pick["actions"]
    who = pick.get("who", ["brain"] * len(acts))

    if args.brain_live:
        from cadence import Brain
        brain = Brain.load(run / "brain.npz")
        g = Game(noop_start=pick["start"])
        live = []
        while not g.done and g.decisions < MAX_DECISIONS and g.stuck < STUCK:
            a = int(np.asarray(brain.predict(g.observation()[None])).reshape(-1)[0])
            live.append(a); g.step(a)
        same = live == acts
        print(f"live brain replay: {len(live)} decisions, identical to stored buttons: {same}, x {g.max_x}, flag {g.flag}")
        g.close()

    g = Game(noop_start=pick["start"])
    frames, per_frame = [], []
    for a, w in zip(acts, who):
        keep = []
        g.step(a, keep_frames=keep)
        for f in keep:
            per_frame.append((a, int(g.env._x_position), w))
            frames.append(f)
        if g.done:
            break
    assert g.max_x == pick["max_x"] and g.flag == pick["flag"], (g.max_x, pick["max_x"], g.flag)
    g.close()

    out = (HERE / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kept = list(range(0, len(frames), EVERY))
    if kept[-1] != len(frames) - 1:
        kept.append(len(frames) - 1)
    hold = 45  # repeat the last frame ~1.5 s
    kept += [kept[-1]] * hold
    n_sheets = int(np.ceil(len(kept) / SHEET ** 2))
    h, w = frames[0].shape[:2]
    sheets = []
    for s in range(n_sheets):
        canvas = np.zeros((SHEET * h, SHEET * w, 3), np.uint8)
        for j, fi in enumerate(kept[s * SHEET ** 2:(s + 1) * SHEET ** 2]):
            r, c = divmod(j, SHEET)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[fi]
        img = Image.fromarray(canvas).quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        name = f"{args.name}_{s:02d}.png"
        img.save(out / name, optimize=True)
        sheets.append(name)
    meta = dict(start=pick["start"], round=rnd["round"], flag=pick["flag"], max_x=pick["max_x"],
                decisions=pick["decisions"], fps=60 // EVERY, w=w, h=h, sheet=SHEET, sheets=sheets,
                frames=len(kept), hold=hold,
                button=[per_frame[fi][0] for fi in kept], x=[per_frame[fi][1] for fi in kept],
                button_names=ACTION_NAMES, who=[per_frame[fi][2] == "coach" for fi in kept],
                assisted=bool(args.assist), rescues=pick.get("rescues", []),
                brain_share=pick.get("brain_share", 1.0))
    (out / f"{args.name}.json").write_text(json.dumps(meta))
    total = sum((out / s).stat().st_size for s in sheets)
    print(f"round {rnd['round']} start {pick['start']} x {pick['max_x']} flag {pick['flag']}: "
          f"{len(kept)} frames, {n_sheets} sheets, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
