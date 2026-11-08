"""Picture of what Fabius's eyes see: replays a recorded coach run and, at chosen decisions, draws the screen with
the fovea window, the grey periphery (Cato's eye) and the colour fovea (+ the previous decision's grey there).
    python eyesdemo.py --level 1-2 --start 0 --at 60,120,200 --out shots
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import game  # noqa: E402
import eyes  # noqa: E402

RUNS = [HERE.parent / "coach" / "runs" / d / "runs.json" for d in ("cov_final", "cov_w4_7")]


def panel(screen, prev, ram, periph):
    W = Image.new("RGB", (256 * 3 + 40, 300), (24, 26, 32)); d = ImageDraw.Draw(W)
    s = Image.fromarray(screen); x0, y0 = eyes.window(ram)
    ImageDraw.Draw(s).rectangle([x0, y0, x0 + eyes.FOV_W - 1, y0 + eyes.FOV_H - 1], outline=(255, 220, 0), width=2)
    W.paste(s, (10, 40)); d.text((10, 20), "screen + fovea window", fill=(225, 228, 235))
    p = (periph.reshape(game.ROWS, game.COLS) * 255).astype(np.uint8)
    W.paste(Image.fromarray(p).resize((256, 208), Image.NEAREST), (276, 72)); d.text((276, 20), "periphery: grey 26 x 32", fill=(225, 228, 235))
    f = eyes.fovea(screen, prev, ram); n = eyes.N_FOV_CELLS
    rgb = np.stack([f[i * n:(i + 1) * n].reshape(eyes.FOV_ROWS, eyes.FOV_COLS) for i in range(3)], -1)
    W.paste(Image.fromarray((rgb * 255).astype(np.uint8)).resize((240, 160), Image.NEAREST), (542, 40))
    g = (f[3 * n:].reshape(eyes.FOV_ROWS, eyes.FOV_COLS) * 255).astype(np.uint8)
    W.paste(Image.fromarray(g).resize((120, 80), Image.NEAREST), (542, 210))
    d.text((542, 20), "fovea: colour 24 x 16", fill=(225, 228, 235)); d.text((670, 230), "fovea before\n(grey, same\nwindow)", fill=(225, 228, 235))
    return W


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--level", default="1-2"); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--at", default="60,120,200"); ap.add_argument("--out", default="shots")
    a = ap.parse_args()
    r = next(q for src in RUNS if src.exists() for q in json.loads(src.read_text()) if q["level"] == a.level and q["start"] == a.start)
    at = {int(v) for v in a.at.split(",")}; out = HERE / a.out; out.mkdir(exist_ok=True)
    g = game.Game(noop_start=a.start, level=tuple(int(v) for v in a.level.split("-")), menu="lean", bg_per_frame=True); e = g.env
    prev = g.screen.copy()
    for k, act in enumerate(r["acts"]):
        if g.done or k > max(at):
            break
        sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
        e._time_last, e._x_position_last, e.done = sv
        prev = g.screen.copy(); g.step(int(act))
        if k + 1 in at:
            periph = game.frame_features(g.screen, None)
            periph = np.clip(periph / max(float(np.percentile(periph, 95)), 0.08), 0, 1)
            panel(g.screen, prev, e.ram, periph).save(out / f"{a.level}_d{k + 1:04d}.png")
            print("saved", out / f"{a.level}_d{k + 1:04d}.png")


if __name__ == "__main__":
    main()
