"""Brain-cam: the game on the left, the brain's recorded activity on the right, as an MP4.

    python viz/brain_video.py --name u3ck83-1-1-clear --out ../docs/brain/clips/u3ck83-1-1-clear.mp4

Input: viz/out/<name>.npz from record_brain.py. Nothing is simulated here; every panel draws
recorded numbers:
  what it sees   the brain's real input for the decision: 26 x 32 grey blocks of the present
                 frame and of the previous decision's frame (0 = background)
  body sense     the 42 body numbers (game.proprio + held buttons), first of the 8 copies
  neurons        final activation after settling, per region of the connectome (association
                 sorted by the button each neuron pushes hardest; prefrontal is the working
                 memory, silent because predict() reads no live memory)
  buttons        the 7 motor neurons' final activation; the brain presses the highest one
  settle         the 7 motor neurons over the settle steps of this decision
  region means   mean activation per region over the last 90 decisions
The video runs at 30 fps of real game time (2 frames per decision).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIZ = Path(__file__).resolve().parent
W, H = 1280, 720
GX = 768  # right panel starts here
BG = (16, 18, 24)
FG = (225, 228, 235)
DIM = (130, 136, 150)
HI = (255, 196, 61)
BUTTONS = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
BODY_GROUPS = [("x speed", 9), ("y speed", 9), ("ground/jump/fall/pole", 4), ("rise", 1), ("face", 2),
               ("power", 3), ("hurt/star", 2), ("time left", 8), ("held A B > <", 4)]
REG_COL = {"sensory": (90, 170, 255), "association": (255, 150, 60), "prefrontal": (170, 120, 255),
           "motor": (90, 220, 130)}


def font(size, bold=False):
    for f in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def lut():
    """Activation colour map: -0.1 (inhibited) blue, 0 near black, 1 bright yellow."""
    stops = [(-0.1, (40, 90, 200)), (0.0, (20, 20, 28)), (0.25, (110, 30, 120)), (0.55, (220, 70, 60)),
             (0.8, (250, 160, 40)), (1.0, (255, 245, 150))]
    xs = np.linspace(-0.1, 1.0, 256)
    out = np.zeros((256, 3))
    for c in range(3):
        out[:, c] = np.interp(xs, [s[0] for s in stops], [s[1][c] for s in stops])
    return out.astype(np.uint8)


LUT = lut()


def colorize(a):
    i = np.clip(((np.asarray(a) + 0.1) / 1.1 * 255).round(), 0, 255).astype(np.int64)
    return LUT[i]


def grey_img(v, rows, cols, px):
    g = (np.clip(v, 0, 1).reshape(rows, cols) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1)
    return Image.fromarray(rgb.repeat(px, 0).repeat(px, 1))


def grid_img(a, rows, cols, px):
    c = colorize(a).reshape(rows, cols, 3)
    img = c.repeat(px, 0).repeat(px, 1)
    img[px - 1::px, :, :] = img[px - 1::px, :, :] // 2  # faint cell lines
    img[:, px - 1::px, :] = img[:, px - 1::px, :] // 2
    return Image.fromarray(img)


class Layout:
    def __init__(self, z):
        self.z = z
        self.meta = json.loads(str(z["meta"]))
        self.pops = {k[4:]: z[k] for k in z.files if k.startswith("pop_")}
        a2m = z["a2m"]
        pref = a2m.argmax(1)
        self.assoc_order = np.lexsort((-a2m.max(1), pref))  # grouped by preferred button
        self.assoc_pref = pref[self.assoc_order]
        fin = z["final"]
        self.reg_mean = {r: fin[:, self.pops[r]].mean(1) for r in ("sensory", "association", "prefrontal", "motor")}
        self.f = {s: font(s) for s in (12, 13, 15, 17)}
        self.fb = {s: font(s, True) for s in (13, 15, 18)}
        self.tag = TAG
        self.act = bool(self.meta.get("act")); self.parts = list(self.meta.get("parts", []))
        self.static = self._static()

    def _static(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        m, P = self.meta, self.pops
        d.text((GX + 12, 8), f"Cadence brain {Path(m['brain']).stem}  |  World {m['level']}  |  {self.tag}",
               fill=FG, font=self.fb[15])
        d.text((GX + 12, 34), "WHAT IT SEES  (real input, 26x32 blocks, shown 3x brighter)", fill=DIM, font=self.f[12])
        d.text((GX + 12, 50 + 182 + 2), "now", fill=DIM, font=self.f[12])
        d.text((GX + 264, 50 + 182 + 2), "previous decision", fill=DIM, font=self.f[12])
        y = 254
        d.text((GX + 12, y), "BODY SENSE  (42 numbers from Mario's own body, x8 copies in the input)", fill=DIM, font=self.f[12])
        x = GX + 12
        for name, n in BODY_GROUPS:
            d.line([(x, y + 40), (x + n * 11 - 2, y + 40)], fill=DIM)
            x += n * 11
        x = GX + 12
        labs = ["x spd", "y spd", "state", "", "face", "pow", "", "time", "held"]
        for (name, n), lab in zip(BODY_GROUPS, labs):
            if lab:
                d.text((x, y + 42), lab, fill=DIM, font=self.f[12])
            x += n * 11
        y = 318
        d.text((GX + 12, y), "NEURONS  final activation after settling, by connectome region", fill=DIM, font=self.f[12])
        na, npf, ns = len(P["association"]), len(P["prefrontal"]), len(P["sensory"])
        d.text((GX + 12, y + 18), f"association ({na})", fill=REG_COL["association"], font=self.fb[13])
        d.text((GX + 12, y + 34), "sorted by the button", fill=DIM, font=self.f[12])
        d.text((GX + 12, y + 48), "each pushes hardest", fill=DIM, font=self.f[12])
        if self.act and "thought" in self.parts:
            d.text((GX + 176, y + 18), f"thought memory ({npf})", fill=REG_COL["prefrontal"], font=self.fb[13])
            d.text((GX + 176, y + 34), "prefrontal, live: carries", fill=DIM, font=self.f[12])
            d.text((GX + 176, y + 48), "the last moments' activity", fill=DIM, font=self.f[12])
        else:
            d.text((GX + 176, y + 18), f"prefrontal ({npf})", fill=REG_COL["prefrontal"], font=self.fb[13])
            d.text((GX + 176, y + 34), "working memory: off", fill=DIM, font=self.f[12])
            d.text((GX + 176, y + 48), "(this brain has none)" if self.act else "in predict mode, so 0", fill=DIM, font=self.f[12])
        if self.act and "echo" in self.parts and "history" in P:
            d.text((GX + 340, y + 18), f"echo ({len(P['history'])})", fill=(120, 200, 220), font=self.fb[13])
            d.text((GX + 340, y + 34), "fading copy of past input", fill=DIM, font=self.f[12])
        else:
            d.text((GX + 340, y + 18), f"sensory ({ns})", fill=REG_COL["sensory"], font=self.fb[13])
            d.text((GX + 340, y + 34), "eyes, now (first 832)", fill=DIM, font=self.f[12])
        y = 540
        d.text((GX + 12, y), "BUTTONS  (7 motor neurons, highest = pressed)", fill=DIM, font=self.f[12])
        d.text((GX + 300, 566), "SETTLE of this decision", fill=DIM, font=self.f[12])
        return img

    def frame(self, game, k):
        z, P = self.z, self.pops
        img = self.static.copy()
        g = Image.fromarray(game).resize((768, 720), Image.NEAREST)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        o = z["obs"][k]
        from game import N_FRAME
        img.paste(grey_img(3 * o[:N_FRAME], 26, 32, 7), (GX + 12, 50))
        img.paste(grey_img(3 * o[N_FRAME:2 * N_FRAME], 26, 32, 7), (GX + 264, 50))
        a = int(z["acts"][k])
        n = len(z["acts"])
        d.text((GX + 12, 8 + 0), "", fill=FG)
        d.text((W - 12, 34), f"decision {k + 1}/{n}   x {int(z['xs'][k])}", fill=FG, font=self.fb[13], anchor="ra")
        # body: first copy of 42
        body = o[2 * N_FRAME:2 * N_FRAME + 42]
        x = GX + 12
        for v in body:
            c = int(np.clip(v, 0, 1) * 255)
            d.rectangle([x, 274, x + 9, 292], fill=(c // 3, c, int(c * 0.85)) if v > 0.02 else (34, 36, 44))
            x += 11
        fin = z["final"][k]
        y = 318 + 66
        img.paste(grid_img(fin[P["association"]][self.assoc_order], 16, 16, 9), (GX + 12, y))
        img.paste(grid_img(fin[P["prefrontal"]], 16, 16, 9), (GX + 176, y))
        echo = self.act and "echo" in self.parts and "history" in P
        img.paste(grid_img(fin[(P["history"] if echo else P["sensory"])[:832]], 26, 32, 5), (GX + 340, y - 16))
        x0 = GX + 340; wd = 128; top = y - 16 + 130 + 20; hh = 30
        if self.act and "button" in self.parts and "efference" in P:
            # button memory: the efference region (what it pressed, fading), one cell per button
            d.text((x0, top - 16), "button memory (what it pressed)", fill=DIM, font=self.f[12])
            ef = fin[P["efference"]]
            for i, v in enumerate(ef):
                c = int(np.clip(v, 0, 1) * 255)
                d.rectangle([x0 + i * 22, top, x0 + i * 22 + 19, top + 18], fill=(c, int(c * 0.75), 40) if v > 0.02 else (34, 36, 44))
                d.text((x0 + i * 22 + 2, top + 20), BUTTONS[i][:3] if i != 2 and i != 4 else ("r+j" if i == 2 else "R+j"),
                       fill=DIM, font=self.f[12])
            return self._rest(img, d, fin, k)
        # region means over the last 90 decisions
        d.text((x0, top - 16), "region means, last 90 decisions", fill=DIM, font=self.f[12])
        d.rectangle([x0, top, x0 + wd, top + hh], outline=(50, 54, 64))
        lo = max(0, k - 89)
        for r in ("sensory", "association", "motor"):
            s = self.reg_mean[r][lo:k + 1]
            if len(s) > 1:
                pts = [(x0 + wd * (i + 90 - len(s)) / 89, top + hh - hh * float(np.clip(v / 0.25, 0, 1)))
                       for i, v in enumerate(s)]
                d.line(pts, fill=REG_COL[r], width=2)
        d.text((x0 + wd + 4, top - 2), "0.25", fill=DIM, font=self.f[12])
        d.text((x0 + wd + 4, top + hh - 12), "0", fill=DIM, font=self.f[12])
        return self._rest(img, d, fin, k)

    def _rest(self, img, d, fin, k):
        z, P = self.z, self.pops
        a = int(z["acts"][k])
        # buttons
        mot = fin[P["motor"]]
        by = 560
        for i, (name, v) in enumerate(zip(BUTTONS, mot)):
            yy = by + i * 22
            col = HI if i == a else (95, 100, 115)
            d.text((GX + 12, yy + 2), name, fill=HI if i == a else FG, font=(self.fb[13] if i == a else self.f[13]))
            bx = GX + 96
            d.rectangle([bx, yy + 4, bx + 120, yy + 16], fill=(30, 33, 41))
            zero = bx + int(120 * 0.1 / 1.1)
            end = bx + int(120 * (float(v) + 0.1) / 1.1)
            d.rectangle([min(zero, end), yy + 4, max(zero, end), yy + 16], fill=col)
            d.text((bx + 126, yy + 2), f"{v:+.2f}", fill=DIM if i != a else HI, font=self.f[12])
        # settle plot
        sm = z["settle_motor"][k]  # (steps+1, 7)
        px0, py0, pw, ph = GX + 300, 584, 196, 112
        d.text((px0, 545), "", fill=DIM)
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(50, 54, 64))
        S = len(sm) - 1
        for i in range(7):
            pts = [(px0 + pw * t / S, py0 + ph - ph * (float(v) + 0.1) / 1.1) for t, v in enumerate(sm[:, i])]
            d.line(pts, fill=HI if i == a else (85, 90, 105), width=2 if i == a else 1)
        d.text((px0, py0 + ph + 2), f"steps 0 to {S}, 7 motor neurons", fill=DIM, font=self.f[12])
        return np.asarray(img)


TAG = "solo, no coach"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=27)
    ap.add_argument("--tag", default="solo, no coach", help="right side of the title line")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    import sys
    sys.path.insert(0, str(VIZ.parent))
    z = np.load(VIZ / "out" / f"{args.name}.npz")
    L = Layout(z)
    vid, vdec = z["vid"], z["vid_dec"]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(out), (W, H), fps=30, codec="libx264", pix_fmt_out="yuv420p",
                                    output_params=["-crf", str(args.crf), "-preset", "slow", "-tune", "animation",
                                                   "-movflags", "+faststart", "-threads", "2"],
                                    macro_block_size=16)
    w.send(None)
    poster = None
    for i, (f, k) in enumerate(zip(vid, vdec)):
        fr = L.frame(f, int(k))
        if i == min(len(vid) - 1, 150):
            poster = fr
        w.send(np.ascontiguousarray(fr))
    w.close()
    Image.fromarray(poster).save(out.with_name(out.stem + "_poster.jpg"), quality=85)
    m = L.meta
    print(f"{out.name}: {out.stat().st_size / 1e6:.2f} MB, {len(vid) / 30:.1f} s, {m['end']} at x {m['max_x']}")


if __name__ == "__main__":
    main()
