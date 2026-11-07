"""Live training dashboard: reads a run's log.txt and curve.csv as training writes them, redraws every 5 s.
    python dashboard.py CiceroLowB [--also CiceroLowA]
Shows: chunk and phase (estimated from the time since the last log line), the latest measurement per level
(best x of the 8 fixed starts and flags out of 8), and the learning curve from chunk 0 (mean best x over all levels,
flags below it). Read only: it never touches the run.
"""
from __future__ import annotations
import argparse, csv, re, time
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

HERE = Path(__file__).resolve().parent
LEVELS = [f"{w}-{s}" for w in range(1, 4) for s in range(1, 5)]
BG, FG, DIM, PANEL = (24, 26, 32), (225, 228, 235), (130, 136, 150), (32, 35, 44)
ORANGE, TEAL, GOLD, GREY = (255, 150, 60), (0, 200, 160), (255, 200, 80), (95, 100, 115)
W, H = 720, 1000
# rough flag x per level (the furthest best x seen at a flag in these runs; only for bar scaling)
LEVEL_LEN = {"1-1": 3160, "1-2": 3050, "1-3": 2480, "1-4": 2260, "2-1": 3220, "2-2": 3050, "2-3": 3550, "2-4": 2260,
             "3-1": 3150, "3-2": 3060, "3-3": 2400, "3-4": 2260}
CHUNK_RE = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d) chunk (\d+) \(teach (\d+)s, SEL (\d+)s.*?\): mean (\d+) flags (\d+) \| (.*)$")
DAGGER_RE = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d)\s+dagger \((\d+)s\)")


def read(run):
    d = HERE / "runs" / run
    chunks, daggers = [], []
    for line in (d / "log.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        m = CHUNK_RE.match(line)
        if m:
            lv = {}
            for part in m.group(7).split():
                pass
            for k, f, x in re.findall(r"(\d-\d) (\d+)/m(\d+)", m.group(7)):
                lv[k] = (int(f), int(x))
            chunks.append(dict(when=m.group(1), chunk=int(m.group(2)), teach=int(m.group(3)), sel=int(m.group(4)),
                               mean=int(m.group(5)), flags=int(m.group(6)), levels=lv))
        m = DAGGER_RE.match(line)
        if m:
            daggers.append(dict(when=m.group(1), secs=int(m.group(2)), after=chunks[-1]["chunk"] if chunks else 0))
    curve = []
    with open(d / "curve.csv", newline="") as f:
        for r in csv.DictReader(f):
            curve.append((int(r["chunk"]), float(r["sel_mean_all"]), int(r["sel_flags_all"])))
    return chunks, daggers, curve


def when_ts(s):
    return time.mktime(time.strptime(time.strftime("%Y-") + s, "%Y-%m-%d %H:%M:%S"))


def draw(run, also, last_chunk):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 15); big = ImageFont.truetype("consolab.ttf", 22)
    except OSError:
        font = big = ImageFont.load_default()
    try:
        chunks, daggers, curve = read(run)
    except Exception as e:  # noqa: BLE001  (a half-written file: try again next tick)
        d.text((16, 16), f"waiting for {run} files: {e}", fill=FG, font=font); return img
    last = chunks[-1]
    since = time.time() - when_ts(last["when"])
    typical = last["teach"] + last["sel"] + 15
    if since < last["teach"]:
        phase = f"teaching chunk {last['chunk'] + 1} (lessons from the coach pile)"
    elif since < typical + 30:
        phase = f"measuring chunk {last['chunk'] + 1} (96 plays: 8 starts x 12 levels)"
    else:
        phase = "probably a DAgger round (it plays, the coach labels its mistakes)"
    d.text((16, 12), f"{run} training", fill=FG, font=big)
    d.text((16, 44), f"chunk {last['chunk']} of {last_chunk} done at {last['when'][6:]}   "
                     f"({since / 60:.1f} min ago)", fill=FG, font=font)
    d.text((16, 64), f"now: {phase}", fill=GOLD, font=font)
    d.text((16, 84), "phase is estimated from timing; numbers below are the real log", fill=DIM, font=font)

    # per-level bars from the latest measurement
    y0 = 118
    d.text((16, y0), f"latest measurement (chunk {last['chunk']}): best x of 8 starts, flags out of 8", fill=DIM, font=font)
    prev = chunks[-2]["levels"] if len(chunks) > 1 else {}
    for i, L in enumerate(LEVELS):
        f, x = last["levels"].get(L, (0, 0)); y = y0 + 24 + i * 30
        d.text((16, y), L, fill=FG, font=font)
        d.rectangle([64, y + 2, 64 + 460, y + 20], fill=PANEL)
        frac = min(1.0, x / LEVEL_LEN[L])
        d.rectangle([64, y + 2, 64 + int(460 * frac), y + 20], fill=GOLD if f else ORANGE)
        if L in prev:
            px = min(1.0, prev[L][1] / LEVEL_LEN[L]); xx = 64 + int(460 * px)
            d.line([xx, y, xx, y + 22], fill=FG, width=2)
        d.text((534, y), f"x {x:<5d} flags {f}/8", fill=GOLD if f else FG, font=font)
    d.text((64, y0 + 24 + 12 * 30), "white tick = previous chunk; gold bar = at least one flag", fill=DIM, font=font)

    # learning curve
    cy, ch, cx, cw = 540, 300, 60, 630
    d.text((16, cy - 24), "learning curve: mean best x over all 12 levels per chunk (8 starts each)", fill=DIM, font=font)
    d.rectangle([cx, cy, cx + cw, cy + ch], fill=PANEL)
    series = [(run, curve, TEAL)]
    if also:
        try:
            series.append((also, read(also)[2], GREY))
        except Exception:  # noqa: BLE001
            pass
    top = max(max(m for _, m, _ in c) for _, c, _ in series) * 1.05
    n = last_chunk
    for frac in (0.25, 0.5, 0.75, 1.0):
        yy = cy + ch - int(ch * frac); d.line([cx, yy, cx + cw, yy], fill=(45, 48, 58))
        d.text((8, yy - 8), f"{int(top * frac)}", fill=DIM, font=font)
    for dg in daggers:
        xx = cx + int(cw * dg["after"] / n); d.line([xx, cy, xx, cy + ch], fill=(70, 60, 110))
    for name, c, col in series[::-1]:
        pts = [(cx + int(cw * k / n), cy + ch - int(ch * m / top)) for k, m, _ in c]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2 if name == run else 1)
    k, m, _ = curve[-1]
    d.ellipse([cx + int(cw * k / n) - 4, cy + ch - int(ch * m / top) - 4, cx + int(cw * k / n) + 4, cy + ch - int(ch * m / top) + 4], fill=GOLD)
    d.text((cx, cy + ch + 6), "0", fill=DIM, font=font); d.text((cx + cw - 30, cy + ch + 6), str(n), fill=DIM, font=font)
    leg = f"{run} (teal)" + (f"   {also} (grey)" if also else "") + "   purple lines = DAgger rounds"
    d.text((cx, cy + ch + 26), leg, fill=DIM, font=font)
    # flags per chunk
    fy = cy + ch + 56
    d.text((16, fy), "flags per chunk (out of 96 plays)", fill=DIM, font=font)
    fmax = max(1, max(f for _, _, f in curve))
    for k, _, f in curve:
        if f:
            xx = cx + int(cw * k / n); d.line([xx, fy + 60, xx, fy + 60 - int(40 * f / fmax)], fill=GOLD, width=2)
    d.text((cx + cw - 230, fy), f"latest: {last['flags']} flags, mean {last['mean']}", fill=FG, font=font)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--also", default=None)
    ap.add_argument("--last-chunk", type=int, default=200); ap.add_argument("--pos", default="+1180+20")
    a = ap.parse_args()
    root = tk.Tk(); root.title(f"{a.run}: training dashboard"); root.geometry(a.pos)
    lab = tk.Label(root, bg="black"); lab.pack()
    root.bind("q", lambda e: root.destroy())
    state = {}

    def tick():
        state["img"] = ImageTk.PhotoImage(draw(a.run, a.also, a.last_chunk)); lab.configure(image=state["img"])
        root.after(5000, tick)
    tick(); root.mainloop()


if __name__ == "__main__":
    main()
