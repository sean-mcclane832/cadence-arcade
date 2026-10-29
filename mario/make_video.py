"""Turn the replay page's recorded NES frames into phone-friendly video for docs/mario/watch/.

    python make_video.py            # needs ffmpeg on PATH, or the imageio-ffmpeg package

Input: docs/mario/replay.json + replay_*.png (the brain-alone run) and assist.json +
assist_*.png (the coach-assisted run), written by record.py from the real emulator frames
(every 2nd frame, 30 fps). Nothing is re-simulated here; frames are cut out of the sheets,
scaled 3x with nearest neighbour, and encoded as H.264 MP4 (plays inline on iPhone) and VP9
WebM. In the assisted clip, frames where the coach drives get a red "COACH" bar, the same
per-frame `who` flags the replay page uses for its badge.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs" / "mario"
OUT = DOCS / "watch"
SCALE = 3


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def frames(name: str):
    meta = json.loads((DOCS / f"{name}.json").read_text())
    w, h, k = meta["w"], meta["h"], meta["sheet"]
    sheets = [np.asarray(Image.open(DOCS / s).convert("RGB")) for s in meta["sheets"]]
    for i in range(meta["frames"]):
        s, j = divmod(i, k * k)
        r, c = divmod(j, k)
        yield meta, i, sheets[s][r * h:(r + 1) * h, c * w:(c + 1) * w]


def label(img: Image.Image, text: str, color) -> None:
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:
        font = ImageFont.load_default()
    tw = d.textlength(text, font=font)
    x0 = (img.width - tw) / 2 - 14
    d.rectangle([x0, 10, x0 + tw + 28, 52], fill=color)
    d.text((x0 + 14, 14), text, fill=(255, 255, 255), font=font)


def encode(name: str, out: str, coach_bar: bool) -> None:
    exe = ffmpeg()
    first = poster = None
    procs = []
    for meta, i, f in frames(name):
        img = Image.fromarray(f).resize((f.shape[1] * SCALE, f.shape[0] * SCALE), Image.NEAREST)
        if coach_bar and meta["who"][i]:
            label(img, "COACH DRIVING", (220, 60, 60))
        if i == 90:  # ~3 s in; frame 0 is a black screen
            poster = img.copy()
        if first is None:
            first = img
            size = f"{img.width}x{img.height}"
            common = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size,
                      "-r", str(meta["fps"]), "-i", "-"]
            procs = [
                subprocess.Popen(common + ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                                           "-tune", "animation", "-movflags", "+faststart",
                                           str(OUT / f"{out}.mp4")], stdin=subprocess.PIPE),
                subprocess.Popen(common + ["-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", "-pix_fmt", "yuv420p",
                                           "-row-mt", "1", str(OUT / f"{out}.webm")], stdin=subprocess.PIPE),
            ]
        b = np.asarray(img).tobytes()
        for p in procs:
            p.stdin.write(b)
    for p in procs:
        p.stdin.close()
        assert p.wait() == 0
    poster.save(OUT / f"{out}_poster.png", optimize=True)
    for ext in ("mp4", "webm"):
        print(f"{out}.{ext}: {(OUT / f'{out}.{ext}').stat().st_size / 1e6:.2f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encode("replay", "solo", coach_bar=False)
    encode("assist", "assisted", coach_bar=True)


if __name__ == "__main__":
    main()
