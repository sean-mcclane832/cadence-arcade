"""Fabius's eyes (Sean, 2026-11-07: "the second option for the eyes, it wont need the color of the entire screen").

One eye, two resolutions, like ours:
- periphery: the whole playfield, grey, 8 x 8 px blocks (26 x 32), present and previous decision. This is exactly
  Cato's eye today (1,664 numbers), so the old layout is a strict subset.
- fovea: a 96 x 64 px window around Mario, in colour, 4 x 4 px blocks (24 x 16 = 384 blocks x R, G, B = 1,152),
  plus the previous decision's grey at the SAME screen window (384), so a change there is the world moving
  (and the scroll), not the window following Mario.
The window looks ahead: 24 px behind Mario's centre to 72 px in front, 40 px above to 24 px below.
Body sense (336) is unchanged. Total 1,664 + 336 + 1,536 = 3,536 inputs (0.95's recall memory needs <= 4,096).

Gaze comes from RAM (Mario's on-screen x 0x3AD and y 0xCE), the way an eye points at what it follows. The brain
gets the picture in the window, never the numbers. Colour per block is the block's own colour gated by how far it is
from the frame's background colour (scaled by the window's contrast, 95th percentile, like the grey eye), so the sky
reads 0 and things keep their hue.
"""
from __future__ import annotations
import numpy as np

FOV_W, FOV_H, FOV_BLOCK = 96, 64, 4
BEHIND, ABOVE = 24, 40                        # window starts this far left of / above Mario's centre
FOV_COLS, FOV_ROWS = FOV_W // FOV_BLOCK, FOV_H // FOV_BLOCK   # 24 x 16
N_FOV_CELLS = FOV_COLS * FOV_ROWS            # 384
N_FOVEA = 4 * N_FOV_CELLS                     # RGB now + grey before = 1,536
N_PERIPHERY, N_BODY = 1664, 336
N_FABIUS = N_PERIPHERY + N_BODY + N_FOVEA     # 3,536


def mario_centre(ram) -> tuple[int, int]:
    """Mario's centre in screen pixels (x 0-255, y 0-239)."""
    big = int(ram[0x0754]) == 0              # PlayerSize: 0 big, 1 small
    return int(ram[0x03AD]) + 8, int(ram[0x00CE]) + (16 if big else 24)


def window(ram) -> tuple[int, int]:
    cx, cy = mario_centre(ram)
    return cx - BEHIND, cy - ABOVE


def crop(img: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """FOV_H x FOV_W crop at (x0, y0); outside the screen reads as the image's background (filled by caller)."""
    h, w = img.shape[:2]
    out = np.zeros((FOV_H, FOV_W) + img.shape[2:], dtype=np.float32)
    xs0, ys0 = max(x0, 0), max(y0, 0); xs1, ys1 = min(x0 + FOV_W, w), min(y0 + FOV_H, h)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return out, (xs0 - x0, ys0 - y0, xs1 - x0, ys1 - y0)


def background_rgb(screen: np.ndarray) -> np.ndarray:
    q = screen[32:].reshape(-1, 3).astype(np.int64)
    key = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    k = np.bincount(key).argmax()
    return np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255], np.float32)


def _blocks(a: np.ndarray) -> np.ndarray:
    return a.reshape(FOV_ROWS, FOV_BLOCK, FOV_COLS, FOV_BLOCK, *a.shape[2:]).mean(axis=(1, 3))


def fovea(screen: np.ndarray, prev_screen: np.ndarray, ram) -> np.ndarray:
    """1,536 numbers in [0, 1]: R, G, B distance from background (384 each, row-major), then the previous
    decision's grey distance at the same window (384)."""
    bg = background_rgb(screen)
    x0, y0 = window(ram)
    now, (a, b, c, d) = crop(screen.astype(np.float32), x0, y0)
    fill = np.ones((FOV_H, FOV_W), bool); fill[b:d, a:c] = False
    now[fill] = bg                                             # off-screen = background, reads 0
    blk = _blocks(now)                                         # 16 x 24 x 3, 0-255
    # how much each block differs from the background (0 = sky), scaled by the window's contrast; the block's
    # own colour is kept (Mario red, pipes green, lava orange), only gated by that difference
    dist = np.abs(blk - bg).max(axis=-1) / 255.0
    scale = max(float(np.percentile(dist, 95)), 0.08)
    seen = np.clip(dist / scale, 0, 1)
    col = (blk / 255.0) * seen[..., None]
    gw = np.array([0.299, 0.587, 0.114], np.float32)
    before, _ = crop(prev_screen.astype(np.float32), x0, y0)
    before[fill] = bg
    gprev = np.clip(np.abs(_blocks(before) - bg).max(axis=-1) / 255.0 / scale, 0, 1)
    return np.concatenate([col[..., 0].ravel(), col[..., 1].ravel(), col[..., 2].ravel(), gprev.ravel()])
