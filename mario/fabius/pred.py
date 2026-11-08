"""Fabius's prediction slots (vault Plans/Fabius Build 2026-11-08, step 5).

What the picture does at the next decision, per block of the present grey grid (the periphery's first 832 numbers,
26 x 32 cells of 8 x 8 px), pooled 2 x 4 into 13 x 8 = 104 blocks: the block mean at the next decision minus now,
below -D class 0 (less contrast), above +D class 2 (more), else 1 (same). The last decision of an attempt has no next
picture and gets "same" everywhere.
"""
from __future__ import annotations
import numpy as np

ROWS, COLS = 26, 32
PR, PC = 2, 4
N_BLOCKS = (ROWS // PR) * (COLS // PC)  # 104
D = 0.03  # set from the 1-1 pile so "same" is not > 90% of labels (reported by the tiny test)


def blocks(O: np.ndarray) -> np.ndarray:
    """(n, width) uint8 obs -> (n, 104) block means of the present grey grid, 0..1."""
    g = O[:, :ROWS * COLS].astype(np.float32).reshape(-1, ROWS // PR, PR, COLS // PC, PC) / 255.0
    return g.mean(axis=(2, 4)).reshape(len(O), N_BLOCKS)


def labels(O: np.ndarray, d: float = D) -> np.ndarray:
    """(n, 104) int64 classes for every decision of ONE ordered attempt O (n, width)."""
    b = blocks(O)
    out = np.ones((len(O), N_BLOCKS), np.int64)
    if len(O) > 1:
        dif = b[1:] - b[:-1]
        out[:-1] = np.where(dif < -d, 0, np.where(dif > d, 2, 1))
    return out
