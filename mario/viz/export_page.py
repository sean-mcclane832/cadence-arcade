"""Pack recorded runs (viz/out/<name>.npz) for the static page docs/brain/.

    python viz/export_page.py u3ck83-1-1-clear u3ck83-1-1-death u3ck83-1-4-clear u3ck83-1-4-death

Per run it writes docs/brain/data/<name>.json (small metadata), <name>.bin.gz (gzipped uint8 arrays,
layout in the json; the page unzips it with DecompressionStream) and <name>_NN.webp (sheets of the screen the brain saw at each decision).
Activations are quantized to uint8: q = round((a + 0.1) / 1.1 * 255), so 0 -> -0.1 and 255 -> 1.0
(Cadence activations here live in [-0.1, 1]). Inputs (0..1) are q = round(v * 255).
Saliency is quantized against the run's 99.5th percentile (scale in the json).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VIZ = Path(__file__).resolve().parent
OUT = VIZ.parent.parent / "docs" / "brain" / "data"
SHEET = 8  # 8 x 8 frames per webp sheet
FW, FH = 256, 240


def q_act(a):
    return np.clip(np.round((np.asarray(a) + 0.1) / 1.1 * 255), 0, 255).astype(np.uint8)


def q01(v):
    return np.clip(np.round(np.asarray(v) * 255), 0, 255).astype(np.uint8)


def export(name, titles):
    z = np.load(VIZ / "out" / f"{name}.npz")
    meta = json.loads(str(z["meta"]))
    n = len(z["acts"])
    from game import N_FRAME
    obs = z["obs"]
    parts, layout, off = [], {}, 0

    def add(key, arr):
        nonlocal off
        arr = np.ascontiguousarray(arr, np.uint8)
        layout[key] = dict(offset=off, shape=list(arr.shape))
        parts.append(arr.tobytes()); off += arr.nbytes

    add("act", q_act(z["final"]))                                    # (n, 2519)
    add("eyes", q01(obs[:, :N_FRAME]))                                # (n, 832) present frame
    add("body", q01(obs[:, 2 * N_FRAME:2 * N_FRAME + 42]))            # (n, 42) first copy
    add("settle", q_act(z["settle_motor"]))                           # (n, S+1, 7)
    sal = z["saliency"]
    scale = float(max(np.percentile(sal, 99.5), 1e-6))
    add("sal", q01(sal / scale))                                      # (n, 832)
    # body copies are identical (np.tile), checked here
    b = obs[:, 2 * N_FRAME:].reshape(n, 8, 42)
    assert np.allclose(b, b[:, :1]), "body copies differ"
    # prev eyes of decision k are the present eyes of k-1 (Game.step), checked here
    assert np.allclose(obs[1:, N_FRAME:2 * N_FRAME], obs[:-1, :N_FRAME]), "prev eyes are not the last decision's"
    (OUT / f"{name}.bin.gz").write_bytes(gzip.compress(b"".join(parts), 9, mtime=0))
    # frame sheets
    fr = z["dec_frames"]
    sheets = []
    for s in range(0, n, SHEET * SHEET):
        chunk = fr[s:s + SHEET * SHEET]
        img = np.zeros((SHEET * FH, SHEET * FW, 3), np.uint8)
        for i, f in enumerate(chunk):
            r, c = divmod(i, SHEET)
            img[r * FH:(r + 1) * FH, c * FW:(c + 1) * FW] = f
        rows = (len(chunk) + SHEET - 1) // SHEET
        fn = f"{name}_{len(sheets):02d}.webp"
        Image.fromarray(img[:rows * FH]).save(OUT / fn, quality=55, method=4)
        sheets.append(fn)
    pops = {k[4:]: z[k] for k in z.files if k.startswith("pop_")}
    a2m = z["a2m"]
    pref = a2m.argmax(1)
    order = np.lexsort((-a2m.max(1), pref))
    info = dict(
        meta, title=titles.get(name, name), n=n, acts=z["acts"].tolist(), xs=z["xs"].tolist(),
        steps=z["steps"].tolist(), flips=z["flips"].tolist(), body_flip=z["body_flip"].tolist(),
        n_patch=int(z["n_patch"]), sal_scale=scale, layout=layout, bin=f"{name}.bin.gz",
        sheets=sheets, sheet=SHEET, fw=FW, fh=FH, neurons=int(z["final"].shape[1]),
        regions={k: [int(v.min()), int(v.max()) + 1] for k, v in pops.items() if k != "motor_actions"},
        assoc_order=order.tolist(), assoc_pref=pref[order].tolist(),
    )
    (OUT / f"{name}.json").write_text(json.dumps(info, separators=(",", ":")))
    size = sum((OUT / f).stat().st_size for f in [f"{name}.json", f"{name}.bin.gz", *sheets])
    print(f"{name}: {n} decisions, {size / 1e6:.2f} MB ({meta['end']} at x {meta['max_x']})")
    return dict(name=name, title=info["title"], end=meta["end"], level=meta["level"], max_x=meta["max_x"], n=n)


TITLES = {
    "u3ck83-1-1-clear": "1-1 clear",
    "u3ck83-1-1-death": "1-1 death",
    "u3ck83-1-4-clear": "1-4 Bowser clear",
    "u3ck83-1-4-death": "1-4 death",
}


def main():
    sys.path.insert(0, str(VIZ.parent))
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [export(n, TITLES) for n in sys.argv[1:]]
    (OUT / "runs.json").write_text(json.dumps(runs, indent=1))
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"total docs/brain/data: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
