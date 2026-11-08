"""Split brains for the tiny test (11/8: Fabius collapsed to the majority button, Cato2 learned): which part does it?
SlotsA = Fabius's slots, recall off. RecallA = recall (capacity 8), 7 buttons only. Same teaching as mkfabius.
    python split.py
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eyes, pred  # noqa: E402
from mkfabius import TEACH  # noqa: E402
from cadence import Brain  # noqa: E402
out = HERE / "brains_k8"
Brain.compose(eyes.N_FABIUS, 7 + 3 * pred.N_BLOCKS, slots=[7] + [3] * pred.N_BLOCKS, seed=0, recall=False, **TEACH).save(out / "SlotsA_ck0.npz")
Brain.compose(eyes.N_FABIUS, 7, seed=0, recall=True, recall_capacity=8, **TEACH).save(out / "RecallA_ck0.npz")
print("ok")
