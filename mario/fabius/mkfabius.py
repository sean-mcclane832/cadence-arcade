"""Fresh Fabius and Cato2 brains (vault Plans/Fabius Build 2026-11-08), Cadence 0.95 (.venv-095).

- FabiusA/B: Fabius's eyes (3,536 inputs, eyes.py), outputs 7 buttons + 104 prediction slots of 3 classes
  (less / same / more contrast at the next decision, pred.py), the 0.95 recall memory on.
- Cato2A/B: the control. Same eyes, 7 buttons, no slots, recall=False.
Both get the same teaching, named explicitly because 0.95's defaults change with recall on: normalized teaching,
no surprise gating (as Cato), fast settle and carried gain set alike. The script checks that the two learner
configurations agree in every field and prints any synapse rate difference.
Seeds A=0, B=1. --capacity K sets the recall memory's slots to the lesson burn-in K (deep review 11/8: lessons
write K slots, so play must not hold more than K; the 0.95 default is 128).
    python mkfabius.py --capacity 8
    python mkfabius.py --capacity 16 --out brains_k16      (tiny cost test)
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eyes  # noqa: E402
import pred  # noqa: E402

SEEDS = {"A": 0, "B": 1}
TEACH = dict(modules=(256,), teaching="normalized", learning_surprise=False, learning_fast_settle=True, carried_gain=0.01)


def compose(arm, seed, capacity):
    from cadence import Brain
    if arm == "Fabius":
        return Brain.compose(eyes.N_FABIUS, 7 + 3 * pred.N_BLOCKS, slots=[7] + [3] * pred.N_BLOCKS, seed=seed,
                             recall=True, recall_capacity=capacity, **TEACH)
    return Brain.compose(eyes.N_FABIUS, 7, seed=seed, recall=False, **TEACH)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--out", default="brains0"); a = ap.parse_args()
    out = HERE / a.out; out.mkdir(exist_ok=True)
    rep = {"capacity": a.capacity}
    for tag, seed in SEEDS.items():
        bs = {arm: compose(arm, seed, a.capacity) for arm in ("Fabius", "Cato2")}
        cf, cc = (dataclasses.asdict(bs[a].learner.config) for a in ("Fabius", "Cato2"))
        diff = {k: (cf[k], cc[k]) for k in cf if cf[k] != cc[k]}
        assert not diff, f"learner configs differ: {diff}"
        for arm, b in bs.items():
            p = out / f"{arm}{tag}_ck0.npz"; b.save(p)
            sr = getattr(b.learner, "synapse_rate", None)
            rep[f"{arm}{tag}"] = dict(seed=seed, synapses=int(len(b.connectome.pre)), sensory=len(b.sensory_index),
                                      outputs=int(len(b.learner.output_index)), slots=int(b.learner.slot_count),
                                      recall=getattr(b, "recall", None) is not None,
                                      capacity=getattr(getattr(b, "recall", None), "capacity", None),
                                      synapse_rate=None if sr is None else dict(min=float(np.min(sr)), max=float(np.max(sr)),
                                                                                mean=round(float(np.mean(sr)), 4)))
            print(f"{arm}{tag}", rep[f"{arm}{tag}"], flush=True)
    (out / "made.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
