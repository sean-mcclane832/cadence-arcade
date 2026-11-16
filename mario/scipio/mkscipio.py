"""Fresh Scipio brains on Cadence 0.99.0 (vault Plans/Scipio Design 2026-11-12, "The design, rewrite 2026-11-14").

Cato2's shape and teaching (fabius/mkfabius.py: Fabius eyes 3,536 inputs, 7 outputs, 1 x 256, recall off, normalized
teaching, no surprise gating, fast settle, carried gain 0.01), working trace amplitude 0, plus the memory of each arm:
  ScipioA  seed 0  history_decays=(0.97,) + efference copy amplitude 1.0 decay 0.97
  ScipioB  seed 1  same memory
  ScipioFast seed 0 history_decays=(0.8,) + efference copy amplitude 1.0 decay 0.8 (Cicero's time scale)
  Twin     seed 0  no history, no efference copy (the same brain without the memory)
Every file is checked after saving: reloaded, 3,536 inputs, 7 outputs, recall off, working trace amplitude 0,
and the memory parts as listed. No weights are written by hand.
    python mkscipio.py --out brains0
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fabius"))
import eyes  # noqa: E402

TEACH = dict(modules=(256,), teaching="normalized", learning_surprise=False, learning_fast_settle=True, carried_gain=0.01)
ARMS = {
    "ScipioA": dict(seed=0, history=0.97, efference=0.97),
    "ScipioB": dict(seed=1, history=0.97, efference=0.97),
    "ScipioFast": dict(seed=0, history=0.8, efference=0.8),
    "Twin": dict(seed=0, history=None, efference=None),
    # Marcellus (vault Plans/Marcellus Design 2026-11-16): Twin's brain; the change is in the labeller (--tie-stuck), not here
    "MarcellusA": dict(seed=0, history=None, efference=None),
    "MarcellusB": dict(seed=1, history=None, efference=None),
}


def compose(arm):
    from cadence import Brain
    s = ARMS[arm]
    mem = {}
    if s["history"] is not None:
        mem["history_decays"] = (s["history"],)
    if s["efference"] is not None:
        mem.update(efference_amplitude=1.0, efference_decay=s["efference"])
    b = Brain.compose(eyes.N_FABIUS, 7, seed=s["seed"], recall=False, working_memory_amplitude=0.0, **TEACH, **mem)
    return b


def check(arm, path):
    """The ck0 asserts (deep review fix 5c): reload and check the shape and the memory parts."""
    import cadence
    from cadence import Brain
    s = ARMS[arm]
    b = Brain.load(path)
    assert cadence.__version__ == "0.99.0", cadence.__version__
    assert len(b.sensory_index) == eyes.N_FABIUS == 3536, len(b.sensory_index)
    assert len(b.learner.output_index) == 7 and b.learner.slot_count == 1
    assert getattr(b, "recall", None) is None, "recall must be off"
    assert b.working_memory is not None and float(b.working_memory.amplitude) == 0.0, "working trace amplitude must be 0"
    if s["history"] is None:
        assert b.history is None
    else:
        assert b.history is not None and tuple(b.history.decays) == (s["history"],), b.history.decays
        assert float(b.history.amplitude) == 3.0
    if s["efference"] is None:
        assert b.efference is None
    else:
        assert b.efference is not None and abs(float(b.efference.decay) - s["efference"]) < 1e-12
        assert float(b.efference.amplitude) == 1.0
    return dict(arm=arm, seed=s["seed"], synapses=int(len(b.connectome.pre)), sensory=len(b.sensory_index),
                outputs=int(len(b.learner.output_index)), working_amplitude=float(b.working_memory.amplitude),
                history=None if b.history is None else list(b.history.decays),
                efference=None if b.efference is None else float(b.efference.decay),
                config=dataclasses.asdict(b.learner.config))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="brains0"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--arms", default="ScipioA,ScipioB,ScipioFast,Twin", help="arms to make (others are left untouched)")
    a = ap.parse_args()
    arms = a.arms.split(",")
    out = HERE / a.out; out.mkdir(exist_ok=True)
    rep = {}
    for arm in arms:
        p = out / f"{arm}_ck0.npz"
        if not a.check_only:
            compose(arm).save(p)
        rep[arm] = check(arm, p)
        print(arm, {k: v for k, v in rep[arm].items() if k != "config"}, flush=True)
    cfg = {arm: r.pop("config") for arm, r in rep.items()}
    diff = {k: {arm: cfg[arm][k] for arm in cfg} for k in cfg[arms[0]] if len({json.dumps(cfg[arm][k], default=str) for arm in cfg}) > 1}
    assert not diff, f"learner configs differ between arms: {diff}"
    import cadence
    rep["cadence"] = cadence.__version__
    (out / ("made.json" if arms == list(ARMS)[:4] else f"made_{'_'.join(arms)}.json")).write_text(json.dumps(rep, indent=1))
    print("ck0 asserts PASS for", arms, flush=True)


if __name__ == "__main__":
    main()
