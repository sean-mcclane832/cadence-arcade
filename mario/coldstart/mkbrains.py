"""Fresh (cold-start) connectomes for the night: Cato / Cicero / Livy specs (memory/mem.py SPECS), seeds A=0, B=1.
Nothing transplanted. Livy's echo synapses take 1/8 of the normal step (synapse_rate 0.125, MEMORY.md 14:20 fix).
Then a parity sanity check: every brain plays one act-mode attempt (cold settles, greedy) on 1-1 and 2-1 without errors.
    python mkbrains.py
    python mkbrains.py --arms CiceroLow   (11/5 plateau test: Cicero layout, working memory amplitude 0.3)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402
mem = cs.mem
SEEDS = {"A": 0, "B": 1}
ARMS = ["Cato", "Cicero", "Livy"]
# Arms defined here, not in ../memory/mem.py: (layout spec in mem.SPECS, play-time tune). CiceroLow = Cicero exactly,
# but working_memory_amplitude 0.3 instead of 3.0. Cadence docs/teach.md (0.92.1): in the upstream copy task the
# normalized point agreed with the demonstrator on 0.86 at amplitude 0 or 0.3 and 0.80 at the composed 3.0; "Applications
# that bootstrap by imitation may lower working_memory_amplitude." (training methods review 11/5, missing idea 3)
LOCAL = {"CiceroLow": ("Cicero", dict(working_memory_amplitude=0.3))}


def main():
    from cadence import Brain
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default=",".join(ARMS)); args = ap.parse_args()
    out = HERE / "brains0"; out.mkdir(exist_ok=True)
    cs.worker_init({})
    pf = out / "parity.json"
    rep = json.loads(pf.read_text()) if pf.exists() else {}
    for arm in args.arms.split(","):
        base, tune = LOCAL[arm] if arm in LOCAL else (arm, mem.SPECS[arm]["tune"])
        for tag, seed in SEEDS.items():
            name = f"{arm}{tag}"; p = out / f"{name}_ck0.npz"
            b = mem.compose(base, seed=seed)
            b.retune(**tune)
            if arm == "Livy":
                pre = np.asarray(b.connectome.pre); hist = np.asarray(b.connectome.populations["history"])
                r = np.ones(len(pre)); m = np.isin(pre, hist); r[m] = 0.125; b.learner.synapse_rate = r
            b.save(p)
            b2 = Brain.load(p)
            info = dict(seed=seed, synapses=int(len(b2.connectome.pre)), sensory=len(b2.sensory_index),
                        working_memory=b2.working_memory is not None, efference=b2.efference is not None,
                        history=b2.history is not None,
                        wm_amplitude=float(b2.working_memory.amplitude) if b2.working_memory is not None else None)
            if arm == "Livy":
                info["echo_rate"] = float(np.asarray(b2.learner.synapse_rate)[np.isin(np.asarray(b2.connectome.pre),
                                          np.asarray(b2.connectome.populations["history"]))].mean())
            for L in ("1-1", "2-1"):
                t = time.time()
                r = mem.episode(dict(policy=str(p), kind="act", level=L, start=0))
                info[L] = dict(max_x=r["max_x"], end=r["end"], decisions=r["decisions"], secs=round(time.time() - t, 1),
                               buttons=np.bincount(r["actions"], minlength=7).tolist())
            rep[name] = info
            print(name, info, flush=True)
    pf.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
