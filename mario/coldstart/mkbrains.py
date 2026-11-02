"""Fresh (cold-start) connectomes for the night: Cato / Cicero / Livy specs (memory/mem.py SPECS), seeds A=0, B=1.
Nothing transplanted. Livy's echo synapses take 1/8 of the normal step (synapse_rate 0.125, MEMORY.md 14:20 fix).
Then a parity sanity check: every brain plays one act-mode attempt (cold settles, greedy) on 1-1 and 2-1 without errors.
    python mkbrains.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs  # noqa: E402
mem = cs.mem
SEEDS = {"A": 0, "B": 1}
ARMS = ["Cato", "Cicero", "Livy"]


def main():
    from cadence import Brain
    out = HERE / "brains0"; out.mkdir(exist_ok=True)
    cs.worker_init({})
    rep = {}
    for arm in ARMS:
        for tag, seed in SEEDS.items():
            name = f"{arm}{tag}"; p = out / f"{name}_ck0.npz"
            b = mem.compose(arm, seed=seed)
            b.retune(**mem.SPECS[arm]["tune"])
            if arm == "Livy":
                pre = np.asarray(b.connectome.pre); hist = np.asarray(b.connectome.populations["history"])
                r = np.ones(len(pre)); m = np.isin(pre, hist); r[m] = 0.125; b.learner.synapse_rate = r
            b.save(p)
            b2 = Brain.load(p)
            info = dict(seed=seed, synapses=int(len(b2.connectome.pre)), sensory=len(b2.sensory_index),
                        working_memory=b2.working_memory is not None, efference=b2.efference is not None,
                        history=b2.history is not None)
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
    (out / "parity.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
