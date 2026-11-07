"""Parity check for a cadence upgrade (issue #5): load saved Mario brains, play fixed greedy runs, and write
the action sequences, x and time per act to JSON. Run it under the old venv and the new one, then compare.
    python parity095.py OUT.json [--backend torch] [--precision float32] [--decisions 400]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "memory"))
sys.path.insert(0, str(HERE.parent / "concepts"))
import cplay as C  # noqa: E402
import mem  # noqa: E402
import cadence  # noqa: E402
from cadence import Brain  # noqa: E402
from game import ACTIONS  # noqa: E402

BRAINS = ["CatoA/CatoA_ck0200.npz", "CiceroA/CiceroA_ck0200.npz"]
LEVELS = ["1-1", "2-1", "3-4"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--backend", default=None)
    ap.add_argument("--precision", default=None); ap.add_argument("--decisions", type=int, default=400)
    a = ap.parse_args()
    res = dict(cadence=cadence.__version__, backend=a.backend, precision=a.precision, runs=[])
    for bp in BRAINS:
        path = str(HERE / "runs" / bp)
        p = mem.MemPlayer(path, "greedy", 0.05, (), None)
        if a.backend:  # reload onto the named backend, then the same cold-start patch MemPlayer applies
            kw = dict(backend=a.backend)
            if a.precision:
                kw["precision"] = a.precision
            p.b = mem.make_cold(Brain.load(path, **kw))
        res["placement"] = str(getattr(p.b, "placement", "n/a"))
        for L in LEVELS:
            b = C.CBody(0, "body", "lean", tuple(int(v) for v in L.split("-"))); g = b.g
            p.begin(seed=1000)
            acts, xs, secs = [], [], []
            while not g.done and g.decisions < a.decisions:
                o = b.obs()
                if p.xkey:
                    o = np.concatenate([o, mem.xblock(g.env.ram)])
                t0 = time.perf_counter(); act = p.act(o); secs.append(time.perf_counter() - t0)
                acts.append(int(act)); g.step(act); b.held = ACTIONS[act]
                xs.append(int(g.x))
            res["runs"].append(dict(brain=bp, level=L, acts=acts, max_x=max(xs), ms_per_act=1000 * float(np.median(secs[5:]))))
            print(f"{bp} {L}: {len(acts)} decisions, max x {max(xs)}, {res['runs'][-1]['ms_per_act']:.1f} ms/act", flush=True)
            g.close()
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
