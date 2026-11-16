"""Is the motor readout on a rail where the brain stalls? (#296's check, on lessons-trained Scipio.) Greedy SEL8 play;
per decision the 7 motor activations after the settle (basal_ganglia.state.activation[:, motor_index]).
    rail.py <ckpt> L:start:n ...   (first n decisions of each attempt)"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
HERE = Path(__file__).resolve().parent; MARIO = HERE.parent
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
NAMES = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]


def main():
    import cs, mem, plug
    mem.HORIZON = 250; plug.worker_init(cs.deep_levels(), "smart", {})
    C = mem.C
    ck = sys.argv[1]; out = {}
    for job in sys.argv[2:]:
        L, s, n = job.split(":"); s, n = int(s), int(n)
        pl = mem._player(ck, "greedy", cold=True, senses="fabius"); b = pl.b; mi = b.motor_index
        body = C.CBody(s, "fabius", "lean", tuple(int(v) for v in L.split("-"))); g = body.g
        pl.begin(0); rows = []
        for t in range(n):
            if g.done:
                break
            a = pl.act(body.obs())
            m = np.asarray(b.basal_ganglia.state.activation)[0, mi]
            lab = g.coach() if t in (0, 5, 20, 60) else None
            rows.append(dict(t=t, x=int(g.x), a=NAMES[a], motor=[round(float(v), 3) for v in m], coach=NAMES[lab] if lab is not None else None))
            body.step(a)
        g.close(); out[job] = rows
        for r in rows:
            if r["t"] in (0, 1, 5, 20, 60, 120):
                print(job, r, flush=True)
    Path(HERE / "scratch/rail" / (Path(ck).stem + ".json")).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
