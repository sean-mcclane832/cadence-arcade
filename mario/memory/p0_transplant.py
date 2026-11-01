"""Phase 0.1: transplant W1_F2ck110 into the four memory connectomes + predict parity on 3,000 stored screens."""
import sys, time, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mem
from cadence import Brain
quiet = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
X = []
for L in ("1-1", "1-2", "1-3", "1-4"):
    with np.load(HERE.parent / "concepts" / "runs" / "p4" / f"pile_{L}" / "data.npz") as z:
        X.append(z["X"][np.random.default_rng(0).choice(len(z["y"]), 750, replace=False), :2000])
X = np.concatenate(X).astype(np.float64) / 255
old = Brain.load(mem.START)
po = np.concatenate([np.asarray(old.predict(X[i:i + 500])).reshape(-1) for i in range(0, len(X), 500)])
rep = {}
for name in mem.SPECS:
    out = HERE / "brains0" / f"{name}_ck0.npz"
    info = mem.transplant(name, out, quiet=quiet)
    b = Brain.load(out)
    pn = np.concatenate([np.asarray(b.predict(X[i:i + 500])).reshape(-1) for i in range(0, len(X), 500)])
    info["predict_differ"] = int((pn != po).sum()); info["memories"] = list(b.memories)
    info["populations"] = {k: len(v) for k, v in b.connectome.populations.items()}
    rep[name] = info; print(name, info, flush=True)
mem.jdump(HERE / "runs" / "p0_transplant.json", rep)
