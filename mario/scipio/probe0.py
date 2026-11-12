"""Exploration: load a copied Cato2 checkpoint on the CPU, act on a few pile rows, time it, look at the state objects."""
import os, sys, time, json
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np
t0 = time.time()
import cadence
from cadence import Brain
print("cadence", cadence.__version__, "import s", round(time.time() - t0, 1), flush=True)
ck = sys.argv[1]
t = time.time(); b = Brain.load(ck, backend="cpu"); print("load s", round(time.time() - t, 1), "placement", getattr(b, "placement", None), flush=True)
orig = b._activity
b._activity = lambda: None
z = np.load("P:/projects/cadence-arcade/mario/coldstart/data/fab/4-4/seq_4-4_coach.npz")
O, A, off, li, y, W = (z[k] for k in ("O", "A", "off", "li", "y", "W"))
print("attempt offsets", off.tolist(), "labels at rows 0..9", y[:10].tolist(), "executed", A[:10].tolist(), "W", W[:10].tolist(), flush=True)
b.reset()
for i in range(6):
    x = O[i:i + 1].astype(np.float64) / 255.0
    t = time.time(); a = b.act(x, greedy=True); dt = time.time() - t
    st = np.asarray(b.basal_ganglia.state)
    print(i, "act", np.asarray(a).tolist(), "label", int(y[i]), "s", round(dt, 2), "bg.state", st.shape, np.round(st.reshape(-1)[:7], 3).tolist(), flush=True)
s = orig()
print("activity type", type(s).__name__, [n for n in dir(s) if not n.startswith("_")][:40])
for n in dir(s):
    if n.startswith("_"):
        continue
    v = getattr(s, n)
    if hasattr(v, "shape"):
        print("  ", n, getattr(v, "shape", None), getattr(v, "dtype", None))
print("brain attrs", [n for n in dir(b) if not n.startswith("_")][:120])
wm = b.working_memory
print("wm attrs", [n for n in dir(wm) if not n.startswith("_")])
t = time.time(); p = b.predict(O[:32].astype(np.float64) / 255.0); print("predict 32 rows s", round(time.time() - t, 2), np.asarray(p).tolist(), "labels", y[:32].tolist())
