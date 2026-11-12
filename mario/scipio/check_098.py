"""Leg 1 check: a tiny brain composes and takes one lesson on .venv-098 with the GPU hidden.
Run: CUDA_VISIBLE_DEVICES=-1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python check_098.py
"""
import os, time, json
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "hide the GPU"
import numpy as np
import cadence, torch
from cadence import Brain
out = {"cadence": cadence.__version__, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
       "numpy": np.__version__}
rng = np.random.default_rng(0)
t = time.time(); b = Brain.compose(64, 7, seed=0, modules=(32,), recall=False, teaching="normalized",
                                    learning_surprise=False); out["compose_s"] = round(time.time() - t, 2)
out["synapses"] = int(len(b.connectome.pre)); out["sensory"] = int(len(b.sensory_index))
X = rng.random((32, 64)); y = rng.integers(0, 7, 32)
e0 = np.array(b.efficacy, copy=True) if hasattr(b, "efficacy") else None
t = time.time(); r = b.lesson(X, y); out["lesson_s"] = round(time.time() - t, 2)
out["lesson_return"] = repr(r)[:200]
if e0 is not None:
    out["efficacy_changed"] = int(np.sum(np.array(b.efficacy) != e0))
p = b.predict(X); out["predict_shape"] = list(np.shape(p))
out["backend"] = repr(getattr(b, "backend", None))[:80]
print(json.dumps(out, indent=1))
