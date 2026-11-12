"""Print the structure of a copied checkpoint (keys, shapes, meta, brain layout). numpy only."""
import sys, json, numpy as np
p = sys.argv[1]
z = np.load(p, allow_pickle=False)
def short(o, d=0):
    if isinstance(o, dict):
        return {k: short(v, d + 1) for k, v in o.items()}
    if isinstance(o, list):
        return f"list[{len(o)}] {o[:3]}..{o[-1:]}" if len(o) > 8 else o
    return o
for k in z.files:
    a = z[k]
    if a.dtype.kind in "US" or a.ndim == 0:
        s = a.item() if a.ndim == 0 else str(a)
        try:
            s = json.dumps(short(json.loads(s)))
        except Exception:
            s = str(s)
        print(f"{k}: {a.dtype} {a.shape} :: {s[:6000]}")
    elif a.size == 0:
        print(f"{k}: {a.dtype} {a.shape} EMPTY")
    else:
        print(f"{k}: {a.dtype} {a.shape} min {a.min():.4g} max {a.max():.4g} mean {a.mean():.4g} nonzero {int(np.count_nonzero(a))}")
