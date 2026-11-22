"""Version parity (Romulus step 1): play one checkpoint greedy on SEL8 starts of a few levels and print max x,
end and a hash of the actions per start, plus the Cadence version. Run under each venv and diff the output.
    python vparity.py <ckpt.npz> 1-1,2-1
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "coldstart", HERE.parent, HERE.parent / "memory"):
    sys.path.insert(0, str(p))
import cs  # noqa: E402
import mem  # noqa: E402
import cadence  # noqa: E402

ck, levels = sys.argv[1], sys.argv[2].split(",")
print("cadence", cadence.__version__)
for L in levels:
    for s in cs.SEL8:
        pl = mem._player(ck, "greedy", senses="fabius")
        r = mem.play_attempt(pl, L, s)
        h = hashlib.sha256(bytes(int(a) for a in r["actions"])).hexdigest()[:12]
        print(L, s, r["max_x"], r["end"], r["decisions"], h, flush=True)
