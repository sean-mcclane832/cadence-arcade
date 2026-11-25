"""Fixed wall starts (Romulus design, scoring 4.4; 11/21). Per level, two walls: the top-2 "ends near" x of the six
chunk-100-era brains (research note section 1, walls.py). A wall start replays one of the coach's 6 clean runs
(coachcollect RECIPE starts 0,3,8,13,20,27) up to its first decision at x >= wall - 150, then the brain plays on.
So every brain, old or new, meets the same 12 starts per level.
    python wallsel.py build                      -> walls_fixed.json (prefixes; run once, any venv)
    python wallsel.py score <ckpt.npz> [out.json] -> per level: wall passes (max_x > wall + 100) and clears, of 12
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "coldstart", HERE.parent, HERE.parent / "memory"):
    sys.path.insert(0, str(p))
import mem  # noqa: E402

WALLS = {"1-1": (700, 1100), "1-2": (900, 200), "1-3": (700, 600), "1-4": (1200, 2200), "2-1": (300, 700),
         "2-2": (800, 1000), "2-3": (500, 700), "2-4": (1500, 1700), "3-1": (600, 500), "3-2": (2000, 400),
         "3-3": (1500, 600), "3-4": (400, 1400), "4-1": (3600, 1300), "4-2": (300, 700), "4-3": (600, 400),
         "4-4": (0, 600), "5-1": (700, 200), "5-2": (200, 700), "5-3": (600, 700), "5-4": (1100, 1500),
         "6-1": (1400, 500), "6-2": (700, 300), "6-3": (1000, 900), "6-4": (400, 600)}  # walls.py, 11/21
CLEAN = (0, 3, 8, 13, 20, 27)
LEVELS = json.loads((HERE.parent / "scipio" / "levels-fab.json").read_text())
FIXED = HERE / "walls_fixed.json"


class Idle:
    senses = "fabius"
    def begin(self, seed): pass
    def act(self, o): return 0
    def end(self, o, e): pass


def coach_runs(L):
    z = np.load(Path(LEVELS[L]) / f"seq_{L}_coach.npz"); off = z["off"]
    return [z["A"][off[k]:off[k + 1]] for k in range(len(off) - 1)]


def build():
    """11/25: a straight replay of the smart coach's actions does not always reach its flag (1-2 starts 3 and 13 match
    the stored screens frame for frame up to x 850 / 501, then drift; the coach's own lookahead leaves some state a fresh
    replay does not have). So each prefix is checked screen by screen against the stored coach rows up to its cut, and a
    start whose replay drifts before the cut is dropped (logged in walls_fixed.json "dropped"). Same starts for every brain."""
    out, dropped = {}, {}
    for L, walls in WALLS.items():
        z = np.load(Path(LEVELS[L]) / f"seq_{L}_coach.npz"); off = z["off"]
        out[L] = []
        reps = []
        for k, s in enumerate(CLEAN):  # RECIPE order: the 6 clean runs come first
            A = z["A"][off[k]:off[k + 1]]
            r = mem.play_attempt(Idle(), L, s, prefix=[int(a) for a in A], keep_obs=True)
            ref = z["O"][off[k]:off[k + 1]]; ob = r["obs"]
            n = min(len(ob), len(ref))
            d0 = next((i for i in range(n) if not np.array_equal(ob[i], ref[i])), n)  # first drifted row (n = none)
            reps.append((s, A, r, d0))
        for w in walls:
            for s, A, r, d0 in reps:
                if w - 150 <= r["xs"][0]:  # a wall at the start (4-4: its failure is the start stall): a plain level start
                    m = 0
                else:
                    m = next((i for i, x in enumerate(r["xs"][:d0]) if x >= w - 150), None)
                if m is None:
                    dropped.setdefault(L, []).append(dict(wall=w, start=s, drift_row=d0, drift_x=int(r["xs"][min(d0, len(r["xs"]) - 1)])))
                    continue
                out[L].append(dict(wall=w, start=s, prefix=[int(a) for a in A[:m]], x0=int(r["xs"][m]) if m else 0))
        print(L, [(d["wall"], d["start"], len(d["prefix"]), d["x0"]) for d in out[L]], "dropped", dropped.get(L), flush=True)
    out["dropped"] = dropped
    FIXED.write_text(json.dumps(out))


def score(ck, dst=None):
    fixed = json.loads(FIXED.read_text()); fixed.pop("dropped", None); res = {}
    for L, starts in fixed.items():
        rows = []
        for d in starts:
            r = mem.play_attempt(mem._player(ck, "greedy", senses="fabius"), L, d["start"], prefix=d["prefix"])
            rows.append(dict(wall=d["wall"], start=d["start"], x=r["max_x"], end=r["end"], passed=r["max_x"] > d["wall"] + 100,
                             flag=r["flag"]))
        res[L] = dict(passed=sum(r["passed"] for r in rows), flags=sum(r["flag"] for r in rows), n=len(rows), rows=rows)
        print(L, res[L]["passed"], res[L]["flags"], len(rows), flush=True)
    tot = {k: sum(v[k] for v in res.values()) for k in ("passed", "flags", "n")}
    print("TOTAL", tot)
    if dst:
        Path(dst).write_text(json.dumps(dict(ckpt=str(ck), total=tot, levels=res)))


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
