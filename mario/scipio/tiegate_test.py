"""Check of --tie-stuck (Marcellus): stall scripts labelled by cplay._play (the DAgger labeller) under the old rule (-1) and K=10.
4-2 start 48: a noop script (the start stall). 2-2 start 48: ScipioA ck50 played greedy to its stall, then its own actions as the script.
    tiegate_test.py
"""
import os, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
MARIO = Path(__file__).resolve().parents[1]
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
import numpy as np
import cs, mem, plug
mem.HORIZON = 250
plug.worker_init(cs.deep_levels(), "smart", {})
C = mem.C
N = ["noop", "right", "r+j", "run", "run+j", "jump", "left"]

idx = {0, 3, 6, 9, 12, 15, 30, 40}
for K in (-1, 10):
    out = C._play(dict(C.task("none", "body", "4-2", 48, mode="dagger"), script=[0] * 45, tiew=True, keep_obs=False, tie_stuck=K),
                  None, idx, np.random.default_rng(0))
    print("4-2 noop stall, tie_stuck", K, "labels at", sorted(idx), [N[v] for v in out["labels"]], flush=True)

ck = str(MARIO / "coldstart/runs/ScipioA/ScipioA_ck0050.npz")
pl = mem._player(ck, "greedy", cold=True, senses="fabius")
r = mem.play_attempt(pl, "2-2", 48, stuck=200)
acts = r["actions"]; n = len(acts)
print("2-2 start 48 greedy:", r["end"], "max_x", r["max_x"], "decisions", n, "held at end", N[acts[-1]], flush=True)
idx2 = set(range(n - 30, n, 2))
for K in (-1, 10):
    out = C._play(dict(C.task("none", "body", "2-2", 48, mode="dagger"), script=acts, tiew=True, keep_obs=False, tie_stuck=K),
                  None, idx2, np.random.default_rng(0))
    labs = [N[v] for v in out["labels"]]
    same = sum(l == N[acts[i]] for l, i in zip(labs, sorted(idx2)))
    print("2-2 stall tail, tie_stuck", K, "labels:", labs, f"(== held: {same}/{len(labs)})", flush=True)
