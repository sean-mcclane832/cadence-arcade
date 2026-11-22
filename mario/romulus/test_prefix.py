"""Romulus wall-start tests (11/21). Brain: a checkpoint, level and start from argv.
1. prefix=None and prefix=[] give the same attempt (actions, xs, end) as before.
2. A prefix cut from that attempt at n-45 replays to the same x at every replayed decision, and the attempt then continues;
   when the brain is greedy and memoryless, the whole attempt equals the original.
3. mem.episode with label=True on a prefixed task labels only decisions >= len(prefix).
    python test_prefix.py <ckpt.npz> 1-2 0
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "coldstart", HERE.parent, HERE.parent / "memory"):
    sys.path.insert(0, str(p))
import mem  # noqa: E402

ck, L, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
def play(prefix):
    return mem.play_attempt(mem._player(ck, "greedy", senses="fabius"), L, s, prefix=prefix)
a = play(None); b = play([])
assert (a["actions"], a["xs"], a["end"]) == (b["actions"], b["xs"], b["end"]), "empty prefix changed the attempt"
n = len(a["actions"]); cut = max(0, n - 45)
c = play(a["actions"][:cut])
assert c["xs"][:cut] == a["xs"][:cut], "replayed prefix diverged"
assert c["prefix"] == cut
same = (c["actions"], c["xs"], c["end"]) == (a["actions"], a["xs"], a["end"])
print(f"1 ok; 2 ok: cut {cut}/{n}, x at cut {a['xs'][cut-1] if cut else 0}, continuation identical to original: {same}")
t = dict(policy=ck, kind="act", level=L, start=s, keep_obs=True, label=True, seed=7, senses="fabius", last_all=100,
         tie_stuck=10, prefix=a["actions"][:cut])
r = mem.episode(t)
assert len(r["lab_idx"]) and min(r["lab_idx"]) >= cut, (cut, r["lab_idx"][:5])
print(f"3 ok: {len(r['labels'])} labels, first at {min(r['lab_idx'])} >= {cut}; end {r['end']} x {r['max_x']}")
