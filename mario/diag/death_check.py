"""Where the brain dies late in 1-1: replay a stored solo attempt to its end and compare, for the
last N decisions, the brain's button with the coach's label (and weight), plus Mario's state."""
import sys, json
sys.path.insert(0, "..")
import numpy as np
from game import Game, ACTION_NAMES
run, rnd, start, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
h = json.load(open(f"../runs/{run}/history.json"))
e = next(x for x in next(r for r in h if r["round"] == rnd)["eval"] if x["start"] == start)
g = Game(noop_start=start, menu="lean", bg_per_frame=True)
acts = e["actions"]
for a in acts[:-N]:
    g.step(a)
for a in acts[-N:]:
    lab, w = g.coach_weighted()
    ram = g.env.ram
    print(f"x {g.x:5d} y {int(ram[0x00CE]):3d} fl {int(ram[0x001D])} vx {int(np.int8(np.uint8(ram[0x0057]))):3d}  brain {ACTION_NAMES[a]:10s} coach {ACTION_NAMES[lab]:10s} w{w:.0f} {'<-- differs' if a != lab else ''}")
    g.step(a)
print("end", g.x, "died" if g.done and not g.flag else "")
