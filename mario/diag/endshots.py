"""Save screenshots of the last decisions of a solo attempt. python endshots.py BRAIN LEVEL START OUTPREFIX"""
import sys
sys.path.insert(0, "..")
import numpy as np
from PIL import Image
import dagger as D
from game import MENUS, Game, ACTION_NAMES
brain, level, start, pre = sys.argv[1], tuple(int(v) for v in sys.argv[2].split("-")), int(sys.argv[3]), sys.argv[4]
mb = sorted({a for p in MENUS["lean"] for a in p})
r = D.episode(dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb, level=level, mode="eval", start=start, policy=brain))
acts = r["actions"]; n = len(acts); print("end", r["end"], r["max_x"], n)
g = Game(noop_start=start, level=level, menu="lean", bg_per_frame=True)
shots = [n - 16, n - 8, n - 3]
row = []
for i, a in enumerate(acts):
    if i in shots:
        row.append(g.screen.copy()); print(i, "x", g.x, "brain", ACTION_NAMES[a], "coach", ACTION_NAMES[g.coach()])
    g.step(a)
Image.fromarray(np.concatenate(row, axis=1)).save(pre + ".png")
