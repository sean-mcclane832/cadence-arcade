"""Where the coach itself fails: coach plays alone, then screenshots of its last decisions. python coachshots.py LEVEL START OUT"""
import sys
sys.path.insert(0, "..")
import numpy as np
from PIL import Image
import dagger as D
from game import MENUS, Game, ACTION_NAMES
level, start, pre = tuple(int(v) for v in sys.argv[1].split("-")), int(sys.argv[2]), sys.argv[3]
mb = sorted({a for p in MENUS["lean"] for a in p})
r = D.episode(dict(senses="body", menu="lean", kind="cadence", menu_buttons=mb, level=level, mode="coach", start=start, eps=0.0, seed=0))
acts = r["actions"]; n = len(acts); print("end", r["end"], r["max_x"], n)
g = Game(noop_start=start, level=level, menu="lean", bg_per_frame=True)
shots = [n - 12, n - 6, n - 2]; row = []
for i, a in enumerate(acts):
    if i in shots:
        row.append(g.screen.copy()); print(i, "x", g.x, ACTION_NAMES[a], np.round(g.coach_scores()).astype(int).tolist())
    g.step(a)
Image.fromarray(np.concatenate(row, axis=1)).save(pre + ".png")
