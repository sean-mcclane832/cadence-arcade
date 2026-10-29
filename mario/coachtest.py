"""Can the look-ahead coach clear a level? python coachtest.py 1-2 lean [start]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import Game

lv = tuple(int(v) for v in sys.argv[1].split("-")); menu = sys.argv[2]
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
g = Game(noop_start=start, level=lv, menu=menu)
t = time.time(); xs = []; acts = []
if len(sys.argv) > 4:  # replay a stored coach prefix first: coachtest.py 1-2 lean24 0 runs/levels/coach_1-2_lean_0.json 200
    acts = json.loads(Path(sys.argv[4]).read_text())["actions"][:int(sys.argv[5])]
    for a in acts:
        g.step(a)
while not g.done and g.decisions < 1500 and g.stuck < 80:
    a, w = g.coach_weighted(); acts.append(a); g.step(a); xs.append(g.x)
print(json.dumps(dict(level=sys.argv[1], menu=menu, start=start, flag=g.flag, max_x=g.max_x, last_x=g.x,
                      decisions=g.decisions, seconds=round(time.time() - t), bg=g.bg,
                      time_left=int(g.env._time), x_trace=xs[::25])), flush=True)
Path(__file__).resolve().parent.joinpath(f"runs/levels/coach_{sys.argv[1]}_{menu}_{start}{'_p' if len(sys.argv) > 4 else ''}.json").write_text(
    json.dumps(dict(actions=acts, flag=g.flag, max_x=g.max_x)))
