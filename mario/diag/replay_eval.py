"""H2: where and how do solo attempts fail? Replays stored eval actions, logs the end of each."""
import sys, json, numpy as np
sys.path.insert(0, "..")
from game import Game, ACTION_NAMES
h = json.load(open(sys.argv[1]))
rows = h if isinstance(h, list) else h["history"]
out = []
for rec in rows[-12:]:
    for e in rec.get("eval", [])[:5]:
        g = Game(noop_start=e["start"])
        trace = []
        for a in e["actions"]:
            g.step(a)
            trace.append((g.x, int(g.env.ram[0x001D]), ACTION_NAMES[a][:5]))
        end = "flag" if g.flag else ("died" if g.done else "stuck/timeout")
        out.append((rec["round"], e["start"], e["max_x"], end, len(e["actions"])))
        if end != "died":
            # last 20 decisions: x, float state (0 ground,1 jump,2 fall), action
            print(rec["round"], e["start"], e["max_x"], end, trace[-25:-5:2])
        g.close()
for o in out: print(o)
