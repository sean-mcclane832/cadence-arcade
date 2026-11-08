"""Where do the plateau brains end on a level? Plays each chunk-200 checkpoint greedy from several starts and
records how and where each play ended (x, y, the last screen), to find the waiting spot for the Fabius waiting test.
    python deathmap.py --level 1-2 --starts 0,5,11,17,23,29,37,43 --out deaths_1-2
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
MARIO = HERE.parent
for p in (MARIO / "coldstart", MARIO, MARIO / "memory", MARIO / "concepts"):
    sys.path.insert(0, str(p))
import cplay as C  # noqa: E402
import dagger as D  # noqa: E402
import mem  # noqa: E402
from game import ACTIONS  # noqa: E402

RUNS = ["CatoA", "CatoB", "CiceroA", "CiceroB", "LivyA", "LivyB", "CiceroLowA", "CiceroLowB"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--level", default="1-2")
    ap.add_argument("--starts", default="0,5,11,17,23,29,37,43"); ap.add_argument("--out", default="deaths_1-2")
    ap.add_argument("--ck", type=int, default=200)
    a = ap.parse_args()
    out = HERE / a.out; out.mkdir(exist_ok=True)
    lvl = tuple(int(v) for v in a.level.split("-")); res = []
    from PIL import Image
    for run in RUNS:
        p = mem.MemPlayer(str(MARIO / "coldstart" / "runs" / run / f"{run}_ck{a.ck:04d}.npz"), "greedy", 0.05, (), None)
        for s in (int(v) for v in a.starts.split(",")):
            b = C.CBody(s, "body", "lean", lvl); g = b.g
            p.begin(seed=1000)
            while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
                o = b.obs()
                if p.xkey:
                    o = np.concatenate([o, mem.xblock(g.env.ram)])
                act = p.act(o); g.step(act); b.held = ACTIONS[act]
            end = "flag" if g.flag else ("died" if g.done else "stuck")
            r = dict(run=run, start=s, end=end, x=int(g.x), max_x=int(g.max_x), y=int(g.env.ram[0xCE]), decisions=g.decisions)
            res.append(r); print(r, flush=True)
            if end != "flag":
                Image.fromarray(g.screen).save(out / f"{run}_s{s}_{end}_x{g.x}.png")
            g.close()
    (out / "ends.json").write_text(json.dumps(res))
    bins = Counter((r["end"], r["max_x"] // 50 * 50) for r in res if r["end"] != "flag")
    print("flags", sum(r["end"] == "flag" for r in res), "of", len(res))
    for (e, x), n in bins.most_common(12):
        print(f"{e:6s} x {x}-{x + 49}: {n}")


if __name__ == "__main__":
    main()
