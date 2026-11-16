"""Can the coach get out of the brain's stall? (Harry 11/16, from Bernhard's Castlevania DAgger note: the teacher needed a
recovery rule at the spot where the brain used to stand.)

The brain plays greedy (as SEL8: mem._player greedy cold, Body.step, trainer's plug.worker_init smart) until it has gone
TAKE decisions without a new furthest x; then the coach (g.coach(), the smart coach DAgger labels with) plays on from that
exact state. Escape = the coach passes the stall's max x by ESC px, or reaches the flag, within CAP decisions.
If the coach also stays, DAgger at that spot teaches the stall and no brain change can fix it there.
    rescue.py <ckpt> <out.json> L:start [L:start ...]
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only"
os.environ.setdefault("MARIO_STUCK", "225")
HERE = Path(__file__).resolve().parent; MARIO = HERE.parent
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
TAKE, CAP, ESC = 150, 150, 48
NAMES = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]


def main():
    import cs, mem, plug
    mem.HORIZON = 250
    plug.worker_init(cs.deep_levels(), "smart", {})
    C, D = mem.C, mem.D
    assert D.STUCK == 225
    ck, outp, jobs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out = []
    for job in jobs:
        L, s = job.split(":"); s = int(s); t0 = time.time()
        pl = mem._player(ck, "greedy", cold=True, senses="fabius")
        lvl = tuple(int(v) for v in L.split("-"))
        body = C.CBody(s, "fabius", "lean", lvl); g = body.g
        pl.begin(0); acts = []
        while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < TAKE:
            a = pl.act(body.obs()); acts.append(a); body.step(a)
        rec = dict(level=L, start=s, brain_end=("flag" if g.flag else "died" if g.done else "stalled" if g.stuck >= TAKE else "cap"),
                   brain_max_x=int(g.max_x), x_at_take=int(g.x), decisions_at_take=int(g.decisions),
                   brain_held=NAMES[acts[-1]] if acts else None)
        if rec["brain_end"] == "stalled":
            stall_x = g.max_x; labels = []
            for _ in range(CAP):
                if g.done or g.flag or g.max_x >= stall_x + ESC:
                    break
                a = g.coach(); labels.append(a); body.step(a)
            rec.update(coach_decisions=len(labels), coach_max_x=int(g.max_x),
                       coach_end=("flag" if g.flag else "died" if g.done else "escaped" if g.max_x >= stall_x + ESC else "stayed"),
                       coach_first=NAMES[labels[0]] if labels else None,
                       coach_buttons={NAMES[k]: labels.count(k) for k in sorted(set(labels))})
        g.close(); rec["seconds"] = round(time.time() - t0)
        out.append(rec); print(json.dumps(rec), flush=True)
        outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
