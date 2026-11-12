"""Part B, second probe: (1) blank the whole screen, keep the body sense; (2) the first decision of every level
with nothing held (as in play) and with run held (B + right) written into the body sense: does the held button alone
flip the answer? (3) own-loop on a frozen first frame: feed the brain's own last button back as 'held', 20 times.
  idle.sh 095 probe2.py <ck copy> <tag>"""
import os, sys, json
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np
from cadence import Brain
FAB = "P:/projects/cadence-arcade/mario/coldstart/data/fab"
ACT = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
HELD = {0: (0, 0, 0, 0), 3: (0, 1, 1, 0), 4: (1, 1, 1, 0), 6: (0, 0, 0, 1), 1: (0, 0, 1, 0), 2: (1, 0, 1, 0), 5: (1, 0, 0, 0)}  # A, B, right, left
ck, tag = sys.argv[1], sys.argv[2]
b = Brain.load(ck, backend="cpu"); b._activity = lambda: None
def set_held(x, a):
    x = x.copy(); v = x[:, 1664:2000].reshape(len(x), 8, 42); v[:, :, 38:42] = np.array(HELD[a], float); return x
levels = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
out = {"first": {}}
rows, labs, seqrows, seqlab, seqprev = [], [], [], [], []
for L in levels:
    z = np.load(f"{FAB}/{L}/seq_{L}_coach.npz"); off = z["off"]
    rows.append(z["O"][0]); labs.append(int(z["y"][0]))
    lab = np.full(len(z["O"]), -1); lab[z["li"]] = z["y"]
    for k in range(6):
        seqrows.append(z["O"][off[k]:off[k] + 40]); seqlab.append(lab[off[k]:off[k] + 40]); seqprev.append(np.concatenate([[-1], z["A"][off[k]:off[k] + 39]]))
X0 = np.stack(rows).astype(np.float64) / 255.0
b.reset(); a_none = np.asarray(b.act(X0, greedy=True)).reshape(24, -1)[:, 0]
b.reset(); a_run = np.asarray(b.act(set_held(X0, 3), greedy=True)).reshape(24, -1)[:, 0]
# own loop on the frozen first frame
x = X0.copy(); loop = []
b.reset()
for t in range(20):
    a = np.asarray(b.act(x, greedy=True)).reshape(24, -1)[:, 0]; loop.append(a.copy())
    x = np.stack([set_held(X0[i:i + 1], int(a[i]))[0] for i in range(24)])
loop = np.stack(loop, 1)
for i, L in enumerate(levels):
    out["first"][L] = dict(coach=ACT[labs[i]], nothing_held=ACT[a_none[i]], run_held=ACT[a_run[i]], own_loop_20="".join(str(int(v)) for v in loop[i]))
out["first_summary"] = dict(coach_run_family=int(sum(l in (3, 4) for l in labs)), brain_noop_nothing_held=int((a_none == 0).sum()),
                            brain_right_nothing_held=int(np.isin(a_none, (3, 4)).sum()), brain_noop_run_held=int((a_run == 0).sum()),
                            stuck_noop_after_20=int((loop[:, -5:] == 0).all(1).sum()), stuck_levels=[levels[i] for i in range(24) if (loop[i, -5:] == 0).all()])
# blank screen, 24 levels x 6 attempts x 40 rows, teacher forced
S = len(seqrows); lab = np.stack(seqlab); prev = np.stack(seqprev)
def run(edit):
    b.reset(); ans = np.zeros((S, 40), int)
    for t in range(40):
        x = np.stack([r[t] for r in seqrows]).astype(np.float64) / 255.0
        ans[:, t] = np.asarray(b.act(edit(x), greedy=True)).reshape(S, -1)[:, 0]
    return ans
def sc(a, base=None):
    ok = a == lab; ch = lab != prev
    return dict(all=round(float(ok.mean()), 3), repeats=round(float(ok[~ch].mean()), 3), changes=round(float(ok[ch].mean()), 3), n=int(ok.size),
                change_share=round(float(ch.mean()), 3), equals_prev=round(float((a == prev).mean()), 3),
                answers_changed=None if base is None else round(float((a != base).mean()), 3))
base = run(lambda x: x); out["all_levels_base"] = sc(base)
def blank(x):
    x = x.copy(); x[:, :1664] = 0; x[:, 2000:] = 0; return x
def held_only(x):
    y = np.zeros_like(x); v = x[:, 1664:2000].reshape(len(x), 8, 42); w = y[:, 1664:2000].reshape(len(x), 8, 42); w[:, :, 38:42] = v[:, :, 38:42]; return y
def no_held(x):
    x = x.copy(); x[:, 1664:2000].reshape(len(x), 8, 42)[:, :, 38:42] = 0; return x
out["blank_screen_keep_body"] = sc(run(blank), base); out["held_buttons_only"] = sc(run(held_only), base); out["no_held_buttons"] = sc(run(no_held), base)
out["label_equals_prev_in_data"] = round(float((lab == prev).mean()), 3)
json.dump(out, open(f"scratch/probe2_{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "first"}, indent=0)); [print(L, v) for L, v in out["first"].items()]
