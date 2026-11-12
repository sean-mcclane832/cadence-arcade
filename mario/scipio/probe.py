"""Leg 1, question 7 part B: a copied checkpoint acts on recorded coach rows (no game is played).
  idle.sh 095 probe.py <ck copy> <tag> [recall]
Plays like mem.MemPlayer: cold settles (b._activity -> None), reset at each attempt start, one greedy act per row,
the working trace carried from row to row. The coach's clean attempts are fed in order (teacher forced).
Writes scratch/probe_<tag>.json."""
import os, sys, time, json
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np
from cadence import Brain
FAB = "P:/projects/cadence-arcade/mario/coldstart/data/fab"
ACT = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
G = dict(periph_now=(0, 832), periph_prev=(832, 1664), body=(1664, 2000), fov_rgb=(2000, 3152), fov_prev=(3152, 3536))
ck, tag = sys.argv[1], sys.argv[2]
out = {"ck": ck}
def load(off=None):
    b = Brain.load(ck, backend="cpu"); b._activity = lambda: None
    if off: b.retune(**off)
    return b
def seqs(L, n_att):
    z = np.load(f"{FAB}/{L}/seq_{L}_coach.npz"); off = z["off"]
    lab = np.full(len(z["O"]), -1); lab[z["li"]] = z["y"]; w = np.zeros(len(z["O"])); w[z["li"]] = z["W"]
    return [(z["O"][off[k]:off[k + 1]], z["A"][off[k]:off[k + 1]], lab[off[k]:off[k + 1]], w[off[k]:off[k + 1]]) for k in range(n_att)]
def run(b, atts, T, edit=None):
    """atts as parallel streams; returns answers (streams, T)."""
    b.reset(); ans = np.zeros((len(atts), T), int)
    for t in range(T):
        x = np.stack([a[0][t] for a in atts]).astype(np.float64) / 255.0
        if edit is not None: x = edit(x)
        ans[:, t] = np.asarray(b.act(x, greedy=True)).reshape(len(atts), -1)[:, 0]
    return ans
def score(ans, atts, T):
    lab = np.stack([a[2][:T] for a in atts]); w = np.stack([a[3][:T] for a in atts]); prev = np.stack([np.concatenate([[-1], a[1][:T - 1]]) for a in atts])
    ok = ans == lab; ch = lab != prev
    r = dict(all=float(ok.mean()), critical=float(ok[w > 1].mean()), n=int(ok.size), label_changes_share=float(ch.mean()),
             when_label_repeats=float(ok[~ch].mean()), when_label_changes=float(ok[ch].mean()),
             answers={ACT[i]: int((ans == i).sum()) for i in range(7)}, copies_prev_when_wrong=float((ans[~ok] == prev[~ok]).mean()) if (~ok).any() else None)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
T = 150; LV = ["4-4", "1-2", "6-2"]
b = load(); t0 = time.time()
for L in LV:
    atts = seqs(L, 6); a = run(b, atts, T); out[f"{L}_on"] = score(a, atts, T)
    if L == "4-4":
        out["4-4_first12"] = dict(answers=a[0][:12].tolist(), labels=atts[0][2][:12].tolist(), start27=a[5][:12].tolist())
    print(L, out[f"{L}_on"], round(time.time() - t0), "s", flush=True)
# what each input group carries: zero it, same rows
atts = seqs("4-4", 6) + seqs("1-2", 6)
base = run(b, atts, 60); out["mix_base60"] = score(base, atts, 60)
for g, (lo, hi) in G.items():
    def ed(x, lo=lo, hi=hi):
        x = x.copy(); x[:, lo:hi] = 0; return x
    a = run(b, atts, 60, ed); s = score(a, atts, 60); s["answers_changed"] = round(float((a != base).mean()), 3); out[f"zero_{g}"] = s
    print("zero", g, s["all"], s["critical"], s["answers_changed"], flush=True)
def held_off(x):
    x = x.copy(); v = x[:, 1664:2000].reshape(len(x), 8, 42); v[:, :, 38:42] = 0; return x
a = run(b, atts, 60, held_off); s = score(a, atts, 60); s["answers_changed"] = round(float((a != base).mean()), 3); out["zero_held_buttons"] = s
# memory parts off
offs = {"trace_off": dict(working_memory_amplitude=0.0)}
if len(sys.argv) > 3: offs["recall_off"] = dict(recall_amplitude=0.0)
for name, tune in offs.items():
    b2 = load(tune); a = run(b2, atts, 60); s = score(a, atts, 60); s["answers_changed"] = round(float((a != base).mean()), 3); out[name] = s
    print(name, s["all"], s["critical"], s["answers_changed"], flush=True)
# from rest, no memories at all (what `fit`-style scoring on single rows would see)
X = np.concatenate([a[0][:60] for a in atts]).astype(np.float64) / 255.0
p = np.asarray(b.predict(X)).reshape(len(atts), 60, -1)[:, :, 0]; s = score(p, atts, 60); s["answers_changed"] = round(float((p != base).mean()), 3); out["predict_from_rest"] = s
out["seconds"] = round(time.time() - t0)
json.dump(out, open(f"scratch/probe_{tag}.json", "w"), indent=1); print(json.dumps(out)[:200])
