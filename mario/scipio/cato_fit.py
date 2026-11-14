"""Leg 2c run C: agreement with the coach on the same frames, Cato2 brains and the all11dag MLPs.
    idle.sh 095 cato_fit.py cadence <ck copy> <tag>      (Cadence 0.95.0, the runs' own library)
    idle.sh 098 cato_fit.py mlp <net.pt> <tag>
Frames: coach run 5 (off[5]:off[6]) of each of the 24 levels, first 200 decisions, fed in order (teacher forced, as
probe.py: cold settles, one reset, one greedy act per row, the 24 levels as parallel streams). Every scorer was taught
these runs, so this is FIT ON TAUGHT FRAMES, never held-out. Split as mlpcopy.score: change = label != coach's previous
button (prev = A[t-1], -1 at t = 0). Unlabelled rows (y < 0) and padding past a run's end are masked out.
Writes scratch/fit/<tag>.json."""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
import numpy as np
from pathlib import Path
FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
LEVELS = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
T = 200
OUT = Path("scratch/fit"); OUT.mkdir(parents=True, exist_ok=True)


def frames():
    O, Y, P, M, lens = [], [], [], [], {}
    for L in LEVELS:
        z = np.load(FAB / L / f"seq_{L}_coach.npz"); off = z["off"]
        lab = np.full(len(z["O"]), -1); lab[z["li"]] = z["y"]
        o, a, y = z["O"][off[5]:off[6]], z["A"][off[5]:off[6]], lab[off[5]:off[6]]
        n = min(T, len(o)); lens[L] = int(len(o))
        prev = np.concatenate([[-1], a[:-1]])
        pad = T - n
        O.append(np.concatenate([o[:n], np.repeat(o[n - 1:n], pad, 0)]) if pad else o[:T])
        Y.append(np.concatenate([y[:n], np.full(pad, -1)])); P.append(np.concatenate([prev[:n], np.full(pad, -1)]))
        M.append(np.arange(T) < n)
    return np.stack(O), np.stack(Y), np.stack(P), np.stack(M) & (np.stack(Y) >= 0), lens


def score(ans, Y, P, M):
    ok = (ans == Y)[M]; c = (Y != P)[M]; hb = (ans == P)[M]
    per = {L: round(float((ans[i] == Y[i])[M[i]].mean()), 3) for i, L in enumerate(LEVELS)}
    return dict(all=round(float(ok.mean()), 4), repeat=round(float(ok[~c].mean()), 4), change=round(float(ok[c].mean()), 4),
                n=int(ok.size), n_change=int(c.sum()), answers_held_button=round(float(hb.mean()), 4), per_level=per)


if __name__ == "__main__":
    kind, src, tag = sys.argv[1], sys.argv[2], sys.argv[3]; t0 = time.time()
    O, Y, P, M, lens = frames(); out = dict(kind=kind, src=src, frames="coach run 5, first 200 decisions, taught frames", run5_lengths=lens)
    if kind == "cadence":
        from cadence import Brain
        b = Brain.load(src, backend="cpu"); b._activity = lambda: None

        def run(idx):
            b.reset(); ans = np.zeros((len(idx), T), int)
            for t in range(T):
                x = O[idx, t].astype(np.float64) / 255.0
                ans[:, t] = np.asarray(b.act(x, greedy=True)).reshape(len(idx), -1)[:, 0]
            return ans
        one = run(np.array([15]))  # 4-4 alone: check that the 24 batched streams do not interact
        print("single stream done", round(time.time() - t0), "s", flush=True)
        ans = run(np.arange(24))
        out["single_vs_batch_4-4_equal"] = bool((one[0] == ans[15]).all()); out["single_vs_batch_4-4_diff_frames"] = int((one[0] != ans[15]).sum())
    else:
        import torch
        torch.set_num_threads(1)
        net = torch.nn.Sequential(torch.nn.Linear(3536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7)); net.load_state_dict(torch.load(src))
        with torch.no_grad():
            ans = net(torch.from_numpy(O.reshape(-1, 3536).astype(np.float32) / 255.0)).argmax(1).numpy().reshape(24, T)
    out.update(score(ans, Y, P, M)); out["s"] = round(time.time() - t0)
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("per_level", "run5_lengths")}))
