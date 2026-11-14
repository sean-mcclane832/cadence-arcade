"""Leg 2b: the playing parts of Phase 2 (worktree copy of the mario code, GPU hidden, Idle, <= 4 workers).

  idle.sh 095 playstudy.py ties                      held-out coach runs: are change frames timing ties?
  idle.sh 095 playstudy.py play <tag> <ck copy>      SEL8 on 24 levels: where attempts end
  idle.sh 095 playstudy.py label <tag> [starts]      coach scores on the last decisions of those attempts
  idle.sh 098 playstudy.py mlp <net.pt> <tag> --mode none|A [w7]
                                                     the MLP plays SEL8 (and World 7 first sight with w7). --mode is required
                                                     and stored in the result (leg 2c fix: the old "Aonly" token was silent).
                                                     Episodes are appended to scratch/play/mlp_<tag>.jsonl as they land; a rerun skips them.
World 8 is refused here as well as in coach/plug.py. Results: scratch/play/*.json
"""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MARIO = HERE.parent
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
OUT = HERE / "scratch" / "play"; OUT.mkdir(parents=True, exist_ok=True)
FAB = Path("P:/projects/cadence-arcade/mario/coldstart/data/fab")
LEVELS = [f"{w}-{l}" for w in range(1, 7) for l in (1, 2, 3, 4)]
W7 = [f"7-{l}" for l in (1, 2, 3, 4)]
SEL8 = (0, 48, 9, 57, 18, 66, 27, 75)
BTN = (0, 3, 4, 6)  # the four buttons the coach uses


def init():
    import plug
    plug.worker_init({}, "smart", {})


def lvl(L):
    t = tuple(int(v) for v in L.split("-")); assert t[0] != 8, "World 8 is never played"; return t


def per_button(g):
    sc = np.asarray(g.coach_scores(), float); first = np.array([p[0] for p in g.plans])
    return [float(sc[first == b].max()) if (first == b).any() else -1e18 for b in BTN]


def ties_task(L):
    import cplay as C
    z = np.load(FAB / L / f"seq_{L}_coach.npz"); off = z["off"]; A = z["A"][off[5]:off[6]]
    lab = np.full(len(z["O"]), -1); lab[z["li"]] = z["y"]; y = lab[off[5]:off[6]]
    b = C.CBody(27, "body", "lean", lvl(L)); g = b.g; rows = []; t0 = time.time()
    for t, a in enumerate(A):
        if g.done: break
        prev = int(A[t - 1]) if t else -1
        if t < 200 and (y[t] != prev or t % 4 == 0):
            rows.append(dict(t=t, y=int(y[t]), prev=prev, sc=per_button(g)))
        b.step(int(a))
    r = dict(level=L, flag=bool(g.flag), max_x=int(g.max_x), n=len(A), rows=rows, s=round(time.time() - t0)); g.close(); return r


def play_task(task):
    import mem
    r = mem.episode(dict(policy=task["ck"], kind="act", level=task["level"], start=task["start"], senses="fabius"))
    return dict(level=task["level"], start=task["start"], max_x=r["max_x"], end=r["end"], decisions=r["decisions"], actions=[int(a) for a in r["actions"]], xs=r["xs"])


def label_task(ep):
    """Replay one attempt's own buttons; coach scores on its last 16 decisions and on 8 decisions 40-47 before the end."""
    import cplay as C
    A = ep["actions"]; n = len(A); want = set(range(max(0, n - 16), n)) | set(range(max(0, n - 48), max(0, n - 40)))
    b = C.CBody(ep["start"], "body", "lean", lvl(ep["level"])); g = b.g; rows = []
    for t, a in enumerate(A):
        if g.done: break
        if t in want:
            rows.append(dict(t=t, back=n - 1 - t, a=int(a), prev=int(A[t - 1]) if t else -1, sc=per_button(g), x=int(g.x)))
        b.step(int(a))
    g.close()
    return dict(level=ep["level"], start=ep["start"], end=ep["end"], max_x=ep["max_x"], n=n, replay_max_x=int(g.max_x), rows=rows)


class MLPPlayer:
    senses = "fabius"; xkey = False
    def __init__(self, path, noheld=False):
        import torch
        torch.set_num_threads(1); self.t = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(3536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7)); self.net.load_state_dict(torch.load(path)); self.noheld = noheld
    def begin(self, seed=0): pass
    def end(self, o, outcome): pass
    def act(self, o):
        x = self.t.from_numpy((np.round(o * 255) / 255.0).astype(np.float32))[None].clone()
        if self.noheld == "A":      # keep "A held" only, first copy only
            v = x[:, 1664:2000].reshape(1, 8, 42); keep = v[:, 0, 38].clone(); v[:, :, 38:42] = 0; v[:, 0, 38] = keep
        with self.t.no_grad():
            return int(self.net(x).argmax(1)[0])


_M = {}
def mlp_task(task):
    import mem
    key = (task["net"], task.get("mode"))
    if key not in _M: _M[key] = MLPPlayer(task["net"], task.get("mode"))
    r = mem.play_attempt(_M[key], task["level"], task["start"])
    return dict(level=task["level"], start=task["start"], max_x=r["max_x"], end=r["end"], decisions=r["decisions"], actions=[int(a) for a in r["actions"]], xs=r["xs"])


def pmap(fn, tasks, workers=2):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init) as ex:
        return list(ex.map(fn, tasks))


def summ(res, levels):
    return dict(mean=round(float(np.mean([np.mean([r["max_x"] for r in res if r["level"] == L]) for L in levels])), 1),
                flags=sum(r["end"] == "flag" for r in res), ends={e: sum(r["end"] == e for r in res) for e in ("flag", "died", "stuck")},
                per_level={L: [round(float(np.mean([r["max_x"] for r in res if r["level"] == L]))), sum(r["end"] == "flag" for r in res if r["level"] == L)] for L in levels})


if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time()
    if mode == "ties":
        res = pmap(ties_task, LEVELS); json.dump(res, open(OUT / "ties.json", "w"))
        print("replays reached the flag on", sum(r["flag"] for r in res), "of 24")
    elif mode == "play":
        tag, ck = sys.argv[2], sys.argv[3]
        res = pmap(play_task, [dict(ck=ck, level=L, start=s) for L in LEVELS for s in SEL8])
        json.dump(dict(ck=ck, summary=summ(res, LEVELS), episodes=res), open(OUT / f"play_{tag}.json", "w")); print(json.dumps(summ(res, LEVELS)))
    elif mode == "label":
        tag = sys.argv[2]; starts = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 9, 18, 27]
        eps = [e for e in json.load(open(OUT / f"play_{tag}.json"))["episodes"] if e["start"] in starts and e["end"] != "flag"]
        res = pmap(label_task, eps); json.dump(res, open(OUT / f"label_{tag}.json", "w")); print("labelled", len(res), "attempts")
    elif mode == "mlp":
        net, tag = sys.argv[2], sys.argv[3]; levels = LEVELS + (W7 if "w7" in sys.argv else [])
        assert "Aonly" not in sys.argv, "use --mode A"
        m = sys.argv[sys.argv.index("--mode") + 1]; assert m in ("none", "A"), m
        stuck = int(os.environ.get("MARIO_STUCK", "-1")); assert stuck == 225, "set MARIO_STUCK=225"
        part = OUT / f"mlp_{tag}.jsonl"; res = [json.loads(l) for l in open(part)] if part.exists() else []
        done = {(r["level"], r["start"]) for r in res}
        tasks = [dict(net=net, level=L, start=s, mode=None if m == "none" else "A") for L in levels for s in SEL8 if (L, s) not in done]
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        with ProcessPoolExecutor(2, mp_context=get_context("spawn"), initializer=init) as ex, open(part, "a") as fh:
            for f in as_completed([ex.submit(mlp_task, t) for t in tasks]):
                r = f.result(); res.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
        s = summ([r for r in res if r["level"] in LEVELS], LEVELS); s7 = summ([r for r in res if r["level"] in W7], W7) if "w7" in sys.argv else None
        json.dump(dict(net=net, mode=m, stuck=stuck, summary=s, w7=s7, episodes=res), open(OUT / f"mlp_{tag}.json", "w")); print(json.dumps(s)); print(json.dumps(s7))
    print("seconds", round(time.time() - t0))
