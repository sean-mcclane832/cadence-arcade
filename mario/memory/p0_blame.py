"""Phase 0.2 + 0.3: does Cadence's own death write reach the decision that mattered? Key overlap. Records-only check.
For each start: Agrippa plays ONE retry life (sampled at temp, learn(), actor rates 0). After a death:
  - decisive decision = the last decision before death where the coach's label was CRITICAL (weight 10) and the
    brain's action differed (coach replay of the brain's own buttons, tie-aware);
  - vote of the place records for the taken action at k = 0..30 decisions before death;
  - motor margin (top - second activation) there, to see if the vote could change the act;
  - key overlap: cosine of separated place keys vs x distance (1, 5, 20 tiles);
  - weights byte-identical before / after the life (records-only).
    python p0_blame.py --brain brains0/Agrippa_ck0.npz --levels 1-2,2-1 --out runs/p0_blame.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mem  # noqa: E402
C = mem.C
KS = (0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30)


def whash(b):
    g = b.learner.brain
    h = hashlib.sha256()
    for arr in (g.efficacy, g.bias, g.log_gain, b.basal_ganglia.w_critic if hasattr(b.basal_ganglia, "w_critic") else np.zeros(1)):
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def one(task):
    path, level, start, temp = task["brain"], task["level"], task["start"], task["temp"]
    pl = mem.MemPlayer(path, "retry", temp)
    before = whash(pl.b)
    margins, pmax, obs = [], [], []

    def cb(g, o, a):
        st = pl.b.basal_ganglia.state
        s = np.asarray(st.activation)[0, pl.b.learner.output_index]
        srt = np.sort(s); margins.append(float(srt[-1] - srt[-2]))
        p = pl.b.basal_ganglia.probabilities(st, temp); pmax.append(float(p.max()))
        obs.append(o.copy())
    r = mem.play_attempt(pl, level, start, seed=1000 * start, cb=cb)
    after = whash(pl.b)
    out = dict(start=start, level=level, end=r["end"], x=r["max_x"], decisions=r["decisions"], weights_same=before == after,
               hash_before=before, hash_after=after, dev_rate=round(float(np.mean(1 - np.array(pmax))), 4),
               writes=int(pl.ep.place.writes) if hasattr(pl.ep.place, "writes") else None)
    if r["end"] != "died":
        return out
    n = len(r["actions"]); acts = r["actions"]; xs = r["xs"]
    # coach replay of its own buttons, labels on the last 40 decisions
    idx = set(range(max(0, n - 40), n))
    lab = C._play(dict(C.task("none", "body", level, start + mem.RETRY[0], mode="dagger"), script=acts, tiew=True,
                       keep_obs=False), None, idx, np.random.default_rng(0))
    order = sorted(idx)[:len(lab["labels"])]
    crit = [(i, int(l), float(w)) for i, l, w in zip(order, lab["labels"], lab["weights"])]
    decisive = [i for i, l, w in crit if w > 1 and l != acts[i]]
    kstar = (n - 1 - decisive[-1]) if decisive else None
    votes = {}
    for k in KS:
        i = n - 1 - k
        if i < 0:
            continue
        v = pl.vote(obs[i])
        votes[k] = dict(taken=round(float(v[acts[i]]), 4), minv=round(float(v.min()), 4), margin=round(margins[i], 4),
                        x=xs[i])
    if kstar is not None:
        i = n - 1 - kstar; v = pl.vote(obs[i])
        out["kstar_vote"] = round(float(v[acts[i]]), 4); out["kstar_margin"] = round(margins[i], 4)
    # key overlap vs x distance (separated place keys)
    sep = pl.ep.place.separator
    codes = sep.code(np.array(obs, dtype=np.float64))
    codes = codes / np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-12)
    xa = np.array(xs)
    ov = {}
    for tiles in (1, 5, 20):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 120)) if abs(abs(xa[j] - xa[i]) - 16 * tiles) <= 4]
        if pairs:
            c = [float(codes[i] @ codes[j]) for i, j in pairs[:400]]
            ov[tiles] = dict(mean=round(float(np.mean(c)), 3), n=len(c))
    nb = [float(codes[i] @ codes[i + 1]) for i in range(n - 1)]
    ov["next_decision"] = round(float(np.mean(nb)), 3)
    out.update(kstar=kstar, decisive=decisive[-3:], votes=votes, overlap=ov,
               crit_last40=[(n - 1 - i, l, acts[i]) for i, l, w in crit if w > 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brains0/Agrippa_ck0.npz"); ap.add_argument("--levels", default="1-2,2-1")
    ap.add_argument("--starts", default="32,35,38,41,80,85,90,95"); ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = [dict(brain=str((HERE / a.brain).resolve()), level=L, start=int(s), temp=a.temp)
             for L in a.levels.split(",") for s in a.starts.split(",")]
    t = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=mem.worker_init) as ex:
        res = list(ex.map(one, tasks, chunksize=1))
    for r in res:
        print(r["level"], r["start"], r["end"], r["x"], "weights_same", r["weights_same"], "dev", r["dev_rate"],
              "k*", r.get("kstar"), "vote@k*", r.get("kstar_vote"), "margin@k*", r.get("kstar_margin"),
              "v0", r.get("votes", {}).get(0), "v5", r.get("votes", {}).get(5), "ov", r.get("overlap"), flush=True)
    mem.jdump((HERE / a.out).resolve(), dict(results=res, temp=a.temp, seconds=round(time.time() - t)))


if __name__ == "__main__":
    main()
