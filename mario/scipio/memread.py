"""Scipio memory weights, ck0 against ckN (chunk-25 sanity rule and the 25/50 log item, registered 11/14).

For each projection into the association region: relative norm change ||W_N - W_0|| / ||W_0|| of the efficacies.
  history -> association, split by the heard input's group: screen (inputs 0-1663: grey now + before),
            body (1664-1999: body sense x 8), fovea (2000-3535)
  efference -> association
  sensory -> association (for scale)
plus the learned gains (history.hearing per block, efference.hearing).
Sanity rule (chunk 25, no verdict): history -> association AND efference -> association changed by more than 10%.
    py099.sh memread.py ../coldstart/runs/ScipioA/ScipioA_ck0000.npz ../coldstart/runs/ScipioA/ScipioA_ck0025.npz
"""
from __future__ import annotations
import json, sys
import numpy as np

GROUPS = {"screen": (0, 1664), "body": (1664, 2000), "fovea": (2000, 3536)}


def arrays(p):
    with np.load(p) as f:
        return {k: f[k] for k in ("pre", "post", "efficacy")}


def rel(a, b):
    n0 = float(np.linalg.norm(a))
    return dict(norm0=round(n0, 4), normN=round(float(np.linalg.norm(b)), 4),
                rel_change=round(float(np.linalg.norm(b - a)) / n0, 4) if n0 > 0 else None)


def main():
    p0, pN = sys.argv[1], sys.argv[2]
    from cadence import Brain
    bN = Brain.load(pN)
    pops = {k: np.asarray(v) for k, v in bN.connectome.populations.items()}
    A0, AN = arrays(p0), arrays(pN)
    assert np.array_equal(A0["pre"], AN["pre"]) and np.array_equal(A0["post"], AN["post"]), "different connectomes"
    pre, post = A0["pre"], A0["post"]
    n = int(max(pre.max(), post.max())) + 1
    region = np.full(n, "", object)
    for k, v in pops.items():
        region[v] = k
    assoc = np.isin(post, pops["association"]) if "association" in pops else None
    rep = dict(ck0=p0, ckN=pN)
    sens = np.asarray(bN.sensory_index)
    pos = np.full(n, -1); pos[sens] = np.arange(len(sens))
    m = np.isin(pre, sens) & assoc
    rep["sensory_to_association"] = rel(A0["efficacy"][m], AN["efficacy"][m])
    if bN.history is not None:
        glow = np.asarray(bN.history.glow); w = bN.history.width
        hpos = np.full(n, -1); hpos[glow] = np.arange(len(glow)) % w  # history neuron -> the input it hears
        m = (hpos[pre] >= 0) & assoc
        rep["history_to_association"] = rel(A0["efficacy"][m], AN["efficacy"][m])
        for g, (lo, hi) in GROUPS.items():
            mg = m & (hpos[pre] >= lo) & (hpos[pre] < hi)
            rep[f"history_to_association_{g}"] = rel(A0["efficacy"][mg], AN["efficacy"][mg])
        rep["hearing_history"] = None if bN.history.hearing is None else np.ravel(bN.history.hearing).round(5).tolist()
    if bN.efference is not None:
        m = np.isin(pre, np.asarray(bN.efference.glow)) & assoc
        rep["efference_to_association"] = rel(A0["efficacy"][m], AN["efficacy"][m])
        hz = getattr(bN.efference, "hearing", None)
        rep["hearing_efference"] = None if hz is None else np.ravel(hz).round(5).tolist()
    h = rep.get("history_to_association", {}).get("rel_change"); e = rep.get("efference_to_association", {}).get("rel_change")
    if h is not None and e is not None:
        rep["sanity_memory_learned"] = bool(h > 0.10 and e > 0.10)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
