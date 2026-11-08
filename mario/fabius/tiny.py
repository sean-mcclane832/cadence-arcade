"""Tiny tests before the Fabius launch (vault Plans/Fabius Build 2026-11-08, step 6 as redone after the deep review).

One arm per process, lessons as ctrain with --p-seq 1.0 (ordered coach rows, mem.prime burn-in K, cold settles,
eta 0.001), trained on the smoke piles of 1-1 and 4-1, evaluated on 6-4 (held out) and on 160 training rows:
    python tiny.py --arm fabius --brains brains_k8 --K 8         Fabius, button mask scaled by the slot count
    python tiny.py --arm fabius-nofix --brains brains_k8 --K 8   Fabius, plain mask (each slot's nudge beta / 105)
    python tiny.py --arm cato2 --lseed 0 / --lseed 1             the control twice: the noise floor
Every 25 lessons: button agreement (all rows, critical rows W > 1), the majority-class baseline, prediction-slot
accuracy vs always-"same" (Fabius), the change of the association -> button synapses. Writes runs/tiny_*.json.
Decisions: see TINY.md (the mask is set before every lesson; a mask set once at load is dropped by the learner rebuild).
"""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "coldstart"))
import pred as PR  # noqa: E402
import cs  # noqa: E402
from ctrain import Seq  # noqa: E402
mem = cs.mem
SMOKE = HERE.parent / "coldstart" / "data" / "fab_smoke"


def load(L):
    s = Seq(); s.load(SMOKE / L / f"seq_{L}_coach.npz")
    return s, [PR.labels(O) for O, _ in s.atts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--lseed", type=int, default=0)
    ap.add_argument("--brains", default="brains_k8"); ap.add_argument("--lessons", type=int, default=300)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mask", default="", help="btn: button neurons x slot count; all: every output neuron x slot count "
                    "(set before every lesson: the learner is rebuilt at the first lesson after a config change)")
    a = ap.parse_args()
    from cadence import Brain
    data = {L: load(L) for L in ("1-1", "4-1", "6-4")}
    train = [(L, r) for L in ("1-1", "4-1") for r in data[L][0].rows]
    held = [("6-4", r) for r in data["6-4"][0].rows]
    seen = [train[int(j)] for j in np.random.default_rng(99).choice(len(train), 160, replace=False)]
    allp = np.concatenate([np.concatenate(data[L][1]) for L in data])
    bal = (np.bincount(allp.ravel(), minlength=3) / allp.size).round(3).tolist()
    brain = {"cato2": "Cato2A", "slots": "SlotsA", "recall": "RecallA"}.get(a.arm, "FabiusA")
    b = mem.make_cold(Brain.load(HERE / a.brains / f"{brain}_ck0.npz"))
    b.learner.config = dataclasses.replace(b.learner.config, eta=0.001, eta_bias=0.0001)
    slotted = b.learner.slot_count > 1
    def set_mask():
        if a.mask:
            if a.mask.startswith("mix"):  # mixN: buttons x slot count, prediction slots x N
                b.learner.output_mask[b.learner.output_index[7:]] = float(a.mask[3:])
                b.learner.output_mask[b.learner.output_index[:7]] = float(b.learner.slot_count)
                return
            idx = b.learner.output_index[:7] if a.mask == "btn" else b.learner.output_index
            b.learner.output_mask[idx] = float(b.learner.slot_count)
    c = b.connectome
    btn = np.isin(np.asarray(c.post), np.asarray(b.learner.output_index[:7])) & \
        np.isin(np.asarray(c.pre), np.asarray(c.populations["association"]))
    w0 = np.array(b.learner.brain.efficacy)[btn]

    def batch(rows):
        items, ys, ws = [], [], []
        for L, (at, i, y, w) in rows:
            seq, P = data[L]; O, A = seq.atts[at]; items.append((O, A, i)); ws.append(w)
            ys.append(np.concatenate([[y], P[at][i]]) if slotted else y)
        return items, np.array(ys, np.int64), np.array(ws)

    def score(rows):
        acts, ys, ws = [], [], []
        for k in range(0, len(rows), 32):
            items, y, w = batch(rows[k:k + 32])
            x = mem.prime(b, items, a.K)
            acts.append(np.asarray(b.act(x, greedy=True)).reshape(len(y), -1)); ys.append(y); ws.append(w)
        b.reset()
        act, y, w = np.concatenate(acts), np.concatenate(ys), np.concatenate(ws)
        yb = y[:, 0] if slotted else y
        hit = act[:, 0] == yb
        out = dict(agree=round(float(hit.mean()), 3), agree_crit=round(float(hit[w > 1].mean()), 3) if (w > 1).any() else None,
                   majority=round(float((yb == np.bincount(yb).argmax()).mean()), 3))
        if slotted and act.shape[1] > 1:
            out["slot_acc"] = round(float((act[:, 1:] == y[:, 1:]).mean()), 3)
            out["slot_same"] = round(float((y[:, 1:] == 1).mean()), 3)
        return out

    rng = np.random.default_rng(1000 + a.lseed)
    curve = [dict(lessons=0, seen=score(seen), held=score(held), dw=0.0)]
    print(a.arm, a.lseed, curve[-1], flush=True)
    secs = []
    for n in range(1, a.lessons + 1):
        items, y, _ = batch([train[int(j)] for j in rng.choice(len(train), 32)])
        t = time.time()
        x = mem.prime(b, items, a.K)
        set_mask()
        b.lesson(x, y)
        secs.append(time.time() - t)
        if n % 25 == 0:
            b.reset()
            dw = float(np.linalg.norm(np.array(b.learner.brain.efficacy)[btn] - w0))
            curve.append(dict(lessons=n, seen=score(seen), held=score(held), dw=round(dw, 4)))
            print(a.arm, a.lseed, curve[-1], f"{np.mean(secs[-25:]):.2f}s/lesson", flush=True)
    out = dict(arm=a.arm, lseed=a.lseed, K=a.K, brains=a.brains, train_rows=len(train), held_rows=len(held),
               pred_balance=bal, sec_per_lesson=round(float(np.mean(secs)), 3), curve=curve)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"tiny_{a.arm}{a.mask}_{a.lseed}_{a.brains}_K{a.K}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
