"""Registered pre-launch parity check (design rewrite 11/14, source question 2; deep review fix 5g).

One long live attempt in act mode (greedy, cold settles, Fabius eyes) with the observations QUANTIZED to uint8/255
before the brain sees them (that is what a stored DAgger row holds). Before every act the brain's own history and
efference traces are copied. Then:
  1. traces: max |live - mem._exact(t)| for t < HORIZON and t >= HORIZON, at HORIZON 250 (the Scipio setting) and
     at a full-attempt horizon (exact).
  2. actions: a fresh copy of the brain is primed with mem.prime(K=0) on every decision of the stored attempt
     (batches of 32, as lessons do) and acts greedy; its actions must equal the live ones.
  3. prime timing at HORIZON 250 on 32-row batches.
    python parity.py ../coldstart/../scipio/brains0/ScipioA_ck0.npz --level 1-1 --start 0
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "memory"))
import mem  # noqa: E402


class QPlayer(mem.MemPlayer):
    """Greedy act-mode player that sees uint8/255 observations and records its traces before each act."""

    def begin(self, seed=0):
        super().begin(seed); self.H, self.E = [], []

    def act(self, o):
        q = np.round(o * 255).astype(np.uint8).astype(np.float64) / 255.0
        assert o.min() >= -1e-9 and o.max() <= 1 + 1e-9, ("observation outside [0, 1]", float(o.min()), float(o.max()))
        if self.b.history is not None and len(self.b.history.trace) == 1:
            self.H.append(self.b.history.trace[0].copy())
        elif self.b.history is not None:
            self.H.append(np.zeros(self.b.history.width))
        if self.b.efference is not None and len(self.b.efference.trace) == 1:
            self.E.append(self.b.efference.trace[0].copy())
        elif self.b.efference is not None:
            self.E.append(np.zeros(7))
        return super().act(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("brain"); ap.add_argument("--level", default="1-1")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stuck", type=int, default=900)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    from cadence import Brain
    pl = QPlayer(a.brain, "greedy", senses="fabius")
    t0 = time.time()
    r = mem.play_attempt(pl, a.level, a.start, keep_obs=True, stuck=a.stuck, max_dec=900)
    live_s = time.time() - t0
    O, A = r["obs"], np.asarray(r["actions"], np.int64); n = len(A)
    d_h, d_e = mem.mem_decays(pl.b)
    rep = dict(brain=a.brain, level=a.level, start=a.start, decisions=n, end=r["end"], max_x=r["max_x"], live_s=round(live_s, 1),
               d_h=d_h, d_e=d_e)
    for hz in (250, 100000):
        mem.HORIZON = hz
        dh = np.zeros(n); de = np.zeros(n)
        for t in range(n):
            H, E = mem._exact(O, A, t, d_h, d_e)
            if H is not None:
                dh[t] = np.abs(H - pl.H[t]).max()
            if E is not None:
                de[t] = np.abs(E - pl.E[t]).max()
        lo = slice(0, min(n, 250)); hi = slice(250, n)
        rep[f"horizon_{hz}"] = dict(hist_max_lt250=float(dh[lo].max()), hist_max_ge250=float(dh[hi].max()) if n > 250 else None,
                                    eff_max_lt250=float(de[lo].max()), eff_max_ge250=float(de[hi].max()) if n > 250 else None)
        # actions after prime (K=0), fresh brain, batches of 32
        b2 = mem.make_cold(Brain.load(a.brain))
        acts = []; tp = []
        for s in range(0, n, 32):
            items = [(O, A, t) for t in range(s, min(n, s + 32))]
            t1 = time.time(); x = mem.prime(b2, items, 0); tp.append(time.time() - t1)
            acts.append(np.asarray(b2.act(x, greedy=True)).reshape(len(items), -1)[:, 0])
        acts = np.concatenate(acts)
        mism = np.nonzero(acts != A)[0]
        rep[f"horizon_{hz}"].update(action_mismatches=int(len(mism)), first_mismatches=mism[:10].tolist(),
                                    prime_s_per_32=round(float(np.mean(tp)), 4))
        print(hz, rep[f"horizon_{hz}"], flush=True)
    h = rep["horizon_250"]
    rep["pass"] = bool(h["hist_max_lt250"] < 1e-6 and h["eff_max_lt250"] < 1e-6
                       and (h["hist_max_ge250"] is None or h["hist_max_ge250"] < 2e-3)
                       and (h["eff_max_ge250"] is None or h["eff_max_ge250"] < 2e-3) and h["action_mismatches"] == 0)
    print(json.dumps(rep), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
