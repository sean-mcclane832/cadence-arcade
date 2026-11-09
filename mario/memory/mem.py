"""Memory brains for Mario (MEMORY.md). Shared pieces:

- SPECS: the four new brains (Cato, Agrippa, Cicero, Livy) and the Cadence settings of their memory parts.
- transplant(): W1_F2ck110's learned weights into a freshly composed connectome that has the memory regions
  (synapses matched by (pre, post) pair; Cadence appends efference / history regions last, so every old
  neuron keeps its index). New memory projections start scaled by `quiet` so the warm brain starts close to
  the old one; learning grows them.
- MemPlayer: plays one attempt in ACT mode (stream memories on, reset() at the attempt start), greedy, or as
  a "retry life" (sampled low-temperature acts + learn(r, done, next_obs) with actor rates 0: only the
  episodic records change, never the weights).
- blame window (OURS, not Cadence's): optional driver-side write of a death into the place records of the
  decisions 5..30 before it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "concepts")); sys.path.insert(0, str(HERE.parent))
import cplay as C  # noqa: E402
import dagger as D  # noqa: E402

START = HERE.parent / "brains" / "W1_F2ck110.npz"
FROZEN = dict(actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
PLACE = dict(memory_choice=True, memory_demonstrations=0.0, memory_fan_in=32)
EFF = dict(efference_amplitude=1.0, efference_decay=0.8)
ECHO = dict(history_decays=(0.9,))
XN, XSTEP, XWIDTH, XAMP = 112, 32, 32.0, 8.0  # x block: bumps every 2 tiles, ~4 tiles wide, up to x 3,584
# compose settings (layout) and play-time amplitudes (retune) per brain
SPECS = {
    "Cato": dict(compose={}, tune=dict(working_memory_amplitude=0.0), parts=[]),
    "Agrippa": dict(compose=dict(PLACE, memory_context=0.0), tune=dict(working_memory_amplitude=0.0), parts=["place"]),
    # exploratory (14:20): sparser place keys, to test whether key locality is the bottleneck
    "AgrippaSparse": dict(compose=dict(PLACE, memory_context=0.0, memory_expansion=4096, memory_winners=8),
                          tune=dict(working_memory_amplitude=0.0), parts=["place"]),
    # exploratory (Sean's idea, 14:55): place memory keyed on x from game RAM (where am I only; decisions still from
    # the screen). The x block is extra sensory input with ZERO weights into the cortex; only the episodic key sees it.
    "AgrippaX": dict(compose=dict(PLACE, memory_context=0.0), tune=dict(working_memory_amplitude=0.0), parts=["place"],
                     xinputs=XN),
    "Cicero": dict(compose=dict(EFF), tune=dict(working_memory_amplitude=3.0), parts=["thought", "button"]),
    "Livy": dict(compose=dict(EFF, **ECHO, **PLACE), tune=dict(working_memory_amplitude=3.0),
                 parts=["thought", "button", "echo", "place"]),
}
PART_OFF = {"thought": dict(working_memory_amplitude=0.0), "button": dict(efference_amplitude=0.0),
            "echo": dict(history_amplitude=0.0), "place": dict(memory_amplitude=0.0),
            "x": dict(),  # "x" (AgrippaX only): x block zeroed at play, see MemPlayer.xamp
            "recall": dict(recall_amplitude=0.0)}  # Cadence 0.95 recall memory (Fabius's recall-off gate)
RETRY = (0, 37, 83, 110, 131, 151, 173, 197, 211, 19)  # idle frames per life (gauntlet.py's 5, extended)


def _arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}




def xblock(ram, amp=XAMP):
    """Mario's level x from game RAM as overlapping bumps (AgrippaX's place key: where am I)."""
    x = int(ram[0x006D]) * 256 + int(ram[0x0086])
    c = np.arange(XN) * XSTEP
    return amp * np.exp(-((x - c) / XWIDTH) ** 2)


def compose(name, seed=0):
    from cadence import Brain
    return Brain.compose(2000 + SPECS[name].get("xinputs", 0), 7, modules=(256,), seed=seed, teaching="normalized", learning_surprise=False,
                         **SPECS[name]["compose"])


def transplant(name, out, quiet=0.1, src=START):
    """Write brain `name` to `out`: the composed memory connectome carrying src's learned parameters."""
    import dataclasses
    from cadence import Brain
    old_b = Brain.load(src)
    new = compose(name)
    new.learner.config = dataclasses.replace(new.learner.config, **{f.name: getattr(old_b.learner.config, f.name)
                                                                    for f in dataclasses.fields(old_b.learner.config)})
    tmp = Path(str(out) + ".fresh.npz"); new.save(tmp)
    old, n = _arrays(src), _arrays(tmp)
    tmp.unlink()
    N = int(max(n["pre"].max(), n["post"].max())) + 1
    extra = SPECS[name].get("xinputs", 0)
    shift = lambda j: np.where(j < 2000, j, j + extra)  # noqa: E731  (x block widens the sensory region)
    old["pre"], old["post"] = shift(old["pre"]), shift(old["post"])
    kn = n["pre"] * N + n["post"]; ko = old["pre"] * N + old["post"]
    order = np.argsort(kn); pos = np.searchsorted(kn[order], ko); mp = order[pos]
    assert (kn[mp] == ko).all(), "old synapse missing in new connectome"
    for k in ("sign", "count", "plastic_synapses"):
        if k == "sign" and extra:  # a wider sensory region shifts the random draws; sign is only the initial
            n[k] = n[k].copy(); n[k][mp] = old[k]; continue  # efficacy (copied below) is what the settle uses
        assert (n[k][mp] == old[k]).all(), k
    m = len(old["bias"]); nm = shift(np.arange(m))
    assert (n["plastic_neurons"][nm] == old["plastic_neurons"]).all()
    for k in ("efficacy", "velocity", "second_moment"):
        n[k] = n[k].copy(); n[k][mp] = old[k]
    for k in ("log_gain", "bias", "velocity_bias", "second_moment_bias"):
        n[k] = n[k].copy(); n[k][nm] = old[k]
    n["actor/w_critic"] = old["actor/w_critic"].copy()
    # quiet start for the memory projections that W1 never used (prefrontal, efference, history -> association)
    pops = new.connectome.populations
    mem_pre = np.zeros(N, bool)
    for r in ("prefrontal", "efference", "history"):
        if r in pops:
            mem_pre[np.asarray(pops[r])] = True
    q = mem_pre[n["pre"]]
    n["efficacy"][q] *= quiet
    if extra:
        n["efficacy"][(n["pre"] >= 2000) & (n["pre"] < 2000 + extra)] = 0.0  # the x block never reaches the cortex
    n["efficacy"] = n["efficacy"] + 0.0  # no signed zeros (-0.0 + 0.0 steps would flip bytes without changing values)
    np.savez(out, **n)
    b = Brain.load(out)
    b.retune(**SPECS[name]["tune"])
    b.save(out)
    return dict(synapses=len(kn), old_synapses=len(ko), memory_synapses=int(q.sum()), quiet=quiet)


def make_cold(b):
    """Every settle starts from rest (as predict and every lesson do), instead of warm from the last moment's
    activity. The memories still carry the past: they enter the drive. Driver-side choice, see MEMORY.md 0.1."""
    b._activity = lambda: None
    return b


def onehot(a, n=7):
    o = np.zeros(n); o[a] = 1.0; return o


class MemPlayer:
    """A brain playing in act mode. mode: 'greedy' or 'retry' (sampled at `temp` + learn, actor rates 0)."""

    def __init__(self, path, mode="greedy", temp=0.05, off=(), blame=None, cold=True, senses="body"):
        from cadence import Brain
        self.path = str(path)
        self.b = Brain.load(self.path)
        if cold:
            make_cold(self.b)
        self.mode, self.temp, self.blame = mode, temp, blame
        if off:
            tune = {}
            for p in off:
                tune.update(PART_OFF[p])
            if tune:
                self.b.retune(**tune)
        if mode == "retry":
            self.b.retune(**FROZEN)
        self.ep = getattr(self.b, "_episodic", None)
        self.senses = senses  # dagger.Body senses mode ("fabius": 3,536 inputs, fabius/eyes.py)
        self.xkey = len(self.b.sensory_index) == D.n_inputs(senses) + XN  # AgrippaX: x block after the senses
        self.xamp = 0.0 if "x" in off else XAMP  # end-of-night ablation: x block zeroed

    def begin(self, seed=0):
        self.b.reset()
        self.b.basal_ganglia.rng = np.random.default_rng(seed)
        self.pending = None
        self.codes, self.acts = [], []

    def act(self, o):
        x = o[None].astype(np.float64)
        if self.mode == "greedy":
            return int(np.asarray(self.b.act(x, greedy=True)).reshape(-1)[0])
        if self.pending is not None:  # outcome of the last decision: nothing, mid-attempt
            self.b.learn([0.0], [False], x)
        a = int(np.asarray(self.b.act(x, temperature=self.temp)).reshape(-1)[0])
        self.pending = True
        if self.blame:
            self.codes.append(x[0]); self.acts.append(a)
        return a

    def end(self, o, outcome):
        """outcome: 'flag' (+1), 'died' (-1), 'stuck' (0); all end the attempt (done)."""
        if self.mode != "retry" or self.pending is None:
            return
        r = {"flag": 1.0, "died": -1.0}.get(outcome, 0.0)
        self.b.learn([r], [True], o[None].astype(np.float64))
        self.pending = None
        if self.blame and outcome != "flag" and self.ep is not None:
            lo, hi, val = self.blame
            n = len(self.acts)
            idx = [n - 1 - k for k in range(lo, hi + 1, 5) if n - 1 - k >= 0]
            from cadence.memories import record_action
            for i in idx:  # one write per decision, batch 1 (the store's fast part is per stream)
                record_action(self.ep.place, self.codes[i][None], np.array([self.acts[i]]), np.array([float(val)]),
                              np.zeros(1), None, self.b.learner.slot_offsets)

    def vote(self, o):
        """The place records' vote for this observation (motor values)."""
        if self.ep is None:
            return np.zeros(7)
        return np.asarray(self.ep.place.recall(o[None].astype(np.float64)))[0]


def play_attempt(player, level, start, seed=0, keep_obs=False, cb=None, max_dec=None, stuck=None):
    """One attempt in act mode. Returns dict like cplay._play (max_x, flag, end, actions, xs[, obs])."""
    lvl = tuple(int(v) for v in level.split("-")) if isinstance(level, str) else tuple(level)
    senses = getattr(player, "senses", "body")
    body = C.CBody(int(start), senses, "lean", lvl); g = body.g
    n_in = D.n_inputs(senses)
    player.begin(seed)
    acts, xs, obs = [], [], []
    o = None
    cap = max_dec or D.MAX_DECISIONS
    lim = stuck or D.STUCK  # stuck: an attempt-specific limit (Fabius DAgger), default the global one
    while not g.done and g.decisions < cap and g.stuck < lim:
        o = body.obs()
        if getattr(player, "xkey", False):
            o = np.concatenate([o, xblock(g.env.ram, getattr(player, "xamp", XAMP))])
        a = player.act(o)
        if keep_obs:
            obs.append(np.round(o[:n_in] * 255).astype(np.uint8))
        acts.append(a); xs.append(int(g.x))
        if cb is not None:
            cb(g, o, a)
        body.step(a)
    end = "flag" if g.flag else ("died" if g.done else "stuck")
    oe = body.obs()
    if getattr(player, "xkey", False):
        oe = np.concatenate([oe, xblock(g.env.ram, getattr(player, "xamp", XAMP))])
    player.end(oe, end)
    out = dict(start=int(start), max_x=int(g.max_x), flag=bool(g.flag), end=end, decisions=g.decisions,
               actions=acts, xs=xs)
    if keep_obs:
        out["obs"] = np.array(obs, np.uint8).reshape(-1, n_in)
    g.close()
    return out


# ------------------------------------------------------------------ worker-side episode (process pools)
_P = {}


def _free_device():
    """Free the last checkpoint's brain before the next one is placed: it sits in a reference cycle, so only gc frees
    it, and torch keeps freed blocks reserved (11/9: ~125 MB of VRAM per checkpoint per worker filled the 8 GB card
    at Fabius chunk 12)."""
    import gc
    gc.collect()
    torch = sys.modules.get("torch")  # never imported here: CPU-only paths stay as they were
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _player(path, mode, temp=0.05, off=(), blame=None, fresh=False, cold=True, senses="body"):
    import os
    st = os.stat(path); key = (path, st.st_mtime_ns, mode, temp, tuple(off), tuple(blame or ()), cold, senses)
    if fresh or key not in _P:
        _P.clear(); _free_device(); _P[key] = MemPlayer(path, mode, temp, off, blame, cold, senses)
    return _P[key]


def episode(task):
    """task: dict(policy, level, start, kind='act'|'predict', off=(), keep_obs, label (dagger), seed)."""
    kind = task.get("kind", "act")
    if kind == "predict":
        r = C.episode(C.task(task["policy"], "body", task["level"], task["start"]))
        return r
    pl = _player(task["policy"], "greedy", off=task.get("off", ()), cold=kind != "warm", senses=task.get("senses", "body"))
    played = play_attempt(pl, task["level"], task["start"], keep_obs=task.get("keep_obs", False), stuck=task.get("stuck"))
    if task.get("label"):
        acts = played["actions"]; n = len(acts)
        idx = {i for i in range(n) if i >= n - task.get("last_all", 60) or i % task.get("every", 3) == 0}
        rng = np.random.default_rng(task.get("seed", 0))
        lab = C._play(dict(C.task("none", "body", task["level"], task["start"], mode="dagger"), script=acts, tiew=True,
                           keep_obs=False), None, idx, rng)
        order = sorted(idx)[:len(lab["labels"])]
        played.update(lab_idx=np.array(order, np.int64), labels=lab["labels"], weights=lab["weights"])
    return played


def retry_series(task):
    """Ten (or n) lives in a row from one held start, place records kept between lives (or wiped: control)."""
    path, level, start = task["policy"], task["level"], int(task["start"])
    lives, control = task.get("lives", 10), task.get("control", False)
    temp, blame = task.get("temp", 0.05), task.get("blame")
    off = tuple(task.get("off", ()))
    pl = MemPlayer(path, "retry", temp, off, blame)
    res = []
    for i in range(lives):
        if control and i:
            pl = MemPlayer(path, "retry", temp, off, blame)
        r = play_attempt(pl, level, start + RETRY[i % len(RETRY)], seed=1000 * start + i)
        res.append(dict(life=i, x=r["max_x"], flag=r["flag"], end=r["end"], dec=r["decisions"]))
    if task.get("save_to"):
        pl.b.save(task["save_to"])
    return dict(start=start, level=level, control=control, lives=res)


def worker_init():
    D._worker_init()


def jdump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ------------------------------------------------------------------ burn-in lessons (training with memory)
HORIZON = 64  # decisions of exact history / efference kept (0.9 ** 64 = 0.001)


def _exact(O, A, t, d_h, d_e):
    """Exact heard history and efference copy at decision t of an attempt (before deciding at t)."""
    lo = max(0, t - HORIZON)
    H = E = None
    if d_h is not None:
        if t > lo:
            w = (1 - d_h) * d_h ** np.arange(t - 1 - lo, -1, -1)
            H = w @ (O[lo:t].astype(np.float64) / 255.0)
        else:
            H = np.zeros(O.shape[1])
    if d_e is not None:
        E = np.zeros(7)
        if t > lo:
            w = (1 - d_e) * d_e ** np.arange(t - 1 - lo, -1, -1)
            np.add.at(E, np.asarray(A[lo:t], np.int64), w)
    return H, E


def mem_decays(b):
    d_h = None if b.history is None else float(b.history.decays[0])
    d_e = None if b.efference is None else float(b.efference.decay)
    return d_h, d_e


HELD_ACTION = {(0, 0, 0, 0): 0, (0, 0, 1, 0): 1, (1, 0, 1, 0): 2, (0, 1, 1, 0): 3, (1, 1, 1, 0): 4, (1, 0, 0, 0): 5,
               (0, 0, 0, 1): 6}  # body sense held bits (A, B, right, left) -> the action pressed last


def last_action(o_uint8):
    bits = tuple(int(v > 127) for v in o_uint8[1664 + 38:1664 + 42])
    return HELD_ACTION.get(bits, 0)


def _static(O):
    """A pile row (no order known): the stationary guess. History = this screen, button memory = the button the body
    sense says was pressed last (as if held), thought memory = burn-in on this same screen."""
    return O[0].astype(np.float64) / 255.0, onehot(last_action(O[0]))


def set_exact(b, items, ts, d_h, d_e):
    B = len(items)
    if d_h is None and d_e is None:
        return
    Hs, Es = [], []
    for (O, A, _), t in zip(items, ts):
        if len(A) == 0:
            H, E = _static(O)
        else:
            H, E = _exact(O, A, t, d_h, d_e)
        Hs.append(H); Es.append(E)
    if d_h is not None:
        b.history.trace = np.stack(Hs)
    if d_e is not None:
        e = b.efference
        e.trace = np.stack(Es); e.last = np.zeros((B, 7)); e.cold = np.zeros(B, bool)
        for r, ((O, A, _), t) in enumerate(zip(items, ts)):
            if len(A) == 0:
                e.last[r, last_action(O[0])] = 1.0
            elif t > 0:
                e.last[r, int(A[t - 1])] = 1.0
            else:
                e.cold[r] = True


def prime(b, items, K):
    """Bring every stream to decision i of its attempt: K greedy burn-in acts on the K previous observations (the
    working trace needs real settles), exact heard history and efference copy (these are fixed functions of what
    was seen and pressed). items: list of (O uint8 (n, width), A actions (n,), i). Rows with i - j < 0 have their
    trace cleared after that act (attempt start). Pile rows (A empty) get the stationary guess (_static).
    Returns the target observations (B, width) as floats."""
    d_h, d_e = mem_decays(b)
    B = len(items)
    b.reset()
    for j in range(K, 0, -1):
        ts = [i - j for (_, _, i) in items]
        tt = [max(t, 0) for t in ts]
        x = np.stack([O[t] for (O, _, _), t in zip(items, tt)]).astype(np.float64) / 255.0
        set_exact(b, items, tt, d_h, d_e)
        b.act(x, greedy=True)
        neg = np.array([r for r, (t, it) in enumerate(zip(ts, items)) if t < 0 and len(it[1])], np.int64)
        if len(neg) and b.working_memory is not None:
            b.working_memory.reset(B, neg)
    set_exact(b, items, [i for (_, _, i) in items], d_h, d_e)
    return np.stack([O[i] for (O, _, i) in items]).astype(np.float64) / 255.0
