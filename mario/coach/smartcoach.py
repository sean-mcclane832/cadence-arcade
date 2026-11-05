"""Coach variants for game.Game. install(g, name, **opts) swaps g.coach_weighted / g.coach_scores on one Game.

Every variant keeps game.py's plan menu (the lean menu, HORIZON 16) and its labels (first button of the chosen
plan), so labels stay compatible with every brain and every pile. What changes is how plans are scored.

  base    game.Game's own coach: the plan that ends furthest right at the horizon without dying.
  verify  base, plus a doom check: a plan only counts as alive if, from where it ends, at least one short
          tail (tails()) survives T more decisions. Fixes "ends mid-air over a pit at the horizon" picks.
  smart   verify + commitment + stall search (random sticky sequences) + a fallback to base scoring when the
          doom check makes the coach procrastinate. The one to use; see COACH.md.

Labels are always one of the lean menu's buttons {run, run+jump, wait, left} (game.RUN, RUNJUMP, NOOP, LEFT).
"""
from __future__ import annotations
import numpy as np

import game
from game import ACTIONS, FRAMES, RUN, RUNJUMP, NOOP, LEFT, JUMP, RIGHT, RJUMP

DEAD, ALIVE, FLAG, PIPE = 0, 1, 2, 3
FLAG_BONUS, PIPE_BONUS, DEATH = 1e6, 5e5, -1e5


def tails(T):
    return [[RUN] * T, [NOOP] * T, [LEFT] * T, [RUNJUMP] * T,
            [NOOP] * 2 + [RUNJUMP] * (T - 2), [LEFT] * 2 + [RUNJUMP] * (T - 2),
            [RUNJUMP] * 2 + [RUN] * (T - 2), [LEFT] * (T // 2) + [RUN] * (T - T // 2)]


class Sim:
    """Simulation from the decision's root state (single nes-py backup slot: every branch replays from root)."""

    def __init__(self, g):
        self.g = g; self.env = g.env; self.frames = 0

    def __enter__(self):
        env = self.env
        self.saved = (env._time_last, env._x_position_last, env.done)
        env._backup()
        return self

    def __exit__(self, *a):
        env = self.env
        env._restore()
        env._time_last, env._x_position_last, env.done = self.saved

    def run(self, seq):
        """Restore root, play seq. Returns (status, x, decisions played)."""
        env = self.env
        env._restore()
        for t, a in enumerate(seq):
            byte = ACTIONS[a]
            for _ in range(FRAMES):
                env._frame_advance(byte); self.frames += 1
                if env._flag_get:
                    return FLAG, int(env._x_position), t + 1
                if env._is_dying or env._is_dead:
                    return DEAD, int(env._x_position), t + 1
                if env.ram[0x000E] in (2, 3):
                    return PIPE, int(env._x_position), t + 1
        return ALIVE, int(env._x_position), len(seq)


def score_of(st, x):
    return {FLAG: FLAG_BONUS + x, PIPE: PIPE_BONUS + x, DEAD: DEATH + x, ALIVE: float(x)}[st]


class Verify:
    def __init__(self, g, T=8, max_checks=8, commit=False):
        self.g = g; self.T = T; self.max_checks = max_checks; self.tails = tails(T)
        self.calls = 0; self.frames = 0; self.commit = commit; self.last = None

    def doomed(self, sim, plan):
        for tl in self.tails:
            st, x, _ = sim.run(plan + tl)
            if st != DEAD:
                return False
        return True

    def scores(self, want_label=False):
        g = self.g; plans = g.plans
        with Sim(g) as sim:
            x0 = int(g.env._x_position)
            st, x, _ = sim.run(plans[0])
            if st in (FLAG, PIPE) or (st == ALIVE and x - x0 >= game.FAST_GAIN and not self.doomed(sim, plans[0])):
                self.frames += sim.frames
                sc = np.full(len(plans), np.nan); sc[0] = score_of(st, x)
                return sc, True
            raw = np.empty(len(plans)); stats = []
            for i, p in enumerate(plans):
                st, x, _ = sim.run(p); raw[i] = score_of(st, x); stats.append(st)
            sc = raw.copy(); checked = set(); n = 0
            while n < self.max_checks:
                best = sc.max()
                if best < 0 or best >= PIPE_BONUS:
                    break
                cand = [i for i in np.flatnonzero(sc >= best - 2.0) if i not in checked and stats[i] == ALIVE]
                if not cand:
                    break
                i = int(cand[0]); checked.add(i); n += 1
                if self.doomed(sim, plans[i]):
                    sc[i] = DEATH + raw[i]
            self.frames += sim.frames
        return sc, False

    def coach_scores(self):
        sc, fast = self.scores()
        if fast:  # caller wants every plan: fill in (rare path; dagger tiew only)
            return self.full_scores()
        return sc

    def full_scores(self):
        g = self.g
        with Sim(g) as sim:
            out = np.empty(len(g.plans))
            for i, p in enumerate(g.plans):
                st, x, _ = sim.run(p); out[i] = score_of(st, x)
                if st == ALIVE and self.doomed(sim, p):
                    out[i] = DEATH + x
        return out

    def coach_weighted(self):
        g = self.g; self.calls += 1
        sc, fast = self.scores()
        if fast:
            return g.plans[0][0], 1.0
        best = sc.max()
        i = int(np.flatnonzero(sc >= best - 2.0)[0])
        label = g.plans[i][0]
        first = np.array([p[0] for p in g.plans])
        other = sc[first != label]
        alt = other.max() if len(other) else best
        return label, (game.CRITICAL_WEIGHT if best - alt > game.CRITICAL_PX else 1.0)


ESCAPES = __import__("pathlib").Path(__file__).resolve().parent / "data" / "escapes.json"
_ESC = {}


def load_escapes():
    """{level: [[raw x, y, button sequence, area], ...]}: stall escapes the shooting found in coverage runs
    (training levels only). Read-only at label time."""
    if "d" not in _ESC:
        import json
        _ESC["d"] = json.loads(ESCAPES.read_text()) if ESCAPES.exists() else {}
    return _ESC["d"]


SHOOT_BUTTONS = np.array([RUN, RUNJUMP, NOOP, LEFT])  # the lean menu's buttons: labels stay in the brain's set
SHOOT_P = np.array([0.35, 0.30, 0.20, 0.15])


class Smart(Verify):
    """verify + three additions, each active only where the menu is not enough:
    - commitment: the rest of the last chosen plan or sequence is offered again at the next decision as one more
      candidate (scored and verified like the menu, wins ties), so a found route is followed instead of re-found
      and the coach does not dither between near-equal plans (2-1 x 530: small Mario jumping in place at the
      stairs for 40 decisions without it). Only if the coach's own button was played (DAgger: it lapses).
    - stall search: when the best verified plan gains < stall_px from here for stuck_min calls in a row, or no
      real progress for `patience` decisions, or every plan dies: try escapes cached near here (escapes.json,
      optional), then shoot N random 'sticky' sequences of L decisions (each button held 1-4 or 1-10 decisions)
      and keep the best verified one if it beats the menu by 8 px. Finds waits, back-ups, springboard bounces
      and timed jumps the fixed menu lacks. After a miss, wait `backoff` decisions before shooting again.
    - fallback: no real progress for `patience` decisions and no shot found: label with game.py's own scoring
      for this decision (the doom check can make waiting look safer forever, 1-1 x 2370 / 2-4 x 1714).
    """

    def __init__(self, g, T=8, max_checks=8, N=48, L=40, stall_px=24, seed=0, cache=True, learn=False, near=64,
                 stuck_min=3, backoff=2, patience=8):
        super().__init__(g, T, max_checks)
        self.N, self.L, self.stall_px, self.seed = N, L, stall_px, seed
        self.plan = None; self.plan_at = None; self.stalls = 0; self._pending = None
        self.learn, self.near = learn, near
        self.key = f"{g.level[0]}-{g.level[1]}"
        self.escapes = load_escapes().get(self.key, []) if cache else []
        self.found = []; self.cache_hits = 0
        self.stuck_min, self.backoff, self.backoff_until, self.lowgain = stuck_min, backoff, 0, 0
        self.patience, self.noprog, self.best_x0, self.last_dec, self.fallbacks = patience, 0, 0, None, 0
        self.shot_left = 0

    def _shoot(self, rng):
        out = []
        for n in range(self.N):
            # half the shots hold each button 1-4 decisions, half 1-10: short holds find timed presses (2-1's
            # springboard: 5/600 short-hold shots clear it vs 1/600 long-hold), long holds find waits and runs
            hold = 4 if n % 2 == 0 else 10
            seq = []
            while len(seq) < self.L:
                seq += [int(rng.choice(SHOOT_BUTTONS, p=SHOOT_P))] * int(rng.integers(1, hold + 1))
            out.append(seq[:self.L])
        return out

    def _eval(self, sim, seqs, verify_top):
        raw = np.empty(len(seqs)); stats = []
        for i, s in enumerate(seqs):
            st, x, _ = sim.run(s); raw[i] = score_of(st, x); stats.append(st)
        sc = raw.copy(); checked = 0
        for i in np.argsort(-raw):
            if checked >= verify_top or sc.max() >= PIPE_BONUS:
                break
            if stats[i] != ALIVE:
                continue
            checked += 1
            if self.doomed(sim, seqs[i]):
                sc[i] = DEATH + raw[i]
            elif sc[i] >= sc.max():
                break  # best is verified alive
        self._raw = raw
        return sc

    def decide(self):
        g = self.g; plans = [list(p) for p in g.plans]; H = len(plans[0])
        here = (g.decisions, int(g.env._x_position))
        cands = list(plans)
        if (self.plan and self.plan_at == g.decisions and getattr(g, "_last_action", None) == self._pending
                and len(self.plan) >= 1):
            cands.append(self.plan + [RUN] * max(0, H - len(self.plan)))
        has_commit = len(cands) > len(plans)
        with Sim(g) as sim:
            x0 = int(g.env._x_position)
            st, x, _ = sim.run(cands[0])
            if (st in (FLAG, PIPE) or
                                              (st == ALIVE and x - x0 >= game.FAST_GAIN and not self.doomed(sim, cands[0]))):
                self.frames += sim.frames; self.plan = None
                return cands[0], 1.0
            sc = self._eval(sim, cands, self.max_checks); raw_menu = self._raw[:len(plans)].copy()
            # real progress: calls in a row (consecutive decisions) where x has not passed the best x seen
            if self.last_dec is not None and g.decisions == self.last_dec + 1 and x0 <= self.best_x0:
                self.noprog += 1
            else:
                self.noprog = 0
            if self.last_dec is None or g.decisions != self.last_dec + 1:
                self.best_x0 = x0
            self.best_x0 = max(self.best_x0, x0); self.last_dec = g.decisions
            # menu tie rule (game.py): the first plan in menu order within 2 px of the best
            best = sc.max(); i = int(np.flatnonzero(sc >= best - 2.0)[0])
            if has_commit and sc[-1] >= best - 2.0:  # ties go to the plan already being followed (no dithering)
                i = len(cands) - 1
            seq = cands[i]
            first = np.array([c[0] for c in cands]); other = sc[first != seq[0]]
            alt = other.max() if len(other) else best
            wt = game.CRITICAL_WEIGHT if best - alt > game.CRITICAL_PX else 1.0
            raw_x_best = best if best < PIPE_BONUS else 1e9
            # stall: the best menu plan gains < stall_px from HERE for stuck_min calls in a row, or every plan dies.
            # Counted per call (not g.stuck), so it means the same when DAgger asks from a brain's stuck state.
            low = best < PIPE_BONUS and raw_x_best - x0 < self.stall_px
            self.lowgain = self.lowgain + 1 if low else 0
            following_shot = has_commit and i == len(cands) - 1 and self.shot_left > 0
            self.shot_left = self.shot_left - 1 if following_shot else 0
            procrastinating = self.noprog >= self.patience and not following_shot
            stalled = (best < 0 or (self.lowgain >= self.stuck_min or procrastinating)) and g.decisions >= self.backoff_until
            stalled = stalled or (best < 0 and not procrastinating)
            shot_ok = False
            if best < PIPE_BONUS and stalled:
                self.stalls += 1
                y0 = int(g.env.ram[0x00CE]); area = int(g.env.ram[0x0760])
                # 1) escapes found before near here (this level, same area), with small timing shifts
                near = [e for e in self.escapes if e[3] == area and abs(e[0] - x0) <= self.near]
                near.sort(key=lambda e: abs(e[0] - x0) + abs(e[1] - y0))
                tries = []
                for e in near[:8]:
                    sq = list(e[2])
                    tries += [sq, [NOOP] + sq, [NOOP] * 2 + sq, sq[1:], sq[2:]]
                hit = False
                if tries:
                    tsc = self._eval(sim, tries, 4); j = int(np.argmax(tsc))
                    if tsc[j] > best + 8:
                        seq = tries[j]; wt = game.CRITICAL_WEIGHT; best = tsc[j]; hit = True; self.cache_hits += 1
                        shot_ok = True
                # 2) random sticky shooting
                if not hit:
                    rng = np.random.default_rng([self.seed, g.level[0], g.level[1], here[0], here[1]])
                    shots = self._shoot(rng)
                    ssc = self._eval(sim, shots, 6)
                    j = int(np.argmax(ssc))
                    if ssc[j] > best + 8:
                        seq = shots[j]; wt = game.CRITICAL_WEIGHT; best = ssc[j]; shot_ok = True
                        self.found.append([x0, y0, [int(v) for v in seq], area])
                        if self.learn:
                            self.escapes.append(self.found[-1])
                    else:
                        self.backoff_until = g.decisions + self.backoff  # nothing better: let the menu wait a bit
            if shot_ok:
                self.shot_left = len(seq) - 1
            if procrastinating and not shot_ok:
                # no real progress for `patience` decisions and nothing better found: the doom check is being
                # too timid (danger past plan + tail that the tails can't solve, so waiting always looks safer).
                # Fall back to game.py's own scoring for this decision (what got the base coach through).
                self.fallbacks += 1
                rb = raw_menu.max(); k = int(np.flatnonzero(raw_menu >= rb - 2.0)[0]); seq = plans[k]
                fm = np.array([p[0] for p in plans]); o = raw_menu[fm != seq[0]]
                wt = game.CRITICAL_WEIGHT if len(o) and rb - o.max() > game.CRITICAL_PX else 1.0
            self.frames += sim.frames
        self.plan = list(seq[1:])  # offered again next decision (commitment), if the coach's button is played
        return seq, wt

    def coach_weighted(self):
        self.calls += 1
        g = self.g
        seq, wt = self.decide()
        # the next decision's root, if the coach's own button is played (DAgger: the commitment then lapses)
        self.plan_at = None
        if self.plan:
            self._pending = int(seq[0])
            self.plan_at = g.decisions + 1
        return int(seq[0]), wt

    def coach_scores(self):
        return self.full_scores()


VARIANTS = {"verify": Verify, "smart": Smart}


def install(g, name="base", **opts):
    if name == "base":
        return g
    c = VARIANTS[name](g, **opts)
    g.coach_weighted = c.coach_weighted
    g.coach_scores = c.coach_scores
    g.coach = lambda: c.coach_weighted()[0]
    step = g.step

    def tracked_step(a, *args, **kw):  # remember the button played, so a commitment only continues on its own path
        g._last_action = int(a)
        return step(a, *args, **kw)
    g.step = tracked_step
    g._smart = c
    return g
