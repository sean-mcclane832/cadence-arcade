"""Play one solo attempt with a saved brain and record what the brain actually did at every decision.

    python viz/record_brain.py --brain brains/U3ck83.npz --level 1-1 --start 0 --name u3ck83-1-1-clear

Run from mario/. Three passes, all on the real emulator and the real brain:

1. Play. The brain plays alone exactly like the solo eval (dagger.py _play, senses "body", lean
   menu, per-frame background grey): Brain.predict on each observation, no coach, no learning.
   Every 2nd NES frame (30 fps) is kept for the video, plus the observation of every decision.
2. Record. Each stored observation is settled again inside cadence.recording.record_settlements.
   That gives the real neuron states of every settle step (activation, steps+1 x 2519 neurons).
   Recording runs Cadence's NumPy path, which can differ from the fast path at round-off, so the
   script checks the recorded choice against the button the brain actually pressed and reports
   the agreement. predict() reads no live memory: every decision settles from rest on its own
   observation, so the prefrontal (working memory) region stays at 0 the whole time.
3. Poke (occlusion saliency). For each decision, 4 x 4 block patches of the eyes (both the present
   and the previous frame at the same place) are set to 0 (= "background") and the brain is
   settled again. The drop in the chosen button's motor activation, averaged over the patches
   covering each block, is the saliency map; "flips" counts patches that change the choice. One
   more settle with the whole body-sense vector zeroed tells whether the body sense mattered.

Output: viz/out/<name>.npz (not committed; big) used by brain_video.py and export_page.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

from pathlib import Path

import numpy as np

VIZ = Path(__file__).resolve().parent
MARIO = VIZ.parent
sys.path.insert(0, str(MARIO))

import dagger as D  # noqa: E402
from game import ACTION_NAMES, ACTIONS, COLS, FRAMES, N_FRAME, ROWS  # noqa: E402

PATCH, STRIDE = 4, 2


def play(brain, level, start, keep_every=2):
    """The solo eval loop of dagger._play, plus frames and observations."""
    body = D.Body(start, "body", "lean", level)
    g = body.g
    obs, acts, xs, nes = [], [], [], []

    while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
        o = body.obs()
        a = int(np.asarray(brain.predict(o[None])).reshape(-1)[0])
        obs.append(o.astype(np.float64)); acts.append(a); xs.append(int(g.x))
        # Body.step, with the NES frames kept ("body" senses: no fovea state to update)
        g.step(a, keep_frames=nes)
        body.held = ACTIONS[a]
    end = "flag" if g.flag else ("died" if g.done else "stuck")
    # a short tail so the finish is visible
    tail = 60 if g.flag else 40
    for _ in range(tail):
        try:
            s, _, _, _ = g.env.step(0)
            nes.append(s.copy())
        except Exception:  # noqa: BLE001
            break
    n_dec = len(acts)
    # frame k of the NES stream belongs to decision k // FRAMES (capped: the tail shows the last one)
    vid = nes[::keep_every]
    vid_dec = [min((i * keep_every) // FRAMES, n_dec - 1) for i in range(len(vid))]
    # one frame per decision for the page: the screen the brain saw when it decided
    dec_frames = []
    g2 = D.Body(start, "body", "lean", level)  # replay to grab the decision screens exactly
    for a in acts:
        dec_frames.append(g2.g.screen.copy())
        g2.step(a)
    ok = (g2.g.max_x == g.max_x and g2.g.flag == g.flag)
    g2.g.close()
    res = dict(max_x=int(g.max_x), flag=bool(g.flag), end=end, decisions=n_dec, replay_ok=bool(ok))
    g.close()
    return (np.array(obs), np.array(acts), np.array(xs), np.array(vid, np.uint8), np.array(vid_dec),
            np.array(dec_frames, np.uint8), res)


def play_act(path, level, start, off=(), keep_every=2):
    """Act mode (memory brains, mario/memory): the memory player (mem.MemPlayer, greedy, cold settles, memories
    live) plays one attempt exactly like evalm/gauntletm, and every act's real settle is recorded DURING play
    (memories depend on the history, so a replay of single observations would not reproduce them)."""
    sys.path.insert(0, str(MARIO / "memory"))
    import mem  # noqa: E402
    from cadence.recording import record_settlements
    pl = mem.MemPlayer(str(path), "greedy", off=tuple(off))
    pl.begin(0)
    body = mem.C.CBody(int(start), "body", "lean", level); g = body.g
    brain = pl.b; motor = np.asarray(brain.motor_index)
    obs, acts, xs, nes, dec_frames, final, settle_motor, steps = [], [], [], [], [], [], [], []
    while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK:
        o = body.obs()
        if pl.xkey:
            o = np.concatenate([o, mem.xblock(g.env.ram, pl.xamp)])
        recs = []
        with record_settlements(recs.append, label="viz"):
            a = pl.act(o)
        r = max(recs, key=lambda r: r.steps)
        act = np.asarray(r.activation)[:, 0, :]
        final.append(act[-1].astype(np.float32)); settle_motor.append(act[:, motor].astype(np.float32))
        steps.append(int(r.steps))
        obs.append(o[:2000].astype(np.float64)); acts.append(a); xs.append(int(g.x)); dec_frames.append(g.screen.copy())
        g.step(a, keep_frames=nes); body.held = ACTIONS[a]
    end = "flag" if g.flag else ("died" if g.done else "stuck")
    for _ in range(60 if g.flag else 40):
        try:
            s_, _, _, _ = g.env.step(0); nes.append(s_.copy())
        except Exception:  # noqa: BLE001
            break
    n_dec = len(acts)
    vid = nes[::keep_every]
    vid_dec = [min((i * keep_every) // FRAMES, n_dec - 1) for i in range(len(vid))]
    res = dict(max_x=int(g.max_x), flag=bool(g.flag), end=end, decisions=n_dec, replay_ok=True)
    rec_choice = np.array([int(np.argmax(f[motor])) for f in final])
    g.close()
    return (brain, np.array(obs), np.array(acts), np.array(xs), np.array(vid, np.uint8), np.array(vid_dec),
            np.array(dec_frames, np.uint8), res, np.array(final), settle_motor, np.array(steps), rec_choice, r)


def record(brain, obs):
    from cadence.recording import record_settlements
    final, settle_motor, steps, rec_choice = [], [], [], []
    motor = np.asarray(brain.motor_index)
    for o in obs:
        recs = []
        with record_settlements(recs.append, label="viz"):
            c = int(np.asarray(brain.predict(o[None])).reshape(-1)[0])
        r = max(recs, key=lambda r: r.steps)  # the real settle (a 0-step check record comes first)
        act = np.asarray(r.activation)[:, 0, :]
        final.append(act[-1].astype(np.float32))
        settle_motor.append(act[:, motor].astype(np.float32))
        steps.append(int(r.steps)); rec_choice.append(c)
    return np.array(final), settle_motor, np.array(steps), np.array(rec_choice), r


def saliency(brain, obs, acts):
    motor = np.asarray(brain.motor_index)
    masks = []
    for r0 in range(0, ROWS - PATCH + 1, STRIDE):
        for c0 in range(0, COLS - PATCH + 1, STRIDE):
            m = np.zeros((ROWS, COLS), bool); m[r0:r0 + PATCH, c0:c0 + PATCH] = True
            masks.append(m.reshape(-1))
    masks = np.array(masks)  # (P, 832)
    cover = masks.sum(0).astype(np.float64)
    maps, flips, body_flip, base_act = [], [], [], []
    for i, o in enumerate(obs):
        X = np.repeat(o[None], len(masks) + 2, axis=0)
        for p, m in enumerate(masks):
            X[p + 1, :N_FRAME][m] = 0.0
            X[p + 1, N_FRAME:2 * N_FRAME][m] = 0.0
        X[-1, 2 * N_FRAME:] = 0.0  # body sense off
        st = brain._qualified(brain.stimulus(X, memory=False), operation="predict")
        out = np.asarray(st.activation)[:, motor]
        ch = int(np.argmax(out[0]))
        drop = out[0, ch] - out[1:-1, ch]  # (P,)
        flip = np.argmax(out[1:-1], axis=1) != ch
        maps.append((np.clip(drop, 0, None) @ masks) / np.maximum(cover, 1))
        flips.append(int(flip.sum()))
        body_flip.append(int(np.argmax(out[-1]) != ch))
        base_act.append(ch)
        if i % 50 == 0:
            print(f"  saliency {i}/{len(obs)}", flush=True)
    return np.array(maps, np.float32), np.array(flips), np.array(body_flip), len(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-saliency", action="store_true")
    ap.add_argument("--act", action="store_true", help="memory brains: act mode, memories live (implies --no-saliency)")
    ap.add_argument("--off", default="", help="act mode: memory parts off at play, e.g. thought+button")
    ap.add_argument("--parts", default="", help="act mode: memory parts this brain has (for the panels)")
    args = ap.parse_args()
    from cadence import Brain
    t0 = time.time()
    path = (MARIO / args.brain).resolve()
    lv = tuple(int(v) for v in args.level.split("-"))
    if args.act:
        args.no_saliency = True
        (brain, obs, acts, xs, vid, vid_dec, dec_frames, res, final, settle_motor, steps, rec_choice,
         last) = play_act(path, lv, args.start, [p for p in args.off.split("+") if p])
        print(f"played (act mode) {args.level} start {args.start}: {res} ({time.time() - t0:.0f}s)", flush=True)
    else:
        brain = Brain.load(path)
        obs, acts, xs, vid, vid_dec, dec_frames, res = play(brain, lv, args.start)
        print(f"played {args.level} start {args.start}: {res} ({time.time() - t0:.0f}s)", flush=True)
        final, settle_motor, steps, rec_choice, last = record(brain, obs)
    agree = float((rec_choice == acts).mean())
    print(f"recorded {len(obs)} settles, steps {steps.min()}-{steps.max()}, recorded choice = played button "
          f"on {agree:.1%} ({time.time() - t0:.0f}s)", flush=True)
    S = max(len(s) for s in settle_motor)
    sm = np.array([np.vstack([s, np.repeat(s[-1:], S - len(s), 0)]) for s in settle_motor], np.float32)
    pops = {k: np.asarray(v, np.int32) for k, v in brain.connectome.populations.items()}
    # each association neuron's strongest effective weight onto a motor neuron (for sorting/labels)
    con = brain.connectome
    pre, post = np.asarray(con.pre), np.asarray(con.post)
    w = np.asarray(last.weights)
    motor = np.asarray(brain.motor_index)
    assoc = pops["association"]
    a2m = np.zeros((len(assoc), len(motor)))
    pos_a = {int(n): i for i, n in enumerate(assoc)}; pos_m = {int(n): j for j, n in enumerate(motor)}
    sel = np.isin(pre, assoc) & np.isin(post, motor)
    for p_, q_, ww in zip(pre[sel], post[sel], w[sel]):
        a2m[pos_a[int(p_)], pos_m[int(q_)]] += ww
    extra = {}
    if not args.no_saliency:
        sal, flips, body_flip, n_patch = saliency(brain, obs, acts)
        extra = dict(saliency=sal, flips=flips, body_flip=body_flip, n_patch=n_patch)
        print(f"saliency done ({time.time() - t0:.0f}s)", flush=True)
    out = VIZ / "out"; out.mkdir(exist_ok=True)
    meta = dict(res, brain=args.brain, level=args.level, start=args.start, name=args.name,
                record_agree=agree, steps_min=int(steps.min()), steps_max=int(steps.max()),
                actions=ACTION_NAMES, patch=PATCH, stride=STRIDE, act=bool(args.act),
                parts=[p for p in args.parts.split("+") if p])
    np.savez_compressed(out / f"{args.name}.npz", obs=obs.astype(np.float32), acts=acts, xs=xs, vid=vid,
                        vid_dec=vid_dec, dec_frames=dec_frames, final=final, settle_motor=sm, steps=steps,
                        rec_choice=rec_choice, a2m=a2m, meta=json.dumps(meta),
                        **{f"pop_{k.replace('/', '_')}": v for k, v in pops.items()}, **extra)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
