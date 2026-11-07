"""Corvus: a Cadence 0.94 brain learns Flappy Bird by imitating a scripted teacher (DAgger, Cadence docs/teach.md),
live in a window on the main monitor.
    python train.py            (watch round, then own-path rounds until the dev games hit the cap or time runs out)
Keys: space pause, f fast (draw every 10th frame), q quit (saves first).
Seeds: training streams 0+, dev games 1000 (stopping rule), test games 5000 (only in test.py).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from cadence import Brain
from cadence.recording import record_settlements
import game as G
from view import View

HERE = Path(__file__).resolve().parent
STREAMS, FRAMES_PER_ROUND, MAX_ROUNDS, DEV_GAMES, DEV_CAP, MAX_MIN = 32, 1500, 10, 20, 50, 25


def last_activation(recs, row=0):
    if not recs:
        return None
    a = np.asarray(max(recs, key=lambda r: r.steps).activation)
    return a[-1, row] if a.ndim == 3 else a[row] if a.ndim == 2 else a


def play(brain, v, g, label, teach_fn=None, frames=None, learn=True):
    """Run g until all games end (frames=None) or for `frames` frames, drawing row 0. Brain acts unless teach_fn
    drives (watch round). With learn, the teacher's label for each reached state is taught by lesson."""
    assoc, motor = np.asarray(brain.association_index), np.asarray(brain.motor_index)
    brain.reset()  # new set of streams (batch size may change between training and dev games)
    t = 0
    while (frames is None and not g.dead.all()) or (frames is not None and t < frames):
        if not v.pump():
            return False
        while v.paused and v.pump():
            time.sleep(0.05)
        obs = G.np.asarray(g.senses()); label_t = G.teacher(g)
        recs = []
        if teach_fn is None:
            with record_settlements(recs.append, label="flappy"):
                act = np.asarray(brain.act(obs, greedy=True)).reshape(-1)
        else:
            act = label_t
        if learn:
            if teach_fn is not None:
                with record_settlements(recs.append, label="flappy"):
                    brain.lesson(obs, label_t)
            else:
                brain.lesson(obs, label_t)
        died = g.step(act)
        if frames is not None and died.any():
            g.reset(died)
        t += 1
        if not v.fast or t % 10 == 0:
            v.draw_game(g, 0, int(act[0]))
            m = last_activation(recs)
            vote = None
            if m is not None:
                mv = m[motor]; e = np.exp(mv - mv.max()); vote = float(e[1] / e.sum())
            v.draw_brain(obs[0], m, assoc, motor, vote, f"{label} | frame {t} | live score {int(g.score[0])}")
            v.flip()
    return True


def main():
    v = View("Corvus learns Flappy Bird (Cadence 0.94)")
    brain = Brain.compose(inputs=G.INPUTS, actions=2, seed=0, teaching="normalized", learning_surprise=True)
    out = HERE / "brains"; out.mkdir(exist_ok=True)
    log = []; t0 = time.time()
    # round 0: watch the teacher
    g = G.Flappy(STREAMS, seed=0, cap=10**9)
    if not play(brain, v, g, "round 0: watching the teacher", teach_fn=G.teacher, frames=FRAMES_PER_ROUND):
        brain.save(out / "corvus.npz"); return
    for rnd in range(1, MAX_ROUNDS + 1):
        dev = G.Flappy(DEV_GAMES, seed=1000, cap=DEV_CAP)
        if not play(brain, v, dev, f"round {rnd - 1} dev test (no learning)", learn=False):
            break
        med = float(np.median(dev.score)); log.append(dict(round=rnd - 1, dev_median=med, dev_mean=float(dev.score.mean()),
                                                           dev_min=int(dev.score.min()), minutes=round((time.time() - t0) / 60, 1)))
        v.history.append(("teach" if rnd == 1 else "own", med)); print(log[-1], flush=True)
        brain.save(out / "corvus.npz")
        if med >= DEV_CAP or (time.time() - t0) / 60 > MAX_MIN:
            break
        g = G.Flappy(STREAMS, seed=rnd * 10, cap=10**9)
        if not play(brain, v, g, f"round {rnd}: own path, teacher corrects", frames=FRAMES_PER_ROUND):
            break
    brain.save(out / "corvus.npz")
    (HERE / "train_log.json").write_text(json.dumps(log, indent=1))
    print("saved", out / "corvus.npz", flush=True)


if __name__ == "__main__":
    main()
