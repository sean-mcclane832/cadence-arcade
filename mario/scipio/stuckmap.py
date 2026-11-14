"""Where do the Scipio brains get stuck? (Sean 11/14: "render and see exactly what spots it's getting stuck at")

Replays SEL8 (8 starts x 24 levels, greedy) on each run's latest SCORED checkpoint exactly as ctrain.consider scores it
(mem._player greedy cold + mem.play_attempt, seed 0, stuck 225, Body.step keeps the Fabius previous screen, the trainer's
pool initializer plug.worker_init(deep, "smart", {})), checks the x lists against that chunk's history row, and records
how every attempt ended: end x / y, decisions, buttons held over the last 100 decisions, the last screen (PNG) and for
stuck attempts the last 60 decisions (4 s) as an MP4 plus the coach's button at the stuck spot.

    stuckmap.py copy                 copy each run's latest scored checkpoint to scratch/stuck/ (runs/ stay read-only)
    stuckmap.py play [run ...]       replay (2 workers, Idle via the caller), appends scratch/stuck/ends_<tag>.jsonl
    stuckmap.py verify               replay vs history, per level and start
    stuckmap.py page                 stuck/index.html + stuck/ends.json (+ img/, clips/)
Run with CUDA_VISIBLE_DEVICES=-1 CADENCE_BACKEND=cpu MARIO_STUCK=225 on .venv-099 at Idle priority (stuckmap.sh).
"""
from __future__ import annotations
import json, os, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only"
os.environ.setdefault("MARIO_STUCK", "225")
import numpy as np

HERE = Path(__file__).resolve().parent; MARIO = HERE.parent
for p in ("coldstart", "memory", "concepts", "coach", ""):
    sys.path.insert(0, str(MARIO / p))
RUNS_DIR = MARIO / "coldstart" / "runs"
RUNS = ["ScipioA", "ScipioB", "ScipioFast", "Twin"]
SCR = HERE / "scratch" / "stuck"; SCR.mkdir(parents=True, exist_ok=True)
PAGE = HERE / "stuck"
LEVELS = [f"{w}-{s}" for w in range(1, 7) for s in range(1, 5)]
SEL8 = (0, 48, 9, 57, 18, 66, 27, 75)
NAMES = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
CLIP = 60  # decisions kept for the clip: 60 x 4 frames = 4 s
BUCKET = 32


def init():
    import cs, mem, plug
    mem.HORIZON = 250  # Scipio's setting (prime only; play does not read it). Never set in mem.py
    plug.worker_init(cs.deep_levels(), "smart", {})
    import dagger as D
    assert D.STUCK == 225, D.STUCK


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                    ("ext", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2 ** 30


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def save_mp4(frames, path):
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(path), (256, 240), fps=15, codec="libx264", pix_fmt_out="yuv420p",
                                    output_params=["-crf", "32", "-preset", "slow", "-movflags", "+faststart"], macro_block_size=8)
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f))
    w.close()


def play_task(t):
    import dagger as D
    import mem
    run, ck, L, s = t["run"], t["ck"], t["level"], t["start"]
    pl = mem._player(ck, "greedy", cold=True, senses="fabius")  # as mem.episode (kind "act")
    buf, ys = [], []
    fin = {}

    def capture(g):  # runs inside g.close(), after the attempt is over: nothing here can change the play
        fin["screen"] = g.screen.copy(); fin["x"] = int(g.x); fin["max_x"] = int(g.max_x)
        try:
            fin["y"] = int(g.env.ram[0xCE]); fin["yv"] = int(g.env.ram[0xB5])
        except Exception:  # noqa: BLE001
            pass
        if not g.done and g.stuck >= D.STUCK and t.get("coach", True):
            t0 = time.time()
            try:
                lab, wt = g.coach_weighted(); fin["coach"] = int(lab); fin["coach_w"] = float(wt)
            except Exception as e:  # noqa: BLE001
                fin["coach_err"] = str(e)[:120]
            fin["coach_s"] = round(time.time() - t0, 2)

    def cb(g, o, a):
        if not buf:
            orig = g.close
            def close():
                capture(g); orig()
            g.close = close
        buf.append(g.screen.copy())
        if len(buf) > CLIP:
            buf.pop(0)
        ys.append(int(g.env.ram[0xCE]))

    r = mem.play_attempt(pl, L, s, cb=cb)  # seed 0, stuck D.STUCK: as mem.episode
    acts = [int(a) for a in r["actions"]]
    last = acts[-100:]
    held = Counter(NAMES[a] for a in last)
    run_len = 1
    while run_len < len(acts) and acts[-run_len - 1] == acts[-1]:
        run_len += 1
    kind = r["end"] if r["end"] != "stuck" else ("nostart" if r["max_x"] <= 60 else "midstuck")
    tag = f"{run}_{L}_s{s}"
    rec = dict(run=run, ck=Path(ck).name, level=L, start=s, end=r["end"], kind=kind, max_x=r["max_x"], x=fin.get("x"),
               y=fin.get("y", ys[-1] if ys else None), yv=fin.get("yv"), decisions=r["decisions"],
               held100={k: v for k, v in held.most_common()}, last_btn=NAMES[acts[-1]] if acts else None, last_run=run_len,
               coach=NAMES[fin["coach"]] if "coach" in fin else None, coach_w=fin.get("coach_w"), coach_s=fin.get("coach_s"),
               coach_err=fin.get("coach_err"), xs_tail=r["xs"][-CLIP:])
    if r["end"] != "flag":
        d = SCR / "img" / run; d.mkdir(parents=True, exist_ok=True)
        save_png(fin.get("screen", buf[-1]), d / f"{tag}.png"); rec["png"] = f"img/{run}/{tag}.png"
        if r["end"] == "stuck":
            d = SCR / "clips" / run; d.mkdir(parents=True, exist_ok=True)
            save_mp4(buf[1:] + [fin["screen"]] if "screen" in fin else buf, d / f"{tag}.mp4"); rec["clip"] = f"clips/{run}/{tag}.mp4"
    return rec


def scored(run):
    rows = [json.loads(l) for l in open(RUNS_DIR / run / "history.jsonl") if l.strip()]
    return [r for r in rows if r.get("scored") and r.get("solo")]


def cmd_copy():
    man = {}
    for run in RUNS:
        h = scored(run)[-1]; k = h["chunk"]
        src = RUNS_DIR / run / f"{run}_ck{k:04d}.npz"; dst = SCR / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
        man[run] = dict(chunk=k, ck=str(dst), history=h)
        print(run, "chunk", k, "->", dst.name, flush=True)
    (SCR / "manifest.json").write_text(json.dumps(man, indent=1))


def cmd_play(runs):
    man = json.loads((SCR / "manifest.json").read_text())
    runs = runs or RUNS
    g = free_gb(); print(f"free RAM {g:.1f} GB", flush=True)
    assert g >= 4.0, "less than 4 GB free: not starting"
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    tasks = []
    for run in runs:
        part = SCR / f"ends_{run}.jsonl"
        done = {(r["level"], r["start"]) for r in map(json.loads, open(part))} if part.exists() else set()
        tasks += [dict(run=run, ck=man[run]["ck"], level=L, start=s) for L in LEVELS for s in SEL8 if (L, s) not in done]
    print("tasks", len(tasks), flush=True); t0 = time.time(); n = 0
    with ProcessPoolExecutor(2, mp_context=get_context("spawn"), initializer=init) as ex:
        futs = [ex.submit(play_task, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); n += 1
            with open(SCR / f"ends_{r['run']}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if n % 24 == 0:
                print(f"{n}/{len(tasks)} {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


def load_ends(run):
    return [json.loads(l) for l in open(SCR / f"ends_{run}.jsonl")]


def cmd_verify():
    man = json.loads((SCR / "manifest.json").read_text()); out = {}
    for run in RUNS:
        p = SCR / f"ends_{run}.jsonl"
        if not p.exists():
            continue
        ends = {(r["level"], r["start"]): r for r in load_ends(run)}
        solo = man[run]["history"]["solo"]; bad = []
        for L in LEVELS:
            ev = [ends.get((L, s)) for s in SEL8]
            if None in ev:
                bad.append((L, "missing")); continue
            mine = dict(x=[e["max_x"] for e in ev], flags=sum(e["end"] == "flag" for e in ev),
                        mean=round(float(np.mean([e["max_x"] for e in ev]))), died=sum(e["end"] == "died" for e in ev),
                        stuck=sum(e["end"] == "stuck" for e in ev), nostart=sum(e["kind"] == "nostart" for e in ev))
            for k in mine:
                if mine[k] != solo[L][k]:
                    bad.append((L, k, mine[k], solo[L][k]))
        tot = lambda f: sum(f(solo[L]) for L in LEVELS)
        out[run] = dict(chunk=man[run]["chunk"], ok=not bad, mismatches=bad,
                        history=dict(mean=round(float(np.mean([solo[L]["mean"] for L in LEVELS])), 1), died=tot(lambda v: v["died"]),
                                     stuck=tot(lambda v: v["stuck"]), nostart=tot(lambda v: v["nostart"]), flags=tot(lambda v: v["flags"])))
        print(run, "chunk", man[run]["chunk"], "MATCH" if not bad else f"MISMATCH {bad[:6]}", out[run]["history"], flush=True)
    (SCR / "verify.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "copy":
        cmd_copy()
    elif c == "play":
        cmd_play(sys.argv[2:])
    elif c == "verify":
        cmd_verify()
    elif c == "page":
        import stuckpage
        stuckpage.build()
