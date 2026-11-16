"""Chunk-50 registered read: time-only ablation on ScipioA/B ck46/48/50 (registered) + ScipioFast (descriptive).
At most 6 jobs at once, 1 worker each, BelowNormal (children inherit). Logs in scratch/ablate50/."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "scratch" / "ablate50"; RUNS = HERE.parent / "coldstart" / "runs"
PY = str(HERE.parent.parent / ".venv-099" / "Scripts" / "python.exe")
jobs = [(r, c, "none,time,amp0") for r in ("ScipioA", "ScipioB") for c in (46, 48, 50)]
jobs += [("ScipioFast", c, "none,time") for c in (46, 48, 50)]
live, MAX = [], 6
while jobs or live:
    live = [(n, p) for n, p in live if p.poll() is None]
    while jobs and len(live) < MAX:
        r, c, m = jobs.pop(0); n = f"{r}_ck{c:04d}"
        log = open(OUT / f"{n}.log", "w")
        p = subprocess.Popen([PY, "ablate.py", str(RUNS / r / f"{n}.npz"), "--out", str(OUT / f"{n}.json"), "--workers", "1", "--modes", m],
                             cwd=HERE, stdout=log, stderr=subprocess.STDOUT, creationflags=0x4000)
        live.append((n, p)); print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(20)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
