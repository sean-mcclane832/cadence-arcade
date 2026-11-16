"""Scipio public gauntlet runner (Sean 11/14: every 10 chunks, 10 lives to beat the game; public only, never a decision
metric). Idempotent: finds checkpoints at chunks 10, 20, 30, ... of the Scipio runs that have no gauntlet report yet
and plays them with coldstart/cgauntlet.py, ONE process at a time, only while free RAM >= 4 GB.

Gauntlet rules (registered 11/14): real start of 1-1, game order, greedy, weights untouched (no lesson, no DAgger rows),
memories reset each life, 10 lives in total, mem.RETRY's 10 idle-frame starts, stuck limit 225 and the 900-decision
cap each cost a life, STOPS AT 7-4 (World 8 stays unseen). Reports: coldstart/runs/<run>/gauntlet/<run>_ckNNNN.{json,mp4}.
    py099.sh gauntlet_due.py            (all due, oldest first)
    py099.sh gauntlet_due.py --once     (at most one; what the training guard starts)
"""
from __future__ import annotations
import argparse, ctypes, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "coldstart"
RUNS = ("ScipioA", "ScipioB", "Twin", "ScipioFast", "MarcellusA", "MarcellusB", "TwinB")
LOCK = CS / "runs" / "gauntlet.lock"
MIN_FREE_GB = 4.0


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def finished_chunks(run):
    h = CS / "runs" / run / "history.jsonl"
    if not h.exists():
        return set()
    return {json.loads(l)["chunk"] for l in h.read_text().splitlines() if l.strip()}


def due():
    out = []
    for run in RUNS:
        done = finished_chunks(run)
        for k in sorted(done):
            if k == 0 or k % 10:
                continue
            ck = CS / "runs" / run / f"{run}_ck{k:04d}.npz"
            rep = CS / "runs" / run / "gauntlet" / f"{run}_ck{k:04d}.json"
            if ck.exists() and not rep.exists():
                out.append((k, run, ck, rep))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); a = ap.parse_args()
    assert int(os.environ.get("MARIO_STUCK", "225")) == 225
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        if time.time() - LOCK.stat().st_mtime < 3 * 3600:
            print("another gauntlet is running (lock)"); return
        LOCK.unlink(missing_ok=True); return main()
    try:
        for k, run, ck, rep in due():
            if free_gb() < MIN_FREE_GB:
                print(f"{run} ck{k}: only {free_gb():.1f} GB free (< {MIN_FREE_GB}), later"); break
            rep.parent.mkdir(parents=True, exist_ok=True)
            name = f"{run}_ck{k:04d}"
            cmd = [sys.executable, "cgauntlet.py", "--name", name, "--brain", str(ck.relative_to(CS)), "--senses", "fabius",
                   "--lives", "10", "--retries", "mem", "--first", "1-1", "--last", "7-4", "--mode", "greedy",
                   "--label", f"{run} chunk {k}, greedy", "--out", str(rep.with_suffix("").relative_to(CS))]
            print("run:", " ".join(cmd), flush=True)
            t0 = time.time(); r = subprocess.run(cmd, cwd=CS)
            print(f"{name}: exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
            if a.once:
                break
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
