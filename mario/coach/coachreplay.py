"""Watch the coach: replay finished coach runs from a coverage run's runs.json in a window, exactly as they happened
(the recorded moves, with the one dummy simulated frame per decision that harness.py uses to reproduce a coached
trajectory). The newest finished run plays first; when nothing new has finished, it cycles through the others.
    python coachreplay.py [--runs runs/cov_w4_7] [--pos +20+540] [--speed 1.5]
Keys: space pause, n next run, q quit.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import game  # noqa: E402
from game import ACTION_NAMES  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402
import tkinter as tk  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs/cov_w4_7")
    ap.add_argument("--pos", default="+20+540"); ap.add_argument("--speed", type=float, default=1.5)
    a = ap.parse_args()
    src = HERE / a.runs / "runs.json"
    root = tk.Tk(); root.title("Coach (worlds 4-7 coverage run): replays of finished runs"); root.geometry(a.pos)
    head = tk.Label(root, text="", font=("Consolas", 11), bg="#181a20", fg="#e1e4eb", anchor="w", width=64); head.pack(fill="x")
    gl = tk.Label(root, bg="black"); gl.pack()
    st = dict(paused=False, skip=False, quit=False, img=None)
    root.bind("<space>", lambda e: st.update(paused=not st["paused"])); root.bind("n", lambda e: st.update(skip=True))
    root.bind("q", lambda e: st.update(quit=True)); root.protocol("WM_DELETE_WINDOW", lambda: st.update(quit=True))
    shown = set(); cursor = 0

    def pump():
        root.update()
        while st["paused"] and not st["quit"]:
            time.sleep(0.05); root.update()

    while not st["quit"]:
        try:
            runs = json.loads(src.read_text())
        except Exception:  # noqa: BLE001  (being rewritten right now)
            time.sleep(1); pump(); continue
        fresh = [i for i in range(len(runs)) if i not in shown]
        if fresh:
            i = fresh[-1]; tagline = "NEW"
        else:
            cursor = (cursor + 1) % len(runs); i = cursor; tagline = "replay"
        shown.add(i); r = runs[i]; st["skip"] = False
        lvl = tuple(int(v) for v in r["level"].split("-"))
        g = game.Game(noop_start=int(r["start"]), level=lvl, menu="lean", bg_per_frame=True); e = g.env
        done_n = len(runs)
        for k, act in enumerate(r["acts"]):
            if st["quit"] or st["skip"] or g.done or e.done:
                break
            sv = (e._time_last, e._x_position_last, e.done); e._backup(); e._frame_advance(0x80); e._restore()
            e._time_last, e._x_position_last, e.done = sv
            fr = []
            try:
                g.step(int(act), keep_frames=fr)
            except ValueError:   # the run ended a frame early in this replay
                break
            head.configure(text=f"[{tagline}] coach on {r['level']} start {r['start']} | decision {k + 1}/{len(r['acts'])} | "
                                f"x {g.x} | {ACTION_NAMES[int(act)]} | ends: {r['end'].upper()} at x {r['max_x']}   ({done_n}/96 runs done)")
            t0 = time.time()
            for f in fr:
                st["img"] = ImageTk.PhotoImage(Image.fromarray(f).resize((512, 480), Image.NEAREST)); gl.configure(image=st["img"])
                pump(); time.sleep(max(0.0, (1 / 60) / a.speed - (time.time() - t0))); t0 = time.time()
        head.configure(text=f"[{tagline}] coach on {r['level']} start {r['start']}: {r['end'].upper()} at x {r['max_x']}"
                            f" after {r['decisions']} decisions ({r['secs']:.0f} s of thinking)")
        for _ in range(60):
            pump(); time.sleep(1 / 60)
        g.close()


if __name__ == "__main__":
    main()
