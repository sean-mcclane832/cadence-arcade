"""Live viewer: a saved brain plays Mario in a window while a second window shows what the brain is doing.

    python live.py --brain runs/LivyA/LivyA_ck0200.npz --name LivyA            # 12 training levels, 5 tries each
    python live.py --brain ... --name LivyA --first 1-1 --last 1-4 --tries 5 --speed 1.5
    python live.py --follow CiceroLowB        # a run in training: before every try, load its newest checkpoint;
                                              # loop the levels until q (one try each)

Same player and start rules as cgauntlet.py (greedy act mode, memories on, a fresh level start per try; try k starts
after RETRY_STARTS[k-1] idle frames because the emulator is deterministic). Unlike the gauntlet, every level gets all its
tries whether or not an earlier try cleared it. Nothing is learned and nothing is saved except a JSON summary.

Keys (click either window first): space = pause, n = skip to next level, q = quit, + / - = speed.
Window 1: the NES game. Window 2: the brain: what it sees (now / previous decision), the association and
thought-memory regions (real activations after the settle), the echo, button memory, and the 7 motor neurons.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "memory"))
sys.path.insert(0, str(HERE.parent / "concepts")); sys.path.insert(0, str(HERE.parent / "viz"))
import cplay as C  # noqa: E402
import dagger as D  # noqa: E402
import mem  # noqa: E402
from game import ACTIONS, ACTION_NAMES, N_FRAME  # noqa: E402
from brain_video import colorize, grid_img, grey_img  # noqa: E402
from cadence.recording import record_settlements  # noqa: E402
from PIL import Image, ImageDraw, ImageTk  # noqa: E402
import tkinter as tk  # noqa: E402

RETRY_STARTS = (0, 37, 83, 110, 131)
ORDER = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
BG, FG, DIM, HI = (24, 26, 32), (225, 228, 235), (130, 136, 150), (255, 200, 80)


class Viewer:
    def __init__(self, title, speed):
        self.root = tk.Tk(); self.root.title(f"{title}: game"); self.root.geometry("+20+20")
        self.head = tk.Label(self.root, text="", font=("Consolas", 12), bg="#181a20", fg="#e1e4eb", anchor="w", width=60)
        self.head.pack(fill="x")
        self.gl = tk.Label(self.root, bg="black"); self.gl.pack()
        self.brain_win = tk.Toplevel(self.root); self.brain_win.title(f"{title}: brain"); self.brain_win.geometry("+560+20")
        self.bl = tk.Label(self.brain_win, bg="black"); self.bl.pack()
        self.paused = False; self.skip = False; self.quit = False; self.speed = speed
        for w in (self.root, self.brain_win):
            w.bind("<space>", lambda e: setattr(self, "paused", not self.paused))
            w.bind("n", lambda e: setattr(self, "skip", True))
            w.bind("q", lambda e: setattr(self, "quit", True))
            w.bind("+", lambda e: setattr(self, "speed", min(8.0, self.speed * 1.25)))
            w.bind("-", lambda e: setattr(self, "speed", max(0.1, self.speed / 1.25)))
            w.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "quit", True))
        self._g = self._b = None

    def game(self, frame):
        img = Image.fromarray(frame).resize((512, 480), Image.NEAREST)
        self._g = ImageTk.PhotoImage(img); self.gl.configure(image=self._g)

    def brain(self, img):
        self._b = ImageTk.PhotoImage(img); self.bl.configure(image=self._b)

    def pump(self):
        self.root.update()
        while self.paused and not self.quit:
            time.sleep(0.05); self.root.update()


def brain_panel(pl, pops, o, fin, a, text):
    """One picture of the brain at this decision: inputs, regions, motor neurons."""
    W, H = 600, 560
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((10, 8), text, fill=FG)
    d.text((10, 30), "what it sees now", fill=DIM); d.text((250, 30), "previous decision", fill=DIM)
    img.paste(grey_img(3 * o[:N_FRAME], 26, 32, 7), (10, 46))
    img.paste(grey_img(3 * o[N_FRAME:2 * N_FRAME], 26, 32, 7), (250, 46))
    y = 242
    d.text((10, y), "association (256)", fill=(255, 150, 60)); d.text((170, y), "thought memory (256)", fill=(170, 120, 255))
    img.paste(grid_img(fin[pops["association"]], 16, 16, 9), (10, y + 16))
    img.paste(grid_img(fin[pops["prefrontal"]], 16, 16, 9), (170, y + 16))
    if "history" in pops:
        d.text((330, y), "echo (last decision)", fill=(90, 170, 255))
        img.paste(grid_img(fin[pops["history"][:832]], 26, 32, 5), (330, y + 16))
    if "efference" in pops:
        yy = y + 16 + 140 + 6
        d.text((330, yy), "button memory (what it pressed)", fill=DIM)
        for i, v in enumerate(fin[pops["efference"]]):
            c = int(np.clip(v, 0, 1) * 255)
            d.rectangle([330 + i * 24, yy + 16, 330 + i * 24 + 21, yy + 34], fill=(c, int(c * 0.75), 40) if v > 0.02 else (34, 36, 44))
    by = 424
    d.text((10, by - 18), "7 motor neurons (the brain presses the highest)", fill=DIM)
    mot = fin[pops["motor"]]
    for i, (name, v) in enumerate(zip(ACTION_NAMES, mot)):
        yy = by + i * 18
        d.text((10, yy), name, fill=HI if i == a else FG)
        d.rectangle([120, yy + 2, 360, yy + 13], fill=(30, 33, 41))
        zero = 120 + int(240 * 0.1 / 1.1); end = 120 + int(240 * (float(v) + 0.1) / 1.1)
        d.rectangle([min(zero, end), yy + 2, max(zero, end), yy + 13], fill=HI if i == a else (95, 100, 115))
        d.text((368, yy), f"{v:+.2f}", fill=HI if i == a else DIM)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=None); ap.add_argument("--name", default=None)
    ap.add_argument("--follow", default=None, help="run name under runs/: always play its newest checkpoint")
    ap.add_argument("--first", default="1-1"); ap.add_argument("--last", default="3-4")
    ap.add_argument("--tries", type=int, default=5); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="first play seed (follow-run.sh varies it so replays differ)")
    a = ap.parse_args()
    order = ORDER[ORDER.index(a.first):ORDER.index(a.last) + 1]

    def newest():
        cks = sorted((HERE / "runs" / a.follow).glob(f"{a.follow}_ck[0-9]*.npz"))
        return str(cks[-1]) if cks else None

    if a.follow:
        a.name = a.name or a.follow; a.tries = 1; path = newest()
        import itertools
        order = itertools.cycle(order)
    else:
        path = str((HERE / a.brain).resolve()) if not Path(a.brain).is_absolute() else a.brain
    player = mem.MemPlayer(path, "greedy", 0.05, (), None)
    label = a.name if not a.follow else f"{a.name} chunk {int(Path(path).stem.split('_ck')[1])}"
    pops = {k: np.asarray(v, np.int32) for k, v in player.b.connectome.populations.items()}
    motor = np.asarray(player.b.motor_index)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    v = Viewer(a.name, a.speed)
    log = []; seed = a.seed
    for L in order:
        lvl = tuple(int(x) for x in L.split("-")); v.skip = False
        for t in range(a.tries):
            if v.quit or v.skip:
                break
            if a.follow:   # a newer checkpoint since the last try: switch to it
                ck = newest()
                if ck and ck != path:
                    path = ck; player = mem.MemPlayer(path, "greedy", 0.05, (), None)
                    label = f"{a.name} chunk {int(Path(path).stem.split('_ck')[1])}"
                    print(f"switched to {Path(path).name}", flush=True)
            st = RETRY_STARTS[min(t, len(RETRY_STARTS) - 1)]
            b = C.CBody(st, "body", "lean", lvl); g = b.g
            player.begin(seed=1000 + seed); seed += 1
            n = 0
            while not g.done and g.decisions < D.MAX_DECISIONS and g.stuck < D.STUCK and not (v.quit or v.skip):
                o = b.obs()
                if player.xkey:
                    o = np.concatenate([o, mem.xblock(g.env.ram)])
                recs = []
                with record_settlements(recs.append, label="live"):
                    act = player.act(o)
                r = max(recs, key=lambda r: r.steps)
                fin = np.asarray(r.activation)[-1, 0, :].astype(np.float32)
                text = f"{label} | {L} try {t + 1}/{a.tries} | x {g.x} | decision {g.decisions}"
                v.brain(brain_panel(player, pops, o, fin, int(act), text))
                fr = []
                g.step(act, keep_frames=fr); b.held = ACTIONS[act]
                v.head.configure(text=f"{label} | {L}  try {t + 1}/{a.tries} | x {g.x} | {ACTION_NAMES[act]}   [space pause, n next level, q quit, +/- speed {v.speed:.2f}x]")
                t0 = time.time()
                for f in fr:
                    v.game(f); v.pump()
                    time.sleep(max(0.0, (1 / 60) / v.speed - (time.time() - t0))); t0 = time.time()
                    if v.quit:
                        break
            end = "flag" if g.flag else ("died" if g.done else ("stuck" if g.stuck >= D.STUCK else "cap"))
            oe = b.obs()
            if player.xkey:
                oe = np.concatenate([oe, mem.xblock(g.env.ram)])
            player.end(oe, end)
            log.append(dict(level=L, try_=t + 1, start=st, end=end, max_x=int(g.max_x), decisions=g.decisions,
                            brain=Path(path).name))
            print(f"{L} try {t + 1}: {end} x {g.max_x}", flush=True)
            v.head.configure(text=f"{label} | {L} try {t + 1}: {end.upper()} at x {g.max_x}")
            for _ in range(40):
                try:
                    s_, _, _, _ = g.env.step(0)
                except Exception:  # noqa: BLE001
                    break
                v.game(s_); v.pump(); time.sleep(1 / 60 / v.speed)
                if v.quit or v.skip:
                    break
            g.close()
        if v.quit:
            break
    summ = {}
    for e in log:
        s = summ.setdefault(e["level"], dict(flags=0, tries=0, best_x=0))
        s["tries"] += 1; s["flags"] += e["end"] == "flag"; s["best_x"] = max(s["best_x"], e["max_x"])
    out = Path(a.out) if a.out else HERE / "runs" / (a.follow or a.name) / f"live_{time.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(dict(name=a.name, brain=path, tries=log, summary=summ), indent=1))
    print(json.dumps(summ), flush=True)
    print("saved", out, flush=True)
    try:
        v.root.destroy()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
