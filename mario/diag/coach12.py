"""Can the coach itself clear 1-2 from the TEST starts? (no random buttons)"""
import sys, json, time
sys.path.insert(0, "..")
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import dagger as D
from game import MENUS
if __name__ == "__main__":
    mb = sorted({a for p in MENUS["lean"] for a in p})
    starts = [int(s) for s in sys.argv[1].split(",")]
    ex = ProcessPoolExecutor(len(starts), mp_context=get_context("spawn"), initializer=D._worker_init)
    MENU = sys.argv[3] if len(sys.argv) > 3 else "lean"
    ts = [dict(senses="body", menu=MENU, kind="cadence", menu_buttons=mb, level=tuple(int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1-2").split("-")), mode="coach", start=s, eps=0.0, seed=s) for s in starts]
    t = time.time()
    for s, r in zip(starts, ex.map(D.episode, ts)):
        print(json.dumps(dict(start=s, max_x=r["max_x"], end=r["end"], decisions=r["decisions"], s=round(time.time() - t))), flush=True)
