"""At a level change: record clips (solo clears, the best first-sight attempt on a new level), rebuild
docs/mario/levels. python levels_update.py"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ST = HERE / "runs/inorder/state.json"; OUT = HERE.parent / "docs/mario/levels"
st = json.loads(ST.read_text()); PY = sys.executable


def rec(args, name, caption):
    mp4 = OUT / f"{name}.mp4"
    if mp4.exists():
        return
    r = subprocess.run([PY, str(HERE / "record_level.py"), "--state", "runs/inorder/state.json", *args, "--out", str(mp4)],
                       cwd=HERE, capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-400:])
    js = mp4.with_suffix(".json")
    if js.exists():
        m = json.loads(js.read_text()); m["caption"] = caption; js.write_text(json.dumps(m))


for name, l in st["levels"].items():
    c = l.get("cleared")
    if c:
        rec(["--level", name, "--round", str(c["round"])], f"clear-{name}",
            f"Solo clear: {name}, {'first sight' if c['round'] == 0 else 'practice round ' + str(c['round'])}, flag")
# best first sight on a level it had never practised (1-1 excluded: trained overnight)
fs = [(l["first_sight"]["score"][2], max(e["max_x"] for e in l["first_sight"]["eval"]), n)
      for n, l in st["levels"].items() if "first_sight" in l and n != "1-1"]
if fs:
    _, bx, bn = max(fs, key=lambda t: t[1])
    tag = f"firstsight-{bn}"
    for old in OUT.glob("firstsight-*"):
        if not old.name.startswith(tag):
            old.unlink()
    rec(["--level", bn, "--round", "0"], tag, f"Best first sight so far: {bn} before any practice on it, x {bx:,}")
subprocess.run([PY, str(HERE / "build_levels_page.py"), "--state", "runs/inorder/state.json", "--out", str(OUT)], cwd=HERE)
