"""Build docs/mario/levels/index.html from the in-order run (levels.py state.json).

    python build_levels_page.py --state runs/inorder/state.json --out ../docs/mario/levels

Videos: every *.mp4 in the out folder with a .json next to it (record_level.py writes both) is
listed, captioned from that json. Notes: levels_text.json (a list of plain lines) if present.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def med(xs):
    return int(np.median(xs)) if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/inorder/state.json")
    ap.add_argument("--out", default="../docs/mario/levels")
    ap.add_argument("--text", default="levels_text.json")
    args = ap.parse_args()
    st = json.loads((HERE / args.state).read_text())
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    levels = sorted(st["levels"].items(), key=lambda kv: kv[1]["index"])
    hist = st["history"]

    rows = []
    for name, rec in levels:
        fs = rec.get("first_sight")
        if not fs:
            continue
        fx = [e["max_x"] for e in fs["eval"]]
        solo = [h for h in hist if h["level"] == name]
        best = max(solo, key=lambda h: tuple(h["score"]))
        coach_ok = name not in st.get("no_coach", [])
        rows.append(dict(name=name, idx=rec["index"], fs=fx, fs_med=med(fx), fs_best=max(fx),
                         fs_flags=sum(e["flag"] for e in fs["eval"]),
                         best_med=int(best["score"][1]), best_round=best["round"],
                         best_flags=int(best["score"][0]), rounds=rec.get("rounds", 0),
                         practice=rec.get("practice_s", 0) / 60, cleared=rec.get("cleared"),
                         capped=rec.get("capped", False), current=(rec["index"] == st["level"]),
                         coached=rec.get("coached_attempts", 0), labels=rec.get("coach_labels", 0),
                         coach_ok=coach_ok, own=rec.get("self_rows", 0),
                         old_dec=rec.get("old_play_decisions", 0), self_dec=rec.get("self_decisions", 0)))

    # -- chart: first-sight median (bar) and best-of-3 (tick) per level, in play order
    W, H, padl, padb = 360, 200, 40, 34
    ymax = max([r["fs_best"] for r in rows] + [r["best_med"] for r in rows] + [500])
    ymax = int(np.ceil(ymax / 500) * 500)
    n = max(1, len(rows)); bw = min(44.0, (W - padl - 8) / n)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="First-sight distance per level, in play order">']
    for t in range(0, ymax + 1, 500 if ymax <= 2500 else 1000):
        y = H - padb - (H - padb - 10) * t / ymax
        svg.append(f'<line x1="{padl}" x2="{W - 4}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{padl - 4}" y="{y + 3:.1f}" class="ax" text-anchor="end">{t}</text>')
    for i, r in enumerate(rows):
        x = padl + i * bw + bw * 0.15; w = bw * 0.7
        hgt = (H - padb - 10) * r["fs_med"] / ymax
        svg.append(f'<rect x="{x:.1f}" y="{H - padb - hgt:.1f}" width="{w:.1f}" height="{hgt:.1f}" class="bar">'
                   f'<title>{r["name"]} first sight median x {r["fs_med"]} (attempts {r["fs"]})</title></rect>')
        yb = H - padb - (H - padb - 10) * r["fs_best"] / ymax
        svg.append(f'<line x1="{x:.1f}" x2="{x + w:.1f}" y1="{yb:.1f}" y2="{yb:.1f}" class="tick"/>')
        svg.append(f'<text x="{x + w / 2:.1f}" y="{H - padb + 13}" class="ax" text-anchor="middle">{r["name"]}</text>')
        if r["idx"] == 0:
            svg.append(f'<text x="{x + w / 2:.1f}" y="{H - padb + 25}" class="ax" text-anchor="middle">pre</text>')
    svg.append("</svg>")

    def cleared(r):
        if r["cleared"]:
            c = r["cleared"]
            return f'<span class="ok">yes</span>, {"first sight" if c["round"] == 0 else "round " + str(c["round"])} ({html.escape(c["date"][11:])})'
        if r["current"]:
            return '<span class="dim">practising</span>'
        return '<span class="no">no</span> (cap)' if r["capped"] else '<span class="no">no</span>'

    trs = "".join(
        f'<tr><td>{r["name"]}</td><td class="n">{r["fs_med"]:,}</td><td class="n">{r["best_med"]:,}'
        f'<span class="dim"> r{r["best_round"]}</span></td><td>{cleared(r)}</td>'
        f'<td class="n">{r["practice"]:.0f}</td></tr>' for r in rows)
    crs = "".join(
        f'<tr><td>{r["name"]}</td><td class="n">{r["coached"] if r["coach_ok"] else "none"}</td>'
        f'<td class="n">{r["labels"]:,}</td><td class="n">{r["own"]:,}</td>'
        f'<td class="n">{(100 * r["old_dec"] / max(1, r["self_dec"] + r["old_dec"])):.0f}%</td></tr>' for r in rows)

    ret_html = '<p class="dim">Measured at the first level change.</p>'
    if st["retention"]:
        last = st["retention"][-1]
        cells = []
        for name, v in last["results"].items():
            best = next((r for r in rows if r["name"] == name), None)
            cells.append(f'<tr><td>{name}</td><td class="n">{med(v["xs"]):,}</td>'
                         f'<td class="n">{best["best_med"] if best else 0:,}</td><td class="n">{v["flags"]}/3</td></tr>')
        ret_html = (f'<p class="dim">Solo again, after leaving {last["after"]} '
                    f'({time.strftime("%H:%M", time.localtime(last["time"]))}). Median x of 3.</p>'
                    '<div class="tw"><table><thead><tr><th>level</th><th>now</th><th>its best</th><th>flags</th></tr></thead>'
                    f'<tbody>{"".join(cells)}</tbody></table></div>')

    vids = []
    for js in sorted(out.glob("*.json")):
        mp4 = js.with_suffix(".mp4")
        if not mp4.exists():
            continue
        m = json.loads(js.read_text())
        poster = js.with_name(js.stem + "_poster.png")
        what = m.get("caption") or (f'{m["level"]}, round {m["round"]}, start {m["start"]}: '
                                    f'{"FLAG" if m["flag"] else "x " + format(m["max_x"], ",")}')
        vids.append(f'<figure><video controls playsinline muted preload="metadata"'
                    f'{" poster=" + chr(34) + poster.name + chr(34) if poster.exists() else ""}>'
                    f'<source src="{mp4.name}" type="video/mp4"></video>'
                    f'<figcaption>{html.escape(what)}. Every button is the brain\'s. {m["seconds"]} s.</figcaption></figure>')
    vid_html = "".join(vids) or '<p class="dim">No clip yet.</p>'

    tp = HERE / args.text
    lines = json.loads(tp.read_text()) if tp.exists() else []
    notes = "".join(f"<li>{html.escape(l)}</li>" for l in lines)
    now = time.strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cadence Plays Mario In Order</title>
<meta name="description" content="One Cadence brain plays Super Mario Bros. levels in order, like a person. How far does it get on each new level at first sight?">
<meta name="color-scheme" content="dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Silkscreen&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
  :root {{ --bg: #0b0d10; --fg: #e8e6e1; --dim: #8a8f98; --accent: #ffb347; --line: #23272e; --bad: #ff6b6b; --good: #7bd88f; --panel: #12151a; }}
  * {{ box-sizing: border-box; }}
  html, body {{ overflow-x: hidden; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.6 "JetBrains Mono", ui-monospace, monospace; padding: 28px 16px 40px; }}
  main {{ max-width: 640px; margin: 0 auto; }}
  h1 {{ font-family: "Silkscreen", monospace; font-weight: 400; font-size: 22px; letter-spacing: .05em; margin: 0 0 6px; }}
  h2 {{ font-family: "Silkscreen", monospace; font-weight: 400; font-size: 15px; letter-spacing: .05em; margin: 30px 0 8px; }}
  .sub, .dim {{ color: var(--dim); font-size: 12.5px; }}
  .sub {{ margin: 0 0 16px; }}
  a {{ color: var(--accent); }}
  svg {{ width: 100%; height: auto; display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
  svg .grid {{ stroke: var(--line); }} svg .ax {{ fill: var(--dim); font: 9px "JetBrains Mono", monospace; }}
  svg .bar {{ fill: var(--accent); }} svg .tick {{ stroke: var(--fg); stroke-width: 2; }}
  .tw {{ width: 100%; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--dim); font-weight: 400; }}
  td.n, th.n {{ text-align: right; white-space: nowrap; }}
  .ok {{ color: var(--good); }} .no {{ color: var(--bad); }}
  ul {{ padding-left: 18px; margin: 10px 0; }} li {{ margin: 7px 0; }}
  figure {{ margin: 0 0 18px; }}
  video {{ display: block; width: 100%; aspect-ratio: 256 / 240; background: #000; border: 1px solid var(--line); border-radius: 6px; image-rendering: pixelated; }}
  figcaption {{ color: var(--dim); font-size: 12.5px; margin-top: 6px; }}
  footer {{ color: var(--dim); font-size: 12px; margin-top: 32px; }} footer a {{ color: var(--dim); }}
</style>
</head>
<body>
<main>
  <p class="sub"><a href="../../">cadence arcade</a> / <a href="../">mario</a> / levels in order</p>
  <h1>MARIO, LEVEL BY LEVEL</h1>
  <p class="sub">One Cadence brain (cadence-net {html.escape(st.get("cadence", ""))}) plays the levels in order, learning the whole time. Updated {now}.</p>

  <h2>First sight</h2>
  <p class="dim">How far it gets on a level it has never trained on: median x of 3 solo attempts (bar), best of the 3 (white line). 1-1 is marked "pre": the brain started from last night's 1-1 training.</p>
  {"".join(svg)}

  <h2>What's happening</h2>
  <ul>{notes}</ul>

  <h2>Each level</h2>
  <div class="tw"><table><thead><tr><th>level</th><th class="n">first sight</th><th class="n">best solo</th><th>cleared solo</th><th class="n">min</th></tr></thead>
  <tbody>{trs}</tbody></table></div>
  <p class="dim">Distances are Mario's x in pixels, median of 3 solo attempts. "Cleared" = flag in 2 of 3 solo attempts. "min" = practice minutes.</p>

  <h2>Videos</h2>
  {vid_html}

  <h2>Coaching used</h2>
  <div class="tw"><table><thead><tr><th>level</th><th class="n">coached</th><th class="n">labels</th><th class="n">own</th><th class="n">old play</th></tr></thead>
  <tbody>{crs}</tbody></table></div>
  <p class="dim">Coached: tries where the brain plays and a look-ahead planner writes down the button it would have pressed on each screen; the brain never gets driven by it. Labels: coach labels written. Own: screens from the brain's own attempts that made progress. Old-level play: share of its self-play spent on an earlier level.</p>

  <h2>Retention</h2>
  {ret_html}

  <footer>Independent experiment by <a href="https://github.com/sean-mcclane832">sean-mcclane832</a>. Not affiliated with Pragma Research. Code: <a href="https://github.com/sean-mcclane832/cadence-arcade/tree/master/mario">cadence-arcade/mario</a> (levels.py).</footer>
</main>
</body>
</html>
"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {out / 'index.html'} ({len(rows)} levels, {len(vids)} videos)")


if __name__ == "__main__":
    main()
