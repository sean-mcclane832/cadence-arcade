"""Build docs/mario/levels/index.html: BEFORE (blind run) vs AFTER (new senses) per level, with
first-sight baselines.

    python build_levels_page.py --before runs/inorder_blind/state.json --after runs/inorder/state.json \
        --baselines runs/baselines_levels.json --out ../docs/mario/levels

Videos: every *.mp4 in the out folder with a .json next to it (record_level.py writes both).
Notes: levels_text.json (a list of plain lines).
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_KEYS = (("run", "run"), ("runjump", "jump"), ("random", "random"), ("fresh", "fresh"), ("fresh_s", "fresh+"))


def med(xs):
    return int(np.median(xs)) if len(xs) else 0


def summarize(st):
    if st is None:
        return {}
    rows = {}
    for name, rec in st["levels"].items():
        fs = rec.get("first_sight")
        if not fs:
            continue
        solo = [h for h in st["history"] if h["level"] == name]
        best = max(solo, key=lambda h: tuple(h["score"]))
        rows[name] = dict(idx=rec["index"], fs_med=med([e["max_x"] for e in fs["eval"]]),
                          fs_best=max(e["max_x"] for e in fs["eval"]), fs_flags=sum(e["flag"] for e in fs["eval"]),
                          best_med=int(best["score"][1]), best_round=best["round"], cleared=rec.get("cleared"),
                          capped=rec.get("capped", False), current=(rec["index"] == st["level"]),
                          practice=rec.get("practice_s", 0) / 60, coached=rec.get("coached_attempts", 0),
                          labels=rec.get("coach_labels", 0), own=rec.get("self_rows", 0),
                          old_dec=rec.get("old_play_decisions", 0), self_dec=rec.get("self_decisions", 0))
    return rows


def video_html(out):
    """Clips in the out folder (every *.mp4 with a .json next to it, from record_level.py).
    Clips whose json has a caption belong to the in-order run; the rest are learning-fix brains
    (captioned with care on the showcase page), listed here by brain name."""
    run_v, other_v = [], []
    for js in sorted(Path(out).glob("*.json")):
        mp4 = js.with_suffix(".mp4")
        if not mp4.exists():
            continue
        m = json.loads(js.read_text())
        poster = js.with_name(js.stem + "_poster.png")
        pa = f' poster="{poster.name}"' if poster.exists() else ""
        if m.get("caption"):
            what = m["caption"]
            dest = run_v
        else:
            run = str(m.get("run", ""))
            brain = Path(run).stem if run.endswith(".npz") else (Path(run).name + f", round {m['round']}" if run else f"round {m['round']}")
            end = "flag" if m.get("flag") else f'x {m["max_x"]:,}'
            what = f'{brain} on {m["level"]}, start {m["start"]}: {end}'
            dest = other_v
        dest.append(f'<figure><video controls playsinline muted preload="metadata"{pa}>'
                    f'<source src="{mp4.name}" type="video/mp4"></video>'
                    f"<figcaption>{html.escape(what)}. Every button is the brain's. {m['seconds']} s.</figcaption></figure>")
    parts = []
    parts.append('<div class="clips">' + "".join(run_v) + '</div>' if run_v else '<p class="dim">No clip yet.</p>')
    if other_v:
        parts.append('<h3>Learning-fix brains</h3><p class="small dim">Solo clips of the later brains, by file name. '
                     'The <a href="../">Mario page</a> has them with context and the brain roster.</p>'
                     '<div class="clips">' + "".join(other_v) + '</div>')
    return "\n  ".join(parts), len(run_v) + len(other_v)


def load(p):
    p = HERE / p if p else None
    return json.loads(p.read_text()) if p and p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="runs/inorder_blind/state.json")
    ap.add_argument("--after", default="runs/inorder/state.json")
    ap.add_argument("--baselines", default="runs/baselines_levels.json")
    ap.add_argument("--out", default="../docs/mario/levels")
    ap.add_argument("--text", default="levels_text.json")
    args = ap.parse_args()
    sb, sa = load(args.before), load(args.after)
    if sa is not None and sa.get("senses", "blind") == "blind" and sb is None:
        sb, sa = sa, None  # only the blind run so far
    base = load(args.baselines) or {}
    for nm, v in (load(args.baselines.replace(".json", "_senses.json")) or {}).items():
        if nm in base and "fresh" in v.get("median", {}):
            base[nm]["median"]["fresh_s"] = v["median"]["fresh"]
    B, A = summarize(sb), summarize(sa)
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    names = sorted(set(B) | set(A), key=lambda n: (int(n[0]), int(n[2])))

    def best_base(n):
        m = base.get(n, {}).get("median", {})
        return max(m.values()) if m else None

    # -- chart: per level, blind first sight, senses first sight, best baseline tick
    W, H, padl, padb = 360, 210, 40, 30
    vals = [B[n]["fs_med"] for n in names if n in B] + [A[n]["fs_med"] for n in names if n in A] + \
           [best_base(n) or 0 for n in names] + [500]
    ymax = int(np.ceil(max(vals) / 500) * 500)
    n = max(1, len(names)); gw = min(60.0, (W - padl - 8) / n)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="First-sight distance per level: blind brain, new-senses brain, best baseline">']
    step = 500 if ymax <= 2500 else 1000
    for t in range(0, ymax + 1, step):
        y = H - padb - (H - padb - 10) * t / ymax
        svg.append(f'<line x1="{padl}" x2="{W - 4}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{padl - 4}" y="{y + 3:.1f}" class="ax" text-anchor="end">{t}</text>')
    for i, nm in enumerate(names):
        x0 = padl + i * gw + gw * 0.1; bw = gw * 0.38
        for j, (rows, cls) in enumerate(((B, "bar-b"), (A, "bar-a"))):
            if nm in rows:
                hgt = (H - padb - 10) * rows[nm]["fs_med"] / ymax
                svg.append(f'<rect x="{x0 + j * bw:.1f}" y="{H - padb - hgt:.1f}" width="{bw - 1:.1f}" height="{hgt:.1f}" class="{cls}">'
                           f'<title>{nm} first sight {rows[nm]["fs_med"]}</title></rect>')
        bb = best_base(nm)
        if bb:
            yb = H - padb - (H - padb - 10) * bb / ymax
            svg.append(f'<line x1="{x0 - 2:.1f}" x2="{x0 + 2 * bw + 1:.1f}" y1="{yb:.1f}" y2="{yb:.1f}" class="tick"><title>best baseline {bb:.0f}</title></line>')
        svg.append(f'<text x="{x0 + bw:.1f}" y="{H - padb + 13}" class="ax" text-anchor="middle">{nm}</text>')
    svg.append("</svg>")
    legend = ('<p class="legend"><span class="sw b"></span>blind brain <span class="sw a"></span>new senses '
              '<span class="sw t"></span>best baseline</p>')

    def fs_cell(rows, nm):
        if nm not in rows:
            return '<td class="n dim">-</td>'
        v = rows[nm]["fs_med"]; bb = best_base(nm)
        mark = "" if bb is None else (' <span class="ok">&gt;</span>' if v > bb else ' <span class="no">&le;</span>')
        pre = ' <span class="dim">pre</span>' if (rows is B and rows[nm]["idx"] == 0) else ""
        return f'<td class="n">{v:,}{mark}{pre}</td>'

    fs_rows = []
    for nm in names:
        m = base.get(nm, {}).get("median", {})
        cells = "".join(f'<td class="n">{int(m[k]):,}</td>' if k in m else '<td class="n dim">-</td>' for k, _ in BASE_KEYS)
        fs_rows.append(f"<tr><td>{nm}</td>{fs_cell(B, nm)}{fs_cell(A, nm)}{cells}</tr>")

    def clr(r, stopped=False):
        if r is None:
            return '<td class="dim">-</td>'
        if r["cleared"]:
            c = r["cleared"]
            return f'<td><span class="ok">yes</span> {"first sight" if c["round"] == 0 else "r" + str(c["round"])} {html.escape(c["date"][11:])}</td>'
        if r["current"] and not r["capped"]:
            return '<td class="dim">stopped</td>' if stopped else '<td class="dim">practising</td>'
        return '<td><span class="no">no</span></td>'

    pr_rows = "".join(
        f'<tr><td>{nm}</td><td class="n">{(str(format(B[nm]["best_med"], ",")) if nm in B else "-")}</td>{clr(B.get(nm), True)}'
        f'<td class="n">{(str(format(A[nm]["best_med"], ",")) if nm in A else "-")}</td>{clr(A.get(nm), True)}</tr>' for nm in names)

    def coach_rows(rows):
        return "".join(
            f'<tr><td>{nm}</td><td class="n">{r["coached"]}</td><td class="n">{r["labels"]:,}</td><td class="n">{r["own"]:,}</td>'
            f'<td class="n">{100 * r["old_dec"] / max(1, r["self_dec"] + r["old_dec"]):.0f}%</td><td class="n">{r["practice"]:.0f}</td></tr>'
            for nm, r in sorted(rows.items(), key=lambda kv: kv[1]["idx"]))

    def coach_table(rows, title):
        if not rows:
            return ""
        return (f'<p class="dim">{title}</p><div class="tw"><table><thead><tr><th>level</th><th class="n">coached</th><th class="n">labels</th>'
                f'<th class="n">own</th><th class="n">old play</th><th class="n">min</th></tr></thead><tbody>{coach_rows(rows)}</tbody></table></div>')

    def ret_table(st, title):
        if st is None or not st["retention"]:
            return ""
        last = st["retention"][-1]; rows = summarize(st)
        cells = "".join(f'<tr><td>{nm}</td><td class="n">{med(v["xs"]):,}</td><td class="n">{rows.get(nm, {}).get("best_med", 0):,}</td>'
                        f'<td class="n">{v["flags"]}/3</td></tr>' for nm, v in last["results"].items())
        return (f'<p class="dim">{title}: solo again after leaving {last["after"]} '
                f'({time.strftime("%H:%M", time.localtime(last["time"]))}), median x of 3.</p>'
                '<div class="tw"><table><thead><tr><th>level</th><th class="n">now</th><th class="n">its best</th><th class="n">flags</th></tr></thead>'
                f'<tbody>{cells}</tbody></table></div>')

    vid_html, n_vids = video_html(out)
    tp = HERE / args.text
    notes = "".join(f"<li>{html.escape(l)}</li>" for l in (json.loads(tp.read_text()) if tp.exists() else []))
    now = time.strftime("%Y-%m-%d %H:%M")
    cad = html.escape((sa or sb or {}).get("cadence", ""))
    fix_p = HERE / "levels_fix.html"  # the 2026-10-30 learning fix section (hand-written fragment)
    fix_html = fix_p.read_text(encoding="utf-8") if fix_p.exists() else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cadence Plays Mario In Order</title>
<meta name="description" content="One Cadence brain plays Super Mario Bros. levels in order. How far does it get on each new level at first sight, against simple baselines, before and after better senses?">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Silkscreen&display=swap">
<link rel="stylesheet" href="../../assets/site.css">
<style>
  /* page tokens mapped onto the shared site tokens (assets/site.css) */
  :root {{ --dim: var(--c-dim); --accent: var(--c-accent-bg); --blue: var(--c-blue); --line: var(--c-line); --panel: var(--c-surface); }}
  svg {{ width: 100%; height: auto; display: block; background: var(--panel); border: 1px solid var(--line); border-radius: var(--c-radius); }}
  svg .grid {{ stroke: var(--line); }} svg .ax {{ fill: var(--dim); font: 9px var(--c-mono); }}
  svg .bar-b {{ fill: var(--dim); }} svg .bar-a {{ fill: var(--accent); }} svg .tick {{ stroke: var(--blue); stroke-width: 2.5; }}
  .legend {{ color: var(--dim); font-size: 13px; margin: 8px 0 0; }}
  .sw {{ display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; vertical-align: -1px; }}
  .sw.b {{ background: var(--dim); margin-left: 0; }} .sw.a {{ background: var(--accent); }} .sw.t {{ background: var(--blue); height: 3px; vertical-align: 2px; }}
  .site video {{ aspect-ratio: 256 / 240; image-rendering: pixelated; object-fit: contain; }}
</style>
</head>
<body class="site">
<!--site-header-->
<header class="site-header"><div class="site-header-in">
  <a class="site-logo" href="../../">CADENCE<span>ARCADE</span></a>
  <nav class="site-nav" aria-label="Site"><a href="../" aria-current="page">Mario</a><a href="../../brain/">Brain scope</a><a href="../../chat/">Chat</a><a href="../../ember-field/">Ember Field</a><a href="../../breakout/">Breakout</a></nav>
</div></header>
<!--/site-header-->
<main class="page">
  <p class="crumbs"><a href="../../">Home</a> / <a href="../">Mario</a> / Level by level</p>
  <h1>Mario, level by level</h1>
  <p class="lede">One Cadence brain (cadence-net {cad}) plays the levels in order, learning the whole time. Two runs: a nearly blind brain (before), then a fresh brain with better senses (after). Updated {now}.</p>
{fix_html}
  <h2>First sight</h2>
  <p class="small dim">How far it gets on a level before any practice on it: median x of 3 solo attempts.</p>
  {"".join(svg)}
  {legend}

  <h2>What's happening</h2>
  <ul>{notes}</ul>

  <h2>First sight vs baselines</h2>
  <div class="tw"><table><thead><tr><th>level</th><th class="n">blind</th><th class="n">senses</th><th class="n">run</th><th class="n">jump</th><th class="n">random</th><th class="n">fresh</th><th class="n">fresh+</th></tr></thead>
  <tbody>{"".join(fs_rows)}</tbody></table></div>
  <p class="small dim">Median x, same 3 starts as first sight. run = hold run right; jump = hold run+jump; random = random buttons (10 attempts, fixed seeds); fresh = an untrained Cadence brain with the blind input; fresh+ = an untrained brain with the new senses. <span class="ok">&gt;</span> beats the best baseline, <span class="no">&le;</span> does not. The blind brain's 1-1 is "pre": it began from last night's 1-1 training.</p>

  <h2>After practice</h2>
  <div class="tw"><table><thead><tr><th>level</th><th class="n">blind best</th><th>cleared</th><th class="n">senses best</th><th>cleared</th></tr></thead>
  <tbody>{pr_rows}</tbody></table></div>
  <p class="small dim">Best solo median x over the practice rounds (about 38 minutes per level). Cleared = flag in 2 of 3 solo attempts.</p>

  <h2>Videos</h2>
  {vid_html}

  <h2>Coaching used</h2>
  {coach_table(A, "New senses")}
  {coach_table(B, "Blind")}
  <p class="small dim">coached: tries where the brain plays and a look-ahead planner writes down the button it would have pressed on each screen; it never takes the controls. labels: buttons it wrote. own: screens from the brain's own attempts that made progress. old play: share of self-play spent on an earlier level. min: practice minutes.</p>

  <h2>Retention</h2>
  {ret_table(sa, "New senses")}
  {ret_table(sb, "Blind")}

</main>
<!--site-footer-->
<footer class="site-footer"><div class="site-footer-in">
  <p>Independent experiment by <a href="https://github.com/sean-mcclane832">sean-mcclane832</a>. Not affiliated with Pragma Research. Cadence: <a href="https://github.com/muellerberndt/cadence">muellerberndt/cadence</a>.</p>
  <p>Code: <a href="https://github.com/sean-mcclane832/cadence-arcade/tree/master/mario">cadence-arcade/mario</a> (levels.py, game.py)</p>
</div></footer>
<!--/site-footer-->
</body>
</html>
"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {out / 'index.html'} ({len(names)} levels, {n_vids} videos)")


if __name__ == "__main__":
    main()
