"""Summarize the 10x-long Breakout run and build its standalone page.

    ../.venv-classic/Scripts/python.exe long_page.py runs/long10x ../docs/breakout-long/index.html

Same protocol as runs/main (see run.py), only --blocks 300 instead of 30, arms pos32, pos0,
norecall, mlp1, mlp2 (random left out: it does not learn). Curves are the median over 12 seeds of
the mean test score in each 10-block window. Paired tests use the primary metric (last 10 blocks).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ARMS = [
    ("pos32", "Cadence, recall + position term", "#ffb347"),
    ("pos0", "Cadence, recall, no position term", "#c9a0ff"),
    ("norecall", "Cadence, no recall memory", "#5fd3a5"),
    ("mlp1", "Plain MLP, one frame", "#8a8f98"),
    ("mlp2", "Plain MLP, given two frames", "#5fb4ff"),
]
WIN = 10
PAIRS = [("pos32", "pos0"), ("norecall", "pos32"), ("pos32", "mlp1"), ("norecall", "mlp1"),
         ("mlp2", "pos32"), ("mlp2", "norecall")]
# 30-block results from runs/main (classic/README.md), for the before/after column
SHORT = {"pos32": 5.65, "pos0": 5.57, "norecall": 6.97, "mlp1": 5.10, "mlp2": 6.72}


def load(run: Path):
    out = {}
    for key, *_ in ARMS:
        lives = [json.loads(p.read_text()) for p in sorted(run.glob(f"{key}-*.json"))]
        scores = np.array([[np.mean(b) for b in L["score"]] for L in lives])  # seeds x blocks
        out[key] = {
            "seeds": [L["seed"] for L in lives],
            "scores": scores,
            "primary": np.array([L["primary"] for L in lives]),
            "agree": np.array([np.mean(L["agree"][-10:]) for L in lives]),
            "minutes": float(np.mean([L["seconds"] for L in lives]) / 60),
        }
    return out


def summarize(D):
    rows = {}
    for key, *_ in ARMS:
        d = D[key]
        nb = d["scores"].shape[1]
        win = d["scores"].reshape(len(d["seeds"]), nb // WIN, WIN).mean(axis=2)
        rows[key] = {
            "primary_median": float(np.median(d["primary"])),
            "primary_iqr": [float(x) for x in np.percentile(d["primary"], [25, 75])],
            "agree_median": float(np.median(d["agree"])),
            "curve_median": [float(x) for x in np.median(win, axis=0)],
            "curve_q25": [float(x) for x in np.percentile(win, 25, axis=0)],
            "curve_q75": [float(x) for x in np.percentile(win, 75, axis=0)],
            "minutes_per_life": d["minutes"],
            "short_run_primary": SHORT[key],
        }
    tests = []
    for a, b in PAIRS:
        pa = dict(zip(D[a]["seeds"], D[a]["primary"]))
        pb = dict(zip(D[b]["seeds"], D[b]["primary"]))
        diff = np.array([pa[s] - pb[s] for s in sorted(pa)])
        tests.append({"a": a, "b": b, "wins": int((diff > 0).sum()), "n": len(diff),
                      "median_diff": float(np.median(diff)), "p": float(wilcoxon(diff).pvalue)})
    return {"blocks": int(D["pos32"]["scores"].shape[1]), "window": WIN, "arms": rows, "tests": tests}


def svg_chart(S):
    W, H, L, R, T, B = 640, 340, 44, 12, 12, 36
    n = S["blocks"] // WIN
    ymax = max(max(r["curve_q75"]) for r in S["arms"].values())
    ymax = float(np.ceil(ymax / 5) * 5)
    x = lambda i: L + (W - L - R) * (i + 0.5) / n
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Bricks per test game over 300 training blocks">']
    for v in np.arange(0, ymax + 0.1, 5):
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{L-8}" y="{y(v)+4:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for blk in range(0, S["blocks"] + 1, 50):
        xx = L + (W - L - R) * blk / S["blocks"]
        parts.append(f'<text x="{xx:.1f}" y="{H-14}" class="tick" text-anchor="middle">{blk}</text>')
    parts.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-1}" class="tick" text-anchor="middle">training block (1,000 moments each)</text>')
    xs = L + (W - L - R) * 30 / S["blocks"]
    parts.append(f'<line x1="{xs:.1f}" x2="{xs:.1f}" y1="{T}" y2="{H-B}" class="mark"/>'
                 f'<text x="{xs+4:.1f}" y="{T+12}" class="tick">first run ended here</text>')
    for key, _, color in ARMS:
        r = S["arms"][key]
        if key in ("pos32", "mlp2"):
            up = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(r["curve_q75"]))
            dn = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(r["curve_q25"]))))
            parts.append(f'<polygon points="{up} {dn}" fill="{color}" opacity="0.13"/>')
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(r["curve_median"]))
        dash = ' stroke-dasharray="6 4"' if key == "mlp2" else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
    parts.append("</svg>")
    return "\n".join(parts)


def page(S):
    A = S["arms"]
    label = {k: n for k, n, _ in ARMS}
    color = {k: c for k, _, c in ARMS}
    legend = "".join(f'<span><i style="background:{color[k]}"></i>{label[k]}</span>' for k, *_ in ARMS)
    trows = "".join(
        f'<tr><td><i style="background:{color[k]}"></i>{label[k]}</td><td>{A[k]["short_run_primary"]:.1f}</td>'
        f'<td><b>{A[k]["primary_median"]:.1f}</b> <span class="dim">[{A[k]["primary_iqr"][0]:.1f}-{A[k]["primary_iqr"][1]:.1f}]</span></td>'
        f'<td>{A[k]["agree_median"]*100:.0f}%</td></tr>' for k, *_ in ARMS)
    srows = "".join(
        f'<tr><td>{label[t["a"]]} vs {label[t["b"]]}</td><td>{t["wins"]}/{t["n"]}</td>'
        f'<td>{t["median_diff"]:+.1f}</td><td>{t["p"]:.4f}</td></tr>' for t in S["tests"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breakout, Long Run</title>
<meta name="description" content="One-Frame Breakout trained 10x longer: does a Cadence brain keep improving, and does the 0.89.0 recall position term start to matter?">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Silkscreen&display=swap">
<link rel="stylesheet" href="../assets/site.css">
<style>
  /* page tokens mapped onto the shared site tokens (assets/site.css) */
  :root {{ --dim: var(--c-dim); --line: var(--c-line); }}
  svg {{ width: 100%; height: auto; display: block; }}
  svg .grid {{ stroke: var(--line); }} svg .mark {{ stroke: var(--dim); stroke-dasharray: 3 4; }}
  svg .tick {{ fill: var(--dim); font: 11px var(--c-mono); }}
  .chartbox {{ background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--c-radius); padding: 8px; overflow-x: auto; }}
  .chartbox svg {{ min-width: 480px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--dim); margin: 10px 0 0; }}
  i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
  .site td {{ white-space: nowrap; }}
</style>
</head>
<body class="site">
<!--site-header-->
<header class="site-header"><div class="site-header-in">
  <a class="site-logo" href="../">CADENCE<span>ARCADE</span></a>
  <nav class="site-nav" aria-label="Site"><a href="../mario/">Mario</a><a href="../brain/">Brain scope</a><a href="../chat/">Chat</a><a href="../ember-field/">Ember Field</a><a href="../breakout/" aria-current="page">Breakout</a></nav>
</div></header>
<!--/site-header-->
<main class="page">
  <p class="crumbs"><a href="../">Home</a> / <a href="../breakout/">One-Frame Breakout</a> / 10x longer</p>
  <p class="eyebrow">MinAtar Breakout &middot; follow-up</p>
  <h1>Breakout, 10x longer</h1>
  <p class="lede">Same game, players and protocol as <a href="../breakout/">One-Frame Breakout</a>, but each life trains for 300 blocks (300,000 moments) instead of 30. 12 seeds per player. cadence-net 0.89.0.</p>

  <p>The first run stopped while every curve was still climbing. So: what happens if you just keep going?</p>

  <h2>Bricks per test game</h2>
  <div class="chartbox">{svg_chart(S)}</div>
  <div class="legend">{legend}</div>
  <p class="small dim">Median of 12 seeds, each point a 10-block window. Shaded: middle half of seeds for the position-term brain and the two-frame MLP.</p>

  <h2>What it showed</h2>
  <ul>
    <li><b>Cadence keeps learning.</b> All three Cadence brains more than doubled after block 30 and were still rising at block 300.</li>
    <li><b>One frame is a wall for a plain learner, not for Cadence.</b> The one-frame MLP flattens near 6 bricks: it can't tell which way the ball is moving. Every Cadence brain beats it on 12 of 12 seeds, by about 7 bricks. A <a href=\"https://github.com/sean-mcclane832/cadence-lab/blob/master/notes/settle_probe.md\">follow-up probe</a> found why: Cadence wires in a working-memory copy of the last frame by default, so it has the ball's direction built in, not learned.</li>
    <li><b>Handing the MLP the second frame wins in the long run.</b> At 30 blocks it was level with Cadence. By 300 it leads every Cadence brain (12/12 vs the position-term brain, 11/12 vs no-recall), about 6 bricks ahead. The probe says Cadence has nearly the same information (direction readable at 98% vs 99%), so the gap is in how it learns to use it.</li>
    <li><b>The 0.89.0 position term: null, confirmed.</b> On these 12 seeds it looked like +3 bricks (7/12, p = 0.09), so we ran 24 more seeds. Pooled over 36 seeds it wins 17, median -1.0 bricks, p = 0.68. The lean was noise. Recall memory with or without it doesn't beat having no recall at all.</li>
  </ul>

  <h2>Numbers</h2>
  <div class="tw"><table>
    <tr><th>player</th><th>30 blocks</th><th>300 blocks [IQR]</th><th>agrees w/ coach</th></tr>
    {trows}
  </table></div>
  <p class="small dim">Bricks per test game over the last 10 blocks, median over seeds. The 30-block column is the first run (separate lives, same seeds).</p>

  <div class="tw"><table>
    <tr><th>paired, by seed</th><th>wins</th><th>median diff</th><th>Wilcoxon p</th></tr>
    {srows}
  </table></div>

  <h2>How it ran</h2>
  <p class="small dim">One command on a desktop CPU, about 30 minutes per Cadence life and 3 per MLP life, 6 at a time: <code>run.py --seeds 10-21 --arms pos32,pos0,norecall,mlp1,mlp2 --blocks 300 --procs 6 --out runs/long10x</code>. Nothing else changed from the pre-declared protocol in <a href="https://github.com/sean-mcclane832/cadence-arcade/blob/master/classic/run.py">run.py</a>. Raw lives in <a href="https://github.com/sean-mcclane832/cadence-arcade/tree/master/classic/runs/long10x">runs/long10x</a>.</p>

</main>
<!--site-footer-->
<footer class="site-footer"><div class="site-footer-in">
  <p>Independent experiments by <a href="https://github.com/sean-mcclane832">sean-mcclane832</a>. Not affiliated with Pragma Research. Cadence: <a href="https://github.com/muellerberndt/cadence">muellerberndt/cadence</a>. Game: MinAtar Breakout (Young &amp; Tian 2019).</p>
  <p>Code: <a href="https://github.com/sean-mcclane832/cadence-arcade/tree/master/classic">cadence-arcade/classic</a></p>
</div></footer>
<!--/site-footer-->
</body>
</html>
"""


def main(run, out):
    run, out = Path(run), Path(out)
    S = summarize(load(run))
    (run / "summary.json").write_text(json.dumps(S, indent=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(S), encoding="utf-8", newline="\n")
    for t in S["tests"]:
        print(f'{t["a"]} vs {t["b"]}: {t["wins"]}/{t["n"]} {t["median_diff"]:+.2f} p={t["p"]:.4f}')


if __name__ == "__main__":
    main(*sys.argv[1:3])
