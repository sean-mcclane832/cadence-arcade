"""Builds stuck/index.html + stuck/ends.json from scratch/stuck/ends_<run>.jsonl (stuckmap.py page)."""
from __future__ import annotations
import html, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCR = HERE / "scratch" / "stuck"; PAGE = HERE / "stuck"
RUNS = ["ScipioA", "ScipioB", "ScipioFast", "Twin"]
MEMO = {"ScipioA": "slow memory 0.97", "ScipioB": "slow memory 0.97", "ScipioFast": "memory 0.8", "Twin": "no memory"}
LEVELS = [f"{w}-{s}" for w in range(1, 7) for s in range(1, 5)]
B = 32
E = html.escape


def spot_key(e):
    x = e["x"] if e.get("x") is not None else e["max_x"]
    return (e["level"], x // B * B)


def copy(rel):
    if not rel:
        return None
    dst = PAGE / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(SCR / rel, dst)
    return rel


def build():
    man = json.loads((SCR / "manifest.json").read_text())
    ver = json.loads((SCR / "verify.json").read_text())
    notes = json.loads((SCR / "notes.json").read_text()) if (SCR / "notes.json").exists() else {}
    ends = {r: [json.loads(l) for l in open(SCR / f"ends_{r}.jsonl")] for r in RUNS if (SCR / f"ends_{r}.jsonl").exists()}
    runs = [r for r in RUNS if r in ends]
    if PAGE.exists():
        shutil.rmtree(PAGE)
    PAGE.mkdir()
    allends = [dict(e, verified=ver[e["run"]]["ok"]) for r in runs for e in ends[r]]
    (PAGE / "ends.json").write_text(json.dumps(dict(
        note="SEL8 replay of each run's latest scored checkpoint (stuckmap.py). x = where Mario stood at the end, max_x = furthest x. "
             "kind: flag / died / midstuck (225 decisions without a new furthest x, after passing x 60) / nostart (the same, never past x 60).",
        manifest={r: dict(chunk=man[r]["chunk"], ck=Path(man[r]["ck"]).name) for r in runs}, verify=ver, ends=allends), indent=0))

    stuck = [e for e in allends if e["end"] == "stuck"]
    spots = defaultdict(list)
    for e in stuck:
        spots[spot_key(e)].append(e)

    def spot_card(key, es, big=False):
        L, xb = key
        kinds = Counter(e["kind"] for e in es)
        by_run = Counter(e["run"] for e in es)
        held = Counter()
        for e in es:
            held.update(e["held100"])
        tot = sum(held.values())
        heldtxt = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in held.most_common(3))
        coach = Counter(e["coach"] for e in es if e.get("coach"))
        rep = sorted(es, key=lambda e: (RUNS.index(e["run"]), e["start"]))[0]
        img = copy(rep.get("png")); clip = copy(rep.get("clip")) if big else None
        note = notes.get(f"{L}@{xb}", "")
        kindtxt = " + ".join(f"{v} {'never left start' if k == 'nostart' else 'mid-level'}" for k, v in kinds.items())
        ys = sorted({e["y"] for e in es if e.get("y") is not None})
        h = [f'<div class="card"><div class="ch"><b>{L}</b> x {xb}-{xb + B - 1} <span class="n">{len(es)} stuck</span></div>']
        if clip:
            h.append(f'<video src="{clip}" poster="{img}" controls muted loop playsinline preload="none"></video>')
        elif img:
            h.append(f'<img src="{img}" loading="lazy" alt="{L} x {xb}">')
        h.append(f'<div class="meta">{E(kindtxt)}<br>runs: {", ".join(f"{r} {by_run[r]}" for r in RUNS if by_run[r])}'
                 f'<br>held (last 100 decisions): {E(heldtxt)}'
                 + (f'<br>coach at the spot: {", ".join(f"{k} {v}" for k, v in coach.most_common())}' if coach else "")
                 + (f'<br>y {ys[0]}' + (f'-{ys[-1]}' if len(ys) > 1 else "") if ys else "")
                 + (f'<br><i>{E(note)}</i>' if note else "") + '</div></div>')
        return "".join(h)

    # ---- summary table
    rows = []
    for r in runs:
        es = ends[r]; v = ver[r]
        c = Counter(e["kind"] for e in es)
        mean = np.mean([np.mean([e["max_x"] for e in es if e["level"] == L]) for L in LEVELS])
        rows.append(f"<tr><td><b>{r}</b><br><small>{MEMO[r]}</small></td><td>{man[r]['chunk']}</td><td>{mean:.0f}</td>"
                    f"<td>{c['died']}</td><td><b>{c['midstuck']}</b></td><td>{c['nostart']}</td><td>{c['flag']}</td>"
                    f"<td>{'yes' if v['ok'] else 'NO'}</td></tr>")
    top = sorted(spots.items(), key=lambda kv: (-len(kv[1]), LEVELS.index(kv[0][0]), kv[0][1]))
    mid_spots = [(k, es) for k, es in top if any(e["kind"] == "midstuck" for e in es)]

    # ---- level x run grid
    grid = ["<tr><th>level</th>" + "".join(f"<th>{r}</th>" for r in runs) + "</tr>"]
    for L in LEVELS:
        cells = []
        for r in runs:
            es = [e for e in ends[r] if e["level"] == L]
            m = sum(e["kind"] == "midstuck" for e in es); n = sum(e["kind"] == "nostart" for e in es)
            d = sum(e["end"] == "died" for e in es); f = sum(e["end"] == "flag" for e in es)
            t = []
            if m: t.append(f'<span class="mid">{m} stuck</span>')
            if n: t.append(f'<span class="ns">{n} start</span>')
            if d: t.append(f'{d} died')
            if f: t.append(f'<span class="fl">{f} flag</span>')
            cells.append("<td>" + "<br>".join(t) + "</td>")
        grid.append(f'<tr><td><a href="#L{L}">{L}</a></td>' + "".join(cells) + "</tr>")

    # ---- per level
    lv = []
    for L in LEVELS:
        les = [e for e in allends if e["level"] == L]
        ls = [(k, es) for k, es in top if k[0] == L]
        deaths = [e for e in les if e["end"] == "died"]
        dsp = Counter(e["max_x"] // B * B for e in deaths)
        h = [f'<section id="L{L}"><h3>{L}</h3><p class="meta">']
        h.append(" | ".join(f"{r}: " + ", ".join(f"{k} {v}" for k, v in Counter(e['kind'] for e in les if e['run'] == r).most_common()) for r in runs))
        h.append("</p>")
        if dsp:
            dtop = dsp.most_common(3)
            h.append(f'<p class="meta">deaths ({len(deaths)}), most common furthest x: ' + ", ".join(f"{x}-{x + B - 1} ({n})" for x, n in dtop) + "</p>")
            x0 = dtop[0][0]; rep = next(e for e in deaths if e["max_x"] // B * B == x0)
            img = copy(rep.get("png"))
            h.append(f'<details><summary>last screen at the most common death spot ({rep["run"]}, start {rep["start"]})</summary>'
                     f'<img src="{img}" loading="lazy" alt="death {L}"></details>')
        if ls:
            h.append('<div class="cards">' + "".join(spot_card(k, es, big=True) for k, es in ls) + "</div>")
        h.append("</section>")
        lv.append("".join(h))

    vtxt = "; ".join(f"{r} chunk {ver[r]['chunk']} {'matches' if ver[r]['ok'] else 'DOES NOT match (spots below are not the run’s own)'}" for r in runs)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scipio Stuck Spots</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e3e1db;--mid:#b4441c;--ns:#6d4fb3;--fl:#2f7d32}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151514;--fg:#ecebe6;--mut:#a3a19b;--card:#1f1f1d;--line:#34332f;--mid:#f08a5d;--ns:#b9a3f0;--fl:#7fcf83}}}}
:root[data-theme="dark"]{{--bg:#151514;--fg:#ecebe6;--mut:#a3a19b;--card:#1f1f1d;--line:#34332f;--mid:#f08a5d;--ns:#b9a3f0;--fl:#7fcf83}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:980px;padding:16px}}
h1{{font-size:1.4em;margin:.2em 0}} h2{{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid var(--line)}} h3{{margin:1.2em 0 .3em}}
table{{border-collapse:collapse;width:100%;font-size:.9em}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}}
.wrap{{overflow-x:auto}} .meta{{color:var(--mut);font-size:.88em}} .mid{{color:var(--mid);font-weight:600}} .ns{{color:var(--ns)}} .fl{{color:var(--fl)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}} .ch{{margin-bottom:4px}} .n{{float:right;color:var(--mid);font-weight:600}}
img,video{{width:100%;image-rendering:pixelated;border-radius:4px;display:block}} details img{{max-width:320px}}
</style></head><body>
<h1>Where the Scipio brains get stuck</h1>
<p class="meta">SEL8 (8 starts x 24 levels 1-1..6-4, greedy) replayed on each run's latest scored checkpoint, the way the trainer scores it.
Stuck = 225 decisions (15 s) without a new furthest x. "Never left start" = stuck without passing x 60. Spots are 32 px buckets of the x where Mario stood.
Built {__import__('time').strftime('%Y-%m-%d %H:%M')}. Replay check: {E(vtxt)}.
ScipioA is at a later chunk than the others, so counts are not same-age. No verdicts here: the registered read is at chunk 50.</p>
<h2>Counts per run</h2><div class="wrap"><table><tr><th>run</th><th>chunk</th><th>mean x</th><th>died</th><th>stuck mid-level</th><th>never left start</th><th>flags</th><th>replay matches</th></tr>
{''.join(rows)}</table></div>
<h2>Top mid-level stuck spots</h2><div class="cards">{''.join(spot_card(k, es, big=True) for k, es in mid_spots[:10])}</div>
<h2>Never left the start</h2><div class="cards">{''.join(spot_card(k, es) for k, es in top if all(e['kind'] == 'nostart' for e in es))}</div>
<h2>Level by run</h2><div class="wrap"><table>{''.join(grid)}</table></div>
<h2>Per level</h2>{''.join(lv)}
</body></html>"""
    (PAGE / "index.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in PAGE.rglob("*") if p.is_file())
    print("page", PAGE / "index.html", f"{size / 1e6:.1f} MB", len(stuck), "stuck,", len(spots), "spots")
