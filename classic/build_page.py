"""Build the single-file replay page from runs/ and runs/summary.json.

    ../.venv-classic/Scripts/python.exe summarize.py
    ../.venv-classic/Scripts/python.exe build_page.py page/one-frame-breakout.html

Showcase seed: the median seed of the position-term arm by the primary score (a typical seed,
not the best). All six boards replay the same seed's recorded test games, which are the same
games for every player.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARMS = [
    {"key": "pos32", "label": "Cadence, recall + position term", "color": "--pos", "band": True},
    {"key": "pos0", "label": "Cadence, recall, no position term", "color": "--nopos", "band": True},
    {"key": "norecall", "label": "Cadence, no recall memory", "color": "--norec"},
    {"key": "mlp1", "label": "Plain MLP, one frame", "color": "--mlp1"},
    {"key": "mlp2", "label": "Plain MLP, given two frames", "color": "--mlp2", "dash": True},
    {"key": "random", "label": "Random", "color": "--rnd"},
]


def f(x, d=1):
    return f"{x:.{d}f}"


def main(out):
    S = json.loads(Path("runs/summary.json").read_text())
    meta, T = S["meta"], S["table"]
    lives = {}
    for p in Path("runs/main").glob("*-*.json"):
        r = json.loads(p.read_text())
        lives.setdefault(r["arm"], {})[r["seed"]] = r
    pos = lives["pos32"]
    order = sorted(pos, key=lambda s: pos[s]["primary"])
    seed = order[len(order) // 2]
    blocks = meta["blocks"]
    clip_keys = sorted(lives["pos32"][seed]["clips"], key=int)
    clips = {a["key"]: {k: lives[a["key"]][seed]["clips"][k] for k in clip_keys} for a in ARMS}
    stage_names = {k: f"after {int(k) + 1}k training moments" for k in clip_keys}

    pp, rp = S["pos32_vs_pos0"], S.get("recency0", {}).get("pos32_vs_pos0")
    R = S.get("recency0", {}).get("table", {})
    sc = {k: T[k]["score"]["median"] for k in T}

    rows = ""
    for a in ARMS:
        t = T[a["key"]]
        rows += (f"<tr><td><i style=\"background:var({a['color']})\"></i>{a['label']}</td>"
                 f"<td>{f(t['score']['median'])}</td><td>{f(t['score']['q1'])} – {f(t['score']['q3'])}</td>"
                 f"<td>{f(100 * t['agree']['median'], 0)}%</td><td>{f(t['length']['median'], 0)}</td></tr>")
    for k, label in (("pos32", "Recency off: recall + position term"), ("pos0", "Recency off: recall, no position term")):
        if k in R:
            t = R[k]
            rows += (f"<tr><td><i style=\"background:var({'--pos' if k == 'pos32' else '--nopos'});opacity:.5\"></i>{label}</td>"
                     f"<td>{f(t['score']['median'])}</td><td>{f(t['score']['q1'])} – {f(t['score']['q3'])}</td>"
                     f"<td>{f(100 * t['agree']['median'], 0)}%</td><td>{f(t['length']['median'], 0)}</td></tr>")
    table = ("<thead><tr><th>Player</th><th>Bricks / game</th><th>Middle half</th><th>Agrees w/ coach</th>"
             "<th>Moments kept alive</th></tr></thead><tbody>" + rows + "</tbody>")

    def card(tag, p, a, b):
        return (f"<div class=\"vcard\"><span class=\"tag\">{tag}</span>"
                f"<span class=\"big\">{p['wins']} of {len(p['seeds'])} seeds</span>"
                f"<p>The position term scored more bricks than the same brain without it in {p['wins']} seeds and fewer in "
                f"{p['losses']}. Median difference {'+' if p['median_diff'] >= 0 else ''}{f(p['median_diff'], 2)} bricks per game "
                f"({f(a)} vs {f(b)}), Wilcoxon signed-rank p = {f(p['wilcoxon_p'], 3)}.</p></div>")

    vcards = card("Founder memory (as shipped)", pp, sc["pos32"], sc["pos0"])
    pairs = [{"label": "Founder memory", "diff": pp["diff"], "median": pp["median_diff"]}]
    if rp:
        vcards += card("Recency nudge turned off", rp, R["pos32"]["score"]["median"], R["pos0"]["score"]["median"])
        pairs.append({"label": "Recency off", "diff": rp["diff"], "median": rp["median_diff"]})

    text = json.loads(Path("page_text.json").read_text()) if Path("page_text.json").exists() else {}
    fills = dict(seed=seed, pos32=f(sc["pos32"]), pos0=f(sc["pos0"]), norecall=f(sc["norecall"]), mlp1=f(sc["mlp1"]),
                 mlp2=f(sc["mlp2"]), random=f(sc["random"]), wins=pp["wins"], losses=pp["losses"],
                 p=f(pp["wilcoxon_p"], 3), md=f(pp["median_diff"], 2),
                 rwins=rp["wins"] if rp else "", rlosses=rp["losses"] if rp else "", rp=f(rp["wilcoxon_p"], 3) if rp else "",
                 rmd=f(rp["median_diff"], 2) if rp else "",
                 r32=f(R["pos32"]["score"]["median"]) if R else "", r0=f(R["pos0"]["score"]["median"]) if R else "",
                 last=meta["last"], blocks=blocks, version=meta["cadence"])
    text = {k: (v.format(**fills) if isinstance(v, str) else [x.format(**fills) for x in v]) for k, v in text.items()}
    text.setdefault("foot", "")
    text["foot"] += (f"<span>Cadence {meta['cadence']} (cadence-net from PyPI), MinAtar 1.0.15, Python {meta['python']}, CPU. "
                     f"Cadence settings: <code>{json.dumps(meta['cadence_settings'])}</code>. MLP: {meta['mlp']}. "
                     f"Seeds {meta['seeds'][0]}–{meta['seeds'][1]}. Code and recordings: github.com/sean-mcclane832/cadence-arcade (classic/). "
                     f"Built 2026-10-28.</span>")
    data = {"arms": ARMS, "clips": clips, "stageNames": stage_names, "curves": S["curves"], "table": table,
            "vcards": vcards, "pairs": pairs, "text": text, "lastFrom": blocks - meta["last"]}
    html = Path("page_template.html").read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8", newline="\n")
    print("showcase seed", seed, "page bytes", Path(out).stat().st_size)


if __name__ == "__main__":
    main(sys.argv[1])
