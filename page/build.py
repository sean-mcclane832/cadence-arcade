"""Build the replay page from a run folder.

    python page/build.py runs/main page/ember-field.html

Picks the showcase seed as the median Cadence life by final-season net score, embeds its
clips (and the same seed's MLP clips, which see the same embers), the median and quartile
curves of all lives, and a season table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEASON = 30  # windows of 500 moments per season


def load(run: Path):
    lives = {}
    for f in sorted(run.glob("*-*.json")):
        r = json.loads(f.read_text())
        lives.setdefault(r["kind"], {})[r["seed"]] = r
    return json.loads((run / "meta.json").read_text()), lives


def net(r):
    return np.array(r["food"]) - np.array(r["sting"])


def seasons(r):
    n = net(r)
    return [int(n[i * SEASON:(i + 1) * SEASON].sum()) for i in range(3)]


def main(run, out):
    run, out = Path(run), Path(out)
    meta, lives = load(run)
    kinds = ["cadence", "mlp", "random"]
    curve = {}
    for k in kinds:
        m = np.array([net(r) for r in lives[k].values()])
        curve[k] = {"med": np.median(m, 0).round(1).tolist(), "q1": np.percentile(m, 25, 0).round(1).tolist(),
                    "q3": np.percentile(m, 75, 0).round(1).tolist()}
    # showcase: median cadence seed by final-season net
    cad = lives["cadence"]
    order = sorted(cad, key=lambda s: seasons(cad[s])[2])
    seed = order[len(order) // 2]
    show = {"cadence": cad[seed]["clips"], "mlp": lives["mlp"][seed]["clips"]}

    phases = [("First 2,000", 0, 4), ("End of season 1", 24, 30), ("2,000 after turn 1", 30, 34),
              ("End of season 2", 54, 60), ("End of season 3", 84, 90)]
    stats = {}
    for k in kinds:
        F = np.array([r["food"] for r in lives[k].values()]); S = np.array([r["sting"] for r in lives[k].values()])
        stats[k] = [(float(np.median(F[:, a:b].mean(1))), float(np.median(S[:, a:b].mean(1)))) for _, a, b in phases]
    head = "".join(f"<th>{p[0]}</th>" for p in phases)
    rows = "".join(f"<tr><td>{label}</td>" + "".join(f"<td>{f:.1f} / {s:.1f}</td>" for f, s in stats[k]) + "</tr>"
                   for k, label in zip(kinds, ["Cadence brain", "Plain MLP", "Random"]))
    table = f"<thead><tr><th>Player</th>{head}</tr></thead><tbody>{rows}</tbody>"
    c, m, r0 = stats["cadence"], stats["mlp"], stats["random"]
    data = {
        "show": show, "seed": seed, "window": meta["window"], "turns": meta["turns"], "curve": curve, "table": table,
        "lede": (f"A lantern creature has to work out which falling embers feed it and which sting. Then the colours swap, "
                 f"twice, without warning. A <b>Cadence</b> brain learns it from pay alone, in one continuous life: by the "
                 f"end of season one it catches {c[1][0] / r0[1][0]:.1f}&times; the food of random play and takes "
                 f"{(1 - c[1][1] / r0[1][1]) * 100:.0f}% fewer stings, and after each turn it relearns which colour to dodge. "
                 f"A plain neural net plays the same embers next to it, and it's the stronger player here."),
        "story": (f"Watch the first turn. The plain net keeps heading for the colour that used to feed it, which now stings: "
                  f"in the 2,000 moments after the turn it takes {m[2][1]:.1f} stings per {meta['window']} moments, about "
                  f"twice random play. The Cadence brain's stings rise to {c[2][1]:.1f}, about what random play takes. By the "
                  f"end of season two both dodge about as well as they did before the turn. The plain net's food catching comes back to {m[3][0] / m[1][0] * 100:.0f}% of its "
                  f"season-one level and the Cadence brain's to {c[3][0] / c[1][0] * 100:.0f}%. When amber feeds again in "
                  f"season three, the plain net's old skill returns ({m[4][0]:.1f} food) and the Cadence brain stays "
                  f"where season two left it ({c[4][0]:.1f}). Showcase life: seed {seed}, the median Cadence life by "
                  f"season-three score; the plain net on the right is the same seed, so it sees the same embers."),
        "curvecap": (f"Net score (food minus stings) per {meta['window']} moments, median and middle half of 12 lives "
                     f"per player (seeds {meta['seeds'][0]}-{meta['seeds'][1]}). Random play sits near zero."),
        "tblcap": (f"Food / stings per {meta['window']} moments, median of 12 lives. Random play catches about "
                   f"{r0[1][0]:.0f} and {r0[1][1]:.0f}."),
        "foot": (f"Cadence {meta['cadence']} (cadence-net), Python {meta['python']}, CPU. "
                 f"Cadence settings: {json.dumps(meta['cadence_settings'])}. MLP: {meta['mlp']}. "
                 f"Game, players and recordings: P:/projects/cadence-arcade. Built 2026-10-28."),
    }
    html = Path(__file__).with_name("template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out.write_text(html, encoding="utf-8", newline="\n")
    summary = {"seed": seed, "phases": [p[0] for p in phases], "food_sting": stats,
               "seasons": {k: {s: seasons(r) for s, r in lives[k].items()} for k in kinds}}
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: stats[k] for k in kinds}), "seed", seed)
    print("page bytes", out.stat().st_size)


if __name__ == "__main__":
    main(*sys.argv[1:3])
