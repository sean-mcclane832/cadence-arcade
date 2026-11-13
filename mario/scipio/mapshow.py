"""Death and stall map from scratch/play/play_<tag>.json (no game played here)."""
import json, sys, numpy as np
for tag in sys.argv[1:]:
    eps = json.load(open(f"scratch/play/play_{tag}.json"))["episodes"]
    nf = [e for e in eps if e["end"] != "flag"]
    def tail_same(e):  # how many final decisions used one button
        a = e["actions"]; k = 1
        while k < len(a) and a[-k - 1] == a[-1]: k += 1
        return k
    died = [e for e in nf if e["end"] == "died"]; stuck = [e for e in nf if e["end"] == "stuck"]
    lastbtn = lambda es: {b: sum(e["actions"][-1] == b for e in es) for b in (0, 3, 4, 6)}
    never = [e for e in stuck if e["max_x"] <= 60]
    print(f"== {tag}: died {len(died)} stuck {len(stuck)} flag {len(eps)-len(nf)}")
    print("  died: median x", int(np.median([e["max_x"] for e in died])), "median decisions", int(np.median([e["decisions"] for e in died])),
          "last button", lastbtn(died), "same button for the last >=8 decisions:", round(float(np.mean([tail_same(e) >= 8 for e in died])), 3),
          "median run of the last button", int(np.median([tail_same(e) for e in died])))
    print("  stuck: never left the start", len(never), "| last button", lastbtn(stuck), "| one button for the last >=100 decisions:", sum(tail_same(e) >= 100 for e in stuck),
          "| distinct buttons in last 100:", round(float(np.mean([len(set(e['actions'][-100:])) for e in stuck])), 2))
    # share of all decisions equal to the previous one (own play) 
    rep = np.mean([np.mean(np.array(e["actions"][1:]) == np.array(e["actions"][:-1])) for e in eps if len(e["actions"]) > 1])
    print("  own play: decision repeats the previous button", round(float(rep), 3), "(coach runs: 0.772)")
    # spots: deaths clustered within 48 px per level
    spots = []
    for L in sorted(set(e["level"] for e in eps)):
        xs = sorted(e["max_x"] for e in nf if e["level"] == L)
        if not xs: continue
        best = max(((sum(abs(x - c) <= 24 for x in xs), c) for c in xs))
        spots.append((best[0], L, best[1], len(xs)))
    spots.sort(reverse=True)
    print("  levels where >=6 of 8 attempts end within 48 px of one spot:", sum(s[0] >= 6 for s in spots), "of", len(spots), "| top:", [(L, x, f"{n}/{m}") for n, L, x, m in spots[:8]])
    print("  attempts ending at one of those single spots:", sum(s[0] for s in spots if s[0] >= 6), "of", len(nf))
