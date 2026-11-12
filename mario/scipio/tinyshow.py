import json, glob, sys
rows = []
for f in sorted(glob.glob("scratch/tiny/*.json")):
    d = json.load(open(f))
    if "curve" in d:
        c = [r for r in d["curve"] if r["lesson"] >= d["lessons"] - 100 and r["lesson"] % 50 == 0][-3:]
        m = lambda k: sum(r["held"][k] for r in c) / len(c)
        print(f"{f.split('tiny')[-1][1:-5]:16s} lessons {d['lessons']:5d} last3: all {m('all'):.3f} repeat {m('repeat'):.3f} change {m('change'):.3f} held-button {m('answers_held_button'):.3f} | final change {d['curve'][-1]['held']['change']:.3f} s {d['curve'][-1]['s']}")
    else:
        h = d["held"]; print(f"{f.split('tiny')[-1][1:-5]:16s} epochs {d['epochs']} all {h['all']:.3f} repeat {h['repeat']:.3f} change {h['change']:.3f} held-button {h['answers_held_button']:.3f} | seen change {d['seen']['change']:.3f} s {d['s']}")
