import json, glob
for f in sorted(glob.glob("scratch/tiny/*.json")):
    d = json.load(open(f)); n = f.replace("\\", "/").split("/")[-1][:-5]
    if "curve" in d:
        c = [r for r in d["curve"] if r["lesson"] > 3][-3:]; m = lambda k: sum(r["held"][k] for r in c) / len(c)
        s = d["curve"][-1].get("seen", {})
        print(f"{n:20s} lessons {d['curve'][-1]['lesson']:5d} (last {len(c)} points) all {m('all'):.3f} repeat {m('repeat'):.3f} change {m('change'):.3f} heldbtn {m('answers_held_button'):.3f} seen r/c {s.get('repeat')}/{s.get('change')}")
    else:
        h, s = d["held"], d["seen"]; print(f"{n:20s} epochs {d['epochs']:4.0f} all {h['all']:.3f} repeat {h['repeat']:.3f} change {h['change']:.3f} heldbtn {h['answers_held_button']:.3f} seen r/c {s['repeat']}/{s['change']}")
