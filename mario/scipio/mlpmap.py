import json, numpy as np
for t in ("plain_0", "plain_1", "change2_0", "change2_1", "aonly_0", "aonly_1"):
    d = json.load(open(f"scratch/play/mlp_{t}.json")); eps = [e for e in d["episodes"] if e["level"][0] != "7"]
    st = [e for e in eps if e["end"] == "stuck"]
    rep = np.mean([np.mean(np.array(e["actions"][1:]) == np.array(e["actions"][:-1])) for e in eps if len(e["actions"]) > 1])
    print(t, d["summary"]["mean"], d["summary"]["flags"], d["summary"]["ends"], "never left start", sum(e["max_x"] <= 60 for e in st), "one button last 100:", sum(len(set(e["actions"][-100:])) == 1 for e in st), "repeat share", round(float(rep), 3), "w7", (d.get("w7") or {}).get("mean"), (d.get("w7") or {}).get("per_level"))
