"""Leg 2c run A: old A-held-only MLP play (mask set by a literal token, not recorded) against the replay with --mode A.
    python cmpA.py <seed>   compares scratch/play/mlp_aonly_<seed>.json with mlp_aonly_<seed>_modeA.json episode by episode."""
import json, sys
s = sys.argv[1]
old = json.load(open(f"scratch/play/mlp_aonly_{s}.json")); new = json.load(open(f"scratch/play/mlp_aonly_{s}_modeA.json"))
key = lambda e: (e["level"], e["start"])
o = {key(e): e for e in old["episodes"]}; n = {key(e): e for e in new["episodes"]}
assert set(o) == set(n), "episode sets differ"
same_act = sum(o[k]["actions"] == n[k]["actions"] for k in o); same_x = sum(o[k]["max_x"] == n[k]["max_x"] for k in o)
print(json.dumps(dict(seed=s, episodes=len(o), identical_actions=same_act, identical_max_x=same_x,
                      old=dict(mean=old["summary"]["mean"], ends=old["summary"]["ends"]), new=dict(mode=new["mode"], stuck=new["stuck"], mean=new["summary"]["mean"], ends=new["summary"]["ends"]))))
