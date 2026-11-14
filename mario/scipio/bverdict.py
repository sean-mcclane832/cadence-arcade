"""Leg 2c run B: the registered rule on the SEL8 24-level means of the all11dag MLPs (4 seeds); all11 shown without a verdict.
Rule (design note, Leg 2c): clearly beats Cato2 if mean > 985 + 2 SE; worse learner if mean < 940 - 2 SE; else not told apart.
SE = SD / 2 with the sample SD (ddof = 1), fixed in the log before any number."""
import json, numpy as np
g = lambda t: json.load(open(f"scratch/play/mlp_{t}.json"))["summary"]
d = [g(f"all11dag_{s}") for s in range(4)]; m = np.array([x["mean"] for x in d])
mu, sd = m.mean(), m.std(ddof=1); se = sd / 2
v = "MLP clearly beats Cato2" if mu > 985 + 2 * se else ("MLP is the worse learner on equal data" if mu < 940 - 2 * se else "not told apart")
out = dict(all11dag=dict(means=m.tolist(), flags=[x["flags"] for x in d], stuck=[x["ends"]["stuck"] for x in d], mean=round(mu, 1), sd=round(sd, 1), se=round(se, 1),
                         upper=round(985 + 2 * se, 1), lower=round(940 - 2 * se, 1), verdict=v))
try:
    a = [g(f"all11_{s}") for s in range(2)]; out["all11"] = dict(means=[x["mean"] for x in a], flags=[x["flags"] for x in a], stuck=[x["ends"]["stuck"] for x in a])
except FileNotFoundError: pass
json.dump(out, open("scratch/play/runB_verdict.json", "w"), indent=1); print(json.dumps(out))
