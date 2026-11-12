import json, sys
d = json.load(open(sys.argv[1]))
print("born_equals_sign", d["born_equals_sign"], "regions", d["regions"])
for k, b in d["blocks"].items():
    print(f"{k:28s} syn {b['synapses']:8d} rms {b['rms0']:.4f}->{b['rms1']:.4f} relchg {b['rel_change']:.2f} corr_birth {b['corr_with_birth']:.2f} share_of_change {b['share_of_all_change']:.3f}")
print("sa_change_by_group", {g: (round(v["share"], 3), round(v["per_input"], 2)) for g, v in d["sa_change_by_group"].items()})
print("body top", d["sa_change_body_top"][:6])
print("am_buttons norms", d["am_buttons"]["per_button_norm"], "bias", d["am_buttons"]["per_button_bias"])
print("corr right~right+jump~jump", d["am_buttons"]["button_corr"][1][2], d["am_buttons"]["button_corr"][1][5], d["am_buttons"]["button_corr"][2][5])
print("participation", d["am_participation"])
print("am_slots", d.get("am_slots"))
print("jump share by group", d["jump_contrast_share_by_group"], "body top", d["jump_contrast_body_top"][:6])
print("trace_path", d.get("trace_path"))
for e in ("recall_path", "next_path"):
    print(e, d.get(e))
print("recall_maps", {k: (v["rms0"], round(v["rms1"], 5), v["nonzero"]) for k, v in d.get("recall_maps", {}).items()})
print("carried_gain", d["carried_gain"])
print("linear norm birth/now", d["linear_map"]["birth_norm_total"], d["linear_map"]["now_norm_total"])
