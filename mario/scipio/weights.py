"""Leg 1, question 7 (weights study), part A: structure. numpy + pillow only, no brain is run.

Reads COPIES of finished checkpoints in scratch/ck (never the runs' own files). Writes scratch/weights_<run>.json
and pictures to weights/img/.

    idle.sh 098 weights.py Cato2A 0 83
    idle.sh 098 weights.py FabiusA 0 39

Checkpoint facts used (checked in cadence 0.95.0 reference.py): the weight of a synapse is its `efficacy`
(gain 1, count 1, log_gain 0), `sign` is the draw it was born with, neurons are indexed sensory, association,
prefrontal, motor (then recall, next for Fabius). Input layout (fabius/eyes.py, dagger.py):
  0..831 grey periphery now (26 x 32), 832..1663 periphery one decision ago, 1664..1999 body sense (42 x 8 copies),
  2000..3535 fovea: R, G, B now (16 x 24 each), then grey one decision ago at the same window.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CK = HERE / "scratch" / "ck"
IMG = HERE / "weights" / "img"; IMG.mkdir(parents=True, exist_ok=True)
ACTIONS = ["noop", "right", "right+jump", "run", "run+jump", "jump", "left"]
BODY = (["xspeed%d" % i for i in range(9)] + ["yspeed%d" % i for i in range(9)] +
        ["ground", "jumping", "falling", "pole", "rising", "face_right", "face_left", "small", "big", "fire",
         "hurt", "star"] + ["time%d" % i for i in range(8)] + ["held_A", "held_B", "held_right", "held_left"])
GROUPS = dict(periph_now=(0, 832), periph_prev=(832, 1664), body=(1664, 2000), fov_R=(2000, 2384), fov_G=(2384, 2768),
              fov_B=(2768, 3152), fov_prev=(3152, 3536))


def load(run, k):
    z = np.load(CK / f"{run}_ck{k:04d}.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    pops = {n: np.asarray(v) for n, v in meta["populations"].items()}
    return dict(pre=z["pre"], post=z["post"], eff=z["efficacy"], sign=z["sign"], bias=z["bias"], pops=pops, n=meta["n"], z=z)


def region_of(c):
    reg = np.full(c["n"], -1)
    names = [n for n in c["pops"] if "/" not in n]
    for i, n in enumerate(names):
        reg[c["pops"][n]] = i
    return reg, names


def block(c, a, b, eff=None):
    """Dense matrix [pre in region a, post in region b] of efficacies (zeros where no synapse)."""
    pa, pb = c["pops"][a], c["pops"][b]
    ia = np.full(c["n"], -1); ia[pa] = np.arange(len(pa))
    ib = np.full(c["n"], -1); ib[pb] = np.arange(len(pb))
    m = (ia[c["pre"]] >= 0) & (ib[c["post"]] >= 0)
    W = np.zeros((len(pa), len(pb)))
    W[ia[c["pre"][m]], ib[c["post"][m]]] = (c["eff"] if eff is None else eff)[m]
    return W, int(m.sum())


def div_png(a, path, scale=8, vmax=None):
    """Blue (negative) to white to red (positive) picture of a 2-D array."""
    a = np.asarray(a, float); v = vmax or (np.abs(a).max() or 1.0)
    t = np.clip(a / v, -1, 1)
    rgb = np.ones(a.shape + (3,))
    pos, neg = t > 0, t < 0
    rgb[pos, 1] = 1 - t[pos]; rgb[pos, 2] = 1 - t[pos]
    rgb[neg, 0] = 1 + t[neg]; rgb[neg, 1] = 1 + t[neg]
    im = Image.fromarray((rgb * 255).astype(np.uint8)).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path)


def heat_png(a, path, scale=8):
    a = np.asarray(a, float); t = a / (a.max() or 1.0)
    rgb = np.stack([t, t * 0.6, 1 - t], -1) * t[..., None] ** 0.5
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST).save(path)


def panel(vec, path, vmax=None):
    """One input-weight vector (3,536) as a picture: periphery now | periphery prev on top, fovea R G B prev below."""
    v = vmax or (np.abs(vec).max() or 1.0)
    top = np.concatenate([vec[0:832].reshape(26, 32), np.full((26, 2), 0.0), vec[832:1664].reshape(26, 32)], 1)
    fov = [vec[a:b].reshape(16, 24) for a, b in (GROUPS["fov_R"], GROUPS["fov_G"], GROUPS["fov_B"], GROUPS["fov_prev"])]
    bot = np.concatenate([np.concatenate([f, np.zeros((16, 1))], 1) for f in fov], 1)
    W = max(top.shape[1] * 8, bot.shape[1] * 8)
    div_png(top, path.with_suffix(".top.png"), 8, v); div_png(bot, path.with_suffix(".fov.png"), 8, v)
    a, b = Image.open(path.with_suffix(".top.png")), Image.open(path.with_suffix(".fov.png"))
    out = Image.new("RGB", (W, a.height + b.height + 6), (40, 40, 40)); out.paste(a, (0, 0)); out.paste(b, (0, a.height + 6))
    out.save(path); path.with_suffix(".top.png").unlink(); path.with_suffix(".fov.png").unlink()


def main():
    run, k0, k1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    c0, c1 = load(run, k0), load(run, k1)
    assert np.array_equal(c0["pre"], c1["pre"]) and np.array_equal(c0["post"], c1["post"])
    reg, names = region_of(c1)
    rp, rq = reg[c1["pre"]], reg[c1["post"]]
    out = dict(run=run, k0=k0, k1=k1, n=int(c1["n"]), synapses=int(len(c1["pre"])), regions={n: int(len(c1["pops"][n])) for n in names})
    out["born_equals_sign"] = bool(np.allclose(c0["eff"], c0["sign"]))
    # 1. every projection: size, norms, change
    blocks = {}
    d = c1["eff"] - c0["eff"]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m = (rp == i) & (rq == j)
            if m.any():
                w0, w1, dd = c0["eff"][m], c1["eff"][m], d[m]
                blocks[f"{a}->{b}"] = dict(synapses=int(m.sum()), norm0=float(np.linalg.norm(w0)), norm1=float(np.linalg.norm(w1)),
                                           change=float(np.linalg.norm(dd)), rel_change=float(np.linalg.norm(dd) / (np.linalg.norm(w0) + 1e-12)),
                                           corr_with_birth=float(np.corrcoef(w0, w1)[0, 1]), rms0=float(np.sqrt(np.mean(w0 ** 2))),
                                           rms1=float(np.sqrt(np.mean(w1 ** 2))), share_of_all_change=float((dd ** 2).sum() / (d ** 2).sum()),
                                           moved_share=float(np.mean(np.abs(dd) > 1e-9)))
    out["blocks"] = blocks
    out["bias"] = {n: dict(rms0=float(np.sqrt(np.mean(c0["bias"][c1["pops"][n]] ** 2))), rms1=float(np.sqrt(np.mean(c1["bias"][c1["pops"][n]] ** 2)))) for n in names}
    # 2. sensory -> association
    Wsa0, _ = block(c0, "sensory", "association"); Wsa, _ = block(c1, "sensory", "association")
    D = Wsa - Wsa0
    e_in = (D ** 2).sum(1); e_unit = (D ** 2).sum(0)
    out["sa_change_by_group"] = {g: dict(share=float(e_in[a:b].sum() / e_in.sum()), per_input=float(e_in[a:b].mean()),
                                         inputs=b - a) for g, (a, b) in GROUPS.items()}
    body = e_in[1664:2000].reshape(8, 42).sum(0)
    out["sa_change_body_top"] = [(BODY[i], round(float(body[i] / e_in.sum()), 4)) for i in np.argsort(-body)[:12]]
    out["sa_change_unit_gini"] = dict(top10_units_share=float(np.sort(e_unit)[-10:].sum() / e_unit.sum()),
                                      top50_units_share=float(np.sort(e_unit)[-50:].sum() / e_unit.sum()),
                                      min_unit=float(e_unit.min() / e_unit.mean()), max_unit=float(e_unit.max() / e_unit.mean()))
    panel(np.sqrt(e_in), IMG / f"{run}_change_energy.png")
    # 3. association <-> motor (buttons are the first 7 motor neurons)
    Wam0, _ = block(c0, "association", "motor"); Wam, _ = block(c1, "association", "motor")
    Wma, _ = block(c1, "motor", "association")
    out["reciprocal_equal"] = bool(np.allclose(Wam, Wma.T))
    B = Wam[:, :7]
    out["am_buttons"] = dict(rms0=float(np.sqrt(np.mean(Wam0[:, :7] ** 2))), rms1=float(np.sqrt(np.mean(B ** 2))),
                             per_button_norm={ACTIONS[i]: round(float(np.linalg.norm(B[:, i])), 3) for i in range(7)},
                             per_button_bias={ACTIONS[i]: round(float(c1["bias"][c1["pops"]["motor"][i]]), 4) for i in range(7)},
                             button_corr=np.round(np.corrcoef(B.T), 2).tolist())
    if Wam.shape[1] > 7:
        S = Wam[:, 7:]
        out["am_slots"] = dict(rms0=float(np.sqrt(np.mean(Wam0[:, 7:] ** 2))), rms1=float(np.sqrt(np.mean(S ** 2))),
                               change_rel=float(np.linalg.norm(S - Wam0[:, 7:]) / np.linalg.norm(Wam0[:, 7:])),
                               buttons_change_rel=float(np.linalg.norm(B - Wam0[:, :7]) / np.linalg.norm(Wam0[:, :7])))
    div_png(B.T, IMG / f"{run}_units_to_buttons.png", 4)
    # how many units carry each button: participation ratio of the column
    out["am_participation"] = {ACTIONS[i]: round(float((B[:, i] ** 2).sum() ** 2 / (B[:, i] ** 4).sum()), 1) for i in range(7)}
    # 4. first-order input -> button map (through the association, slopes ignored): L = Wsa @ B
    L = Wsa @ B; L0 = Wsa0 @ Wam0[:, :7]
    out["linear_map"] = dict(norm_by_group={g: {ACTIONS[i]: round(float(np.linalg.norm(L[a:b, i])), 3) for i in range(7)} for g, (a, b) in GROUPS.items()},
                             birth_norm_total=float(np.linalg.norm(L0)), now_norm_total=float(np.linalg.norm(L)))
    vmax = float(np.percentile(np.abs(L), 99.9))
    for i in range(7):
        panel(L[:, i], IMG / f"{run}_button_{i}_{ACTIONS[i].replace('+', '_')}.png", vmax)
    # contrasts that matter in play: jump family vs no-jump family, right vs left/noop
    jump = L[:, [2, 4, 5]].mean(1) - L[:, [0, 1, 3, 6]].mean(1)
    panel(jump, IMG / f"{run}_contrast_jump_vs_nojump.png")
    go = L[:, [1, 2, 3, 4]].mean(1) - L[:, [0, 5, 6]].mean(1)
    panel(go, IMG / f"{run}_contrast_go_right_vs_stay.png")
    Lb = L[1664:2000].reshape(8, 42, 7).sum(0)
    out["body_to_buttons"] = {BODY[j]: [round(float(x), 3) for x in Lb[j]] for j in range(42)}
    out["body_copies_alike"] = float(np.mean([np.corrcoef(L[1664:1706].ravel(), L[1664 + 42 * c:1706 + 42 * c].ravel())[0, 1] for c in range(1, 8)]))
    jb = jump[1664:2000].reshape(8, 42).sum(0)
    out["jump_contrast_body_top"] = [(BODY[j], round(float(jb[j]), 3)) for j in np.argsort(-np.abs(jb))[:10]]
    out["jump_contrast_share_by_group"] = {g: round(float((jump[a:b] ** 2).sum() / (jump ** 2).sum()), 3) for g, (a, b) in GROUPS.items()}
    # 5. unit pictures: the 8 units that push the jump family hardest (4 for, 4 against)
    jw = B[:, [2, 4, 5]].mean(1) - B[:, [0, 1, 3, 6]].mean(1)
    order = np.argsort(jw); picks = list(order[-4:][::-1]) + list(order[:4])
    out["unit_pictures"] = [dict(unit=int(u), jump_weight=round(float(jw[u]), 3), file=f"{run}_unit_{int(u)}.png") for u in picks]
    for u in picks:
        panel(Wsa[:, u], IMG / f"{run}_unit_{int(u)}.png")
    # 6. prefrontal (working trace) -> association
    if "prefrontal" in c1["pops"]:
        Wpa0, _ = block(c0, "prefrontal", "association"); Wpa, _ = block(c1, "prefrontal", "association")
        out["trace_path"] = dict(rms0=float(np.sqrt(np.mean(Wpa0 ** 2))), rms1=float(np.sqrt(np.mean(Wpa ** 2))),
                                 rel_change=float(np.linalg.norm(Wpa - Wpa0) / np.linalg.norm(Wpa0)),
                                 fan_in_norm_trace=float(np.mean(np.linalg.norm(Wpa, axis=0))),
                                 fan_in_norm_senses=float(np.mean(np.linalg.norm(Wsa, axis=0))))
    # 7. Fabius extras: recall and next regions into the association, and the recall memory's own maps
    for extra in ("recall", "next"):
        if extra in c1["pops"]:
            W0, _ = block(c0, extra, "association"); W1, _ = block(c1, extra, "association")
            DD = W1 - W0; ee = (DD ** 2).sum(1)
            out[f"{extra}_path"] = dict(rms0=float(np.sqrt(np.mean(W0 ** 2))), rms1=float(np.sqrt(np.mean(W1 ** 2))),
                                        rel_change=float(np.linalg.norm(DD) / np.linalg.norm(W0)),
                                        fan_in_norm=float(np.mean(np.linalg.norm(W1, axis=0))),
                                        change_by_group={g: round(float(ee[a:b].sum() / ee.sum()), 3) for g, (a, b) in GROUPS.items()},
                                        linear_to_buttons_norm=float(np.linalg.norm(W1 @ B)),
                                        senses_linear_to_buttons_norm=float(np.linalg.norm(L)))
    rec = {}
    for key in c1["z"].files:
        if key.startswith("memories/recall/") and c1["z"][key].size and c1["z"][key].dtype.kind == "f":
            a1 = c1["z"][key]; a0 = c0["z"][key] if key in c0["z"].files and c0["z"][key].shape == a1.shape else None
            rec[key.split("/")[-1]] = dict(shape=list(a1.shape), rms1=float(np.sqrt(np.mean(a1 ** 2))),
                                           rms0=None if a0 is None else float(np.sqrt(np.mean(a0 ** 2))), nonzero=int(np.count_nonzero(a1)))
    if rec:
        out["recall_maps"] = rec
    br = json.loads(str(c1["z"]["brain"]))
    out["carried_gain"] = br.get("carried_gain"); out["memories_cfg"] = br.get("memories")
    (HERE / "scratch" / f"weights_{run}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(HERE / "scratch" / f"weights_{run}_mats.npz", Wsa=Wsa.astype(np.float32), B=B.astype(np.float32), L=L.astype(np.float32),
                        bias_assoc=c1["bias"][c1["pops"]["association"]], bias_motor=c1["bias"][c1["pops"]["motor"]])
    print(json.dumps({k: out[k] for k in out if k not in ("body_to_buttons", "linear_map", "memories_cfg")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
