"""Collect the headline numbers that fill the abstract placeholders -> results/summary.json + stdout."""
import json
import numpy as np

J = lambda n: json.load(open(f"results/{n}.json"))
S = {}

# ---- RESULT 1: endogenous L1 -------------------------------------------------------------
e1 = J("e1")
sm = np.array(list(e1["single_mi"].values())); cp = np.array(list(e1["single_cap"].values()))
S["R1_single_cell_MI_bits_range"] = [float(sm.min()), float(sm.max())]
S["R1_single_cell_capacity_bits_range"] = [float(cp.min()), float(cp.max())]
S["R1_capacity_best_case"] = {"baseline": 13.7, "r": 1.0, "bits": e1["single_cap"]["13.7|1.0"]}
S["R1_capacity_lowest_rate_r1"] = e1["single_cap"]["0.6|1.0"]
S["R1_cells_for_90pct_of_2bits"] = {k: v for k, v in e1["n90"].items()}

# ---- RESULT 2: engineered ----------------------------------------------------------------
e2 = J("e2")
pk = np.array(e2["peaks"])
S["R2_CC_capacity_at_peak"] = {f"{p:g}": float(c) for p, c in zip(pk[::3], np.array(e2["cc"]["single_cap"])[::3])}
S["R2_CC_cells_for_90pct"] = {k: v["n90"] for k, v in e2["cc"]["curves"].items()}
S["R2_CAST_single_cell_capacity"] = {k: v["single_cap"] for k, v in e2["cast"].items()}
S["R2_CAST_cells_for_90pct"] = {k: v["n90"] for k, v in e2["cast"].items()}
e6 = J("e6")
S["R2_extrinsic_noise_capacity_at_peak_300"] = {sg: v[-1] for sg, v in e6["cap"].items()}

# ---- RESULT 3: clonality, inhibition, shared nuisance --------------------------------------
e3a = J("e3a")
S["R3a_CC_N100_mu1"] = {c: {"clone_aware": e3a["cc"][f"1.0|{c}|100"]["optimal"], "naive": e3a["cc"][f"1.0|{c}|100"]["naive"]}
                        for c in (1, 2, 5, 10, 25)}
S["R3a_no_clonal_N100"] = e3a["cc"]["0.0|1|100"]["optimal"]
e3b = J("e3b")
inh = {}
for key, v in e3b["series"].items():
    a = np.array(v)[:, 0]
    inh[key] = {"max_bits": float(a.max()), "S_at_max": e3b["smax"][int(a.argmax())], "bits_at_Smax_39.8": float(a[-1])}
S["R3b_overproduction_inhibition"] = inh
e4 = J("e4")
S["R3c_shared_gain_N1000_bits"] = {k: v[-1][1] for k, v in e4["cc"].items()}
S["R3c_L1_shared_baseline_N1000_bits"] = {k: v[-1][1] for k, v in e4["l1"].items()}
e5 = J("e5")
S["R4_multiplex"] = {"site_resolved_N100": {c: v[-1][0] for c, v in e5["resolved"].items()},
                     "pooled_N100": {c: v[-1][0] for c, v in e5["pooled"].items()}}
json.dump(S, open("results/summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
