"""Make all figures from results/*.json  ->  figs/*.png"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Channel, lik_table, dose_response
import experiments as E

# Okabe-Ito colour-blind-safe palette
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
J = lambda n: json.load(open(f"results/{n}.json"))
LOG2K = 2.0


def thresh(ax, y=1.8):
    ax.axhline(LOG2K, color="gray", lw=0.8, ls=":")
    ax.axhline(y, color="gray", lw=0.6, ls="--")
    ax.text(ax.get_xlim()[0], LOG2K + 0.02, "2 bits (all 4 protocols resolved)", fontsize=7, color="gray", va="bottom")


def xy(cur):
    a = np.array(cur)
    return a[:, 0], a[:, 1], a[:, 2]


def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.linspace(0, 6, 400)
    ax[0].plot(x, dose_response(x, "saturating", 1.0), color=C[0], label="saturating")
    ax[0].plot(x, dose_response(x, "biphasic", 1.0), color=C[1], label="biphasic (overproduction inhibition)")
    for s in (0, 1, 2, 3):
        ax[0].axvline(s, color="gray", lw=0.5, ls=":")
    ax[0].set(xlabel="effective dose x = S·ε_o", ylabel="relative transposase activity h(x)", title="Encoding: dose–response")
    ax[0].legend(frameon=False, fontsize=7)
    for a_, (ch, ttl) in zip(ax[1:], [(Channel(), "Engineered (Calling Cards-like)\npeak 3 ins/cell, p_d=0.3"),
                                       (E.l1_channel(3.0, 1.0, p_d=1.0), "Endogenous L1-like\n3 ins/neuron, activity gain r=1")]):
        L = lik_table(ch)
        n = np.arange(0, 14)
        for k in range(4):
            a_.step(n, L[k, : len(n)], where="mid", color=C[k], label=f"protocol {k} (S={ch.protocols[k]:g})")
        a_.set(xlabel="observed insertions per cell", ylabel="probability", title=ttl)
    ax[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Channel model: per-cell count distributions for four stimulation protocols", y=1.2, fontsize=10)
    fig.savefig("figs/fig1_model.png"); plt.close(fig)


def fig2():
    d = J("e1")
    rates, gains = d["rates"], d["gains"]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    M = np.array([[d["single_mi"][f"{lb}|{r}"] for r in gains] for lb in rates])
    im = ax[0].imshow(M, origin="lower", cmap="Blues", aspect="auto")
    ax[0].set_xticks(range(len(gains)), [f"{g:g}" for g in gains]); ax[0].set_yticks(range(len(rates)), [f"{r:g}" for r in rates])
    for i in range(len(rates)):
        for j in range(len(gains)):
            ax[0].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > M.max() * 0.55 else "black")
    ax[0].set(xlabel="activity gain r (stimulus-driven / baseline)", ylabel="baseline L1 insertions per neuron",
              title="Single-cell MI (bits), ideal detection"); ax[0].grid(False)
    for lb, c in zip(rates, [C[0], C[5], C[2], C[4], C[1]]):
        for r, ls in ((0.25, "--"), (1.0, "-")):
            cur = d["curves"][f"{lb}|{r}"]
            n, i, s = xy(cur)
            ax[1].plot(n, i, ls, color=c, label=f"{lb:g} / r={r:g}" if r == 1.0 else None, marker="o", ms=2.5)
            ax[1].fill_between(n, i - s, i + s, color=c, alpha=0.12)
    ax[1].set_xscale("log"); ax[1].set(xlabel="neurons N", ylabel="MI about protocol (bits)",
                                       title="Population MI (solid r=1, dashed r=0.25)")
    ax[1].legend(frameon=False, fontsize=7, title="baseline / gain", title_fontsize=7); thresh(ax[1])
    m3 = np.array([[d["single_mi_pd03"][f"{lb}|{r}"] / d["single_mi"][f"{lb}|{r}"] for r in gains] for lb in rates])
    for j, r in enumerate(gains):
        ax[2].plot(rates, [d["single_cap"][f"{lb}|{r}"] for lb in rates], marker="o", color=C[j], label=f"r={r:g}")
    ax[2].axhline(1.0, color="gray", ls=":", lw=0.8); ax[2].text(0.6, 1.02, "1 bit", fontsize=7, color="gray")
    ax[2].set(xscale="log", xlabel="baseline L1 insertions per neuron", ylabel="single-cell capacity (bits)", ylim=(0, 1.2),
              title="Capacity over continuous dose (Blahut–Arimoto)")
    ax[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Endogenous LINE-1 across the disputed rate range (0.6–13.7 per neuron)", y=1.04, fontsize=10)
    fig.savefig("figs/fig2_L1.png"); plt.close(fig)


def fig3():
    d = J("e2")
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    pk = d["peaks"]
    ax[0].plot(pk, d["cc"]["single_cap"], color=C[0], marker="o", ms=3, label="Calling Cards-like: capacity")
    ax[0].plot(pk, d["cc"]["single_mi"], color=C[0], ls="--", label="Calling Cards-like: MI (4 protocols)")
    for (key, v), c in zip(d["cast"].items(), [C[1], C[2], C[3], C[4], C[5], C[6]]):
        pint, M = key.replace("p", "").split("|M")
        ax[0].scatter([-np.log(1 - float(pint)) * int(M)], [v["single_cap"]], color=c, marker="s", s=22,
                      label=f"CAST p={pint}, M={M}")
    ax[0].set(xscale="log", xlabel="peak activity-driven insertions per cell (pre-detection)", ylabel="single-cell bits",
              title="Per-cell information vs insertion rate"); ax[0].legend(frameon=False, fontsize=6)
    for key, v in d["cc"]["curves"].items():
        n, i, s = xy(v["curve"]); ax[1].plot(n, i, marker="o", ms=2.5, label=f"CC peak {key}")
    ax[1].set_xscale("log"); ax[1].set(xlabel="cells N", ylabel="MI (bits)", title="Calling Cards-like (p_d = 0.3)"); ax[1].legend(frameon=False, fontsize=7); thresh(ax[1])
    for (key, v), c in zip(d["cast"].items(), [C[1], C[2], C[3], C[4], C[5], C[6]]):
        n, i, s = xy(v["curve"]); ax[2].plot(n, i, marker="o", ms=2.5, color=c, label=key.replace("|", " "))
    ax[2].set_xscale("log"); ax[2].set(xlabel="cells N", ylabel="MI (bits)", title="CAST-like, site-capped (p_d = 0.3)"); ax[2].legend(frameon=False, fontsize=6); thresh(ax[2])
    fig.suptitle("Engineered recorders: information vs insertion rate and cell number", y=1.04, fontsize=10)
    fig.savefig("figs/fig3_engineered.png"); plt.close(fig)


def fig4():
    d = J("e3a")
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for N, a_ in zip((100, 1000), ax[:2]):
        for mu, c in zip((0.0, 1.0, 5.0), (C[6], C[0], C[1])):
            cs = [1] if mu == 0 else [1, 2, 5, 10, 25]
            key = lambda cc: f"{mu}|{cc}|{N}"
            if mu == 0:
                a_.axhline(d["cc"][key(1)]["optimal"], color=c, ls=":", label="no clonal term")
                continue
            a_.errorbar(cs, [d["cc"][key(cc)]["optimal"] for cc in cs], [d["cc"][key(cc)]["se"] for cc in cs], color=c, marker="o", ms=3, label=f"μ_c={mu:g} clone-aware")
            a_.plot(cs, np.maximum([d["cc"][key(cc)]["naive"] for cc in cs], -2), color=c, ls="--", marker="x", ms=4, label=f"μ_c={mu:g} naive")
        a_.set(xscale="log", xlabel="clone size (cells sharing insertions)", ylabel="MI (bits)", title=f"Calling Cards-like, N={N}")
        a_.set_xticks([1, 2, 5, 10, 25], ["1", "2", "5", "10", "25"])
    ax[0].legend(frameon=False, fontsize=6)
    for phi, c in zip((0.0, 0.5, 0.9), (C[6], C[0], C[1])):
        Ns = [100, 1000, 10000]
        ax[2].errorbar(Ns, [d["l1"][f"{phi}|{N}"]["optimal"] for N in Ns], [d["l1"][f"{phi}|{N}"]["se"] for N in Ns], color=c, marker="o", ms=3, label=f"φ={phi:g} clone-aware")
        ax[2].plot(Ns, np.maximum([d["l1"][f"{phi}|{N}"]["naive"] for N in Ns], -2), color=c, ls="--", marker="x", ms=4, label=f"φ={phi:g} naive")
    ax[2].set(xscale="log", xlabel="neurons N", ylabel="MI (bits)", title="L1 (3/neuron, r=0.5), clone size 10")
    ax[2].legend(frameon=False, fontsize=6); thresh(ax[2])
    fig.suptitle("Clonally inherited insertions: clone-aware vs naive decoding (naive curves clipped at −2 bits)", y=1.04, fontsize=10)
    fig.savefig("figs/fig4_clonal.png"); plt.close(fig)


def fig5():
    d = J("e3b"); s = np.array(d["smax"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for a_, N in zip(ax, (1, 30)):
        for (kind, sig), (c, ls) in {("saturating", 0.5): (C[0], "-"), ("biphasic", 0.5): (C[1], "-"),
                                     ("saturating", 1.0): (C[0], "--"), ("biphasic", 1.0): (C[1], "--")}.items():
            v = np.array(d["series"][f"{kind}|{sig}|{N}"])
            a_.plot(s, v[:, 0], ls, color=c, label=f"{kind}, σ_o={sig:g}")
        a_.set(xscale="log", xlabel="S_max (protocols at 0, ⅓, ⅔, 1 × S_max; ξ=1)", ylabel="MI (bits)",
               title=f"N = {N} cell{'s' if N > 1 else ''}")
        a_.legend(frameon=False, fontsize=7)
    fig.suptitle("Overproduction inhibition: brighter stimulation can lower information", y=1.04, fontsize=10)
    fig.savefig("figs/fig5_inhibition.png"); plt.close(fig)


def fig6():
    d = J("e4")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (key, cur), c in zip(d["cc"].items(), [C[6], C[5], C[0], C[2], C[4], C[1]]):
        sz, sb = key.split("|"); n, i, se = xy(cur)
        ax[0].errorbar(n, i, se, color=c, marker="o", ms=3, label=f"σ_z={sz}, σ_b={sb}")
    ax[0].set(xscale="log", xlabel="cells N", ylabel="MI (bits)", title="Calling Cards-like (peak 3, p_d 0.3)"); ax[0].legend(frameon=False, fontsize=6); thresh(ax[0])
    for (key, cur), c in zip(d["l1"].items(), [C[6], C[5], C[0], C[1]]):
        sz, sb = key.split("|"); n, i, se = xy(cur)
        ax[1].errorbar(n, i, se, color=c, marker="o", ms=3, label=f"σ_b={sb}")
    ax[1].set(xscale="log", xlabel="neurons N", ylabel="MI (bits)", title="L1 (3/neuron, r=0.5) with unknown per-animal baseline"); ax[1].legend(frameon=False, fontsize=7); thresh(ax[1])
    fig.suptitle("Shared per-animal nuisance (light delivery z, baseline rate b) caps information regardless of N", y=1.04, fontsize=10)
    fig.savefig("figs/fig6_shared.png"); plt.close(fig)


def fig7():
    d = J("e5")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for chi, c in zip(d["chis"], [C[0], C[2], C[4], C[1]]):
        for ro, ls in (("resolved", "-"), ("pooled", "--")):
            v = np.array(d[ro][str(chi)])
            ax[0].errorbar(d["Ns"], v[:, 0], v[:, 1], color=c, ls=ls, marker="o", ms=3, label=f"{ro}, χ={chi:g}" if chi in (0.0, 0.6) or ro == "resolved" else None)
    ax[0].set_xscale("log"); ax[0].set(xlabel="cells N", ylabel="MI (bits)", title="Two colours, two site-sets"); ax[0].legend(frameon=False, fontsize=6)
    ax[0].axhline(1.5, color="gray", ls=":", lw=0.8); ax[0].text(1, 1.5 + 0.03, "1.5 bits = pooled ceiling (outcomes ¼, ½, ¼)", fontsize=7, color="gray")
    thresh(ax[0])
    x = np.arange(len(d["chis"]))
    ax[1].bar(x - 0.2, [d["single"][str(c)]["resolved"] for c in d["chis"]], 0.4, color=C[0], label="site-resolved")
    ax[1].bar(x + 0.2, [d["single"][str(c)]["pooled"] for c in d["chis"]], 0.4, color=C[1], label="pooled count")
    ax[1].set_xticks(x, [f"{c:g}" for c in d["chis"]]); ax[1].set(xlabel="optical cross-talk χ", ylabel="single-cell MI (bits)", title="Single cell"); ax[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Multiplexing: which site is hit encodes which colour", y=1.04, fontsize=10)
    fig.savefig("figs/fig7_multiplex.png"); plt.close(fig)


def fig8():
    d = J("e6")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for sg, c in zip(d["sigmas"], [C[6], C[5], C[0], C[1]]):
        ax.plot(d["peaks"], d["cap"][str(sg)], marker="o", ms=3, color=c, label=f"expression noise σ={sg:g}")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set(xscale="log", xlabel="peak activity-driven insertions per cell (pre-detection)", ylabel="single-cell capacity (bits)",
           title="Extrinsic noise flattens the capacity gain\n(Calling Cards-like, p_d=0.3, dose range 0–3)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig("figs/fig8_ceiling.png"); plt.close(fig)


if __name__ == "__main__":
    import sys, os
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f(); print("made", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
