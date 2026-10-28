"""Experiment drivers.  Usage: python experiments.py e1 e2 e3a e3b e4 e5   (default: all)
Each experiment writes results/<name>.json.  Information is in bits about which of K=4
stimulation protocols was delivered, from N cells' insertion data (count readout)."""
import json, sys, time
import numpy as np
from model import Channel, lik_table, single_cell_table, mc_mutual_information
from infotheory import mi_from_channel, blahut_arimoto
import multiplex as mx

K4 = np.log2(4)
NS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def save(name, obj):
    json.dump(obj, open(f"results/{name}.json", "w"), indent=1)


def exact_single(ch):
    return mi_from_channel(single_cell_table(ch))


def cap_single(ch, smax=None, npts=61):
    """Single-cell capacity (bits) via Blahut-Arimoto over a continuous dose grid [0, S_max]."""
    smax = max(ch.protocols) if smax is None else smax
    grid = tuple(np.linspace(0, smax, npts))
    return blahut_arimoto(single_cell_table(ch.but(protocols=grid)))[0]


def curve(ch, Ns, trials=400, decoder="optimal", seed=0):
    out = []
    shared = ch.sigma_z > 0 or ch.sigma_b > 0
    for N in Ns:
        if N == 1 and not shared:
            out.append((N, exact_single(ch), 0.0))
        else:
            I, se = mc_mutual_information(ch, N, trials=trials, seed=seed, decoder=decoder)
            out.append((N, I, se))
    return [list(map(float, r)) for r in out]


def n_for(cur, frac=0.9, K=4):
    """Smallest N (log-interpolated) reaching frac*log2K bits; None if not reached."""
    tgt = frac * np.log2(K)
    Ns = np.array([c[0] for c in cur]); I = np.array([c[1] for c in cur])
    if I[0] >= tgt:
        return float(Ns[0])
    for i in range(1, len(I)):
        if I[i] >= tgt:
            f = (tgt - I[i - 1]) / (I[i] - I[i - 1])
            return float(np.exp(np.log(Ns[i - 1]) + f * (np.log(Ns[i]) - np.log(Ns[i - 1]))))
    return None


# ---------------------------------------------------------------------------------------
def l1_channel(lam_bg, r, phi=0.0, p_d=1.0, clone_size=1, **kw):
    """Endogenous L1: lam_bg total somatic insertions/neuron (disputed 0.6..13.7), of which a
    fraction phi is clonally inherited; stimulus-driven increment at saturation = r*lam_bg."""
    return Channel(lam_bg=(1 - phi) * lam_bg * p_d / p_d, lam_act=r * lam_bg, p_d=p_d,
                   mu_c=phi * lam_bg * p_d, clone_size=clone_size if phi > 0 else 1, **kw)


def e1():
    """RESULT 1: endogenous LINE-1 over the whole published rate range."""
    rates = [0.6, 1.5, 3.0, 7.0, 13.7]
    gains = [0.1, 0.25, 0.5, 1.0]
    out = {"rates": rates, "gains": gains, "single_mi": {}, "single_cap": {}, "curves": {}, "n90": {},
           "single_mi_pd03": {}}
    for lb in rates:
        for r in gains:
            ch = l1_channel(lb, r, p_d=1.0)
            key = f"{lb}|{r}"
            out["single_mi"][key] = exact_single(ch)
            out["single_cap"][key] = cap_single(ch)
            out["single_mi_pd03"][key] = exact_single(l1_channel(lb, r, p_d=0.3))
            if r in (0.25, 1.0):
                cur = curve(ch, NS, trials=300)
                out["curves"][key] = cur
                out["n90"][key] = n_for(cur)
        print("  e1 rate", lb, flush=True)
    save("e1", out)
    return out


def e2():
    """RESULT 2: engineered recorders vs insertion rate and N."""
    out = {"cc": {}, "cast": {}}
    peaks = list(np.round(np.logspace(-1.5, 1.5, 13), 4))
    out["peaks"] = peaks
    out["cc"]["single_mi"] = []; out["cc"]["single_cap"] = []
    for pk in peaks:
        ch = Channel(lam_bg=0.05, lam_act=pk, p_d=0.3)
        out["cc"]["single_mi"].append(exact_single(ch))
        out["cc"]["single_cap"].append(cap_single(ch))
    out["cc"]["curves"] = {}
    for pk in (0.3, 1.0, 3.0, 10.0):
        cur = curve(Channel(lam_bg=0.05, lam_act=pk, p_d=0.3), NS[:8], trials=300)
        out["cc"]["curves"][str(pk)] = {"curve": cur, "n90": n_for(cur)}
        print("  e2 cc", pk, flush=True)
    for pint in (0.10, 0.25):
        for M in (1, 4, 14):
            lam = -np.log(1 - pint)
            ch = Channel(obs="binomial", M=M, lam_bg=0.005, lam_act=lam, p_d=0.3)
            cur = curve(ch, NS[:8], trials=300)
            key = f"p{pint}|M{M}"
            out["cast"][key] = {"single_mi": exact_single(ch), "single_cap": cap_single(ch),
                                "curve": cur, "n90": n_for(cur),
                                "mean_obs_at_peak": M * 0.3 * pint}
            print("  e2 cast", key, flush=True)
    save("e2", out)
    return out


def e3a():
    """RESULT 3a: clonal inheritance."""
    out = {"cc": {}, "l1": {}}
    for mu in (0.0, 1.0, 5.0):
        for c in (1, 2, 5, 10, 25):
            if mu == 0 and c > 1:
                continue
            for N in (100, 1000):
                ch = Channel(lam_bg=0.05, lam_act=3.0, p_d=0.3, mu_c=mu, clone_size=c)
                Io, so = mc_mutual_information(ch, N, trials=300, seed=1, decoder="optimal")
                In, sn = mc_mutual_information(ch, N, trials=300, seed=1, decoder="naive")
                out["cc"][f"{mu}|{c}|{N}"] = {"optimal": Io, "se": so, "naive": In}
        print("  e3a cc mu", mu, flush=True)
    for phi in (0.0, 0.5, 0.9):
        for N in (100, 1000, 10000):
            ch = l1_channel(3.0, 0.5, phi=phi, p_d=1.0, clone_size=10)
            Io, so = mc_mutual_information(ch, N, trials=300, seed=2, decoder="optimal")
            In, sn = mc_mutual_information(ch, N, trials=300, seed=2, decoder="naive")
            out["l1"][f"{phi}|{N}"] = {"optimal": Io, "se": so, "naive": In}
        print("  e3a l1 phi", phi, flush=True)
    save("e3a", out)
    return out


def e3b():
    """RESULT 3b: overproduction inhibition (biphasic dose response)."""
    smaxs = list(np.round(np.logspace(-1, 1.6, 17), 4))
    frac = np.array([0, 1 / 3, 2 / 3, 1.0])
    out = {"smax": smaxs, "series": {}}
    for kind in ("saturating", "biphasic"):
        for sig in (0.5, 1.0):
            for N in (1, 30):
                vals = []
                for s in smaxs:
                    ch = Channel(kind=kind, xi=1.0, protocols=tuple(s * frac), lam_bg=0.05,
                                 lam_act=3.0, p_d=0.3, sigma_o=sig)
                    if N == 1:
                        vals.append([exact_single(ch), 0.0])
                    else:
                        I, se = mc_mutual_information(ch, N, trials=300, seed=3)
                        vals.append([I, se])
                out["series"][f"{kind}|{sig}|{N}"] = vals
        print("  e3b", kind, flush=True)
    save("e3b", out)
    return out


def e4():
    """Shared (per-animal) nuisance: gain z and baseline scale b are unknown but common to all cells."""
    out = {"cc": {}, "l1": {}}
    combos = [(0, 0), (0, .25), (0, .5), (0, 1.0), (.25, 0), (.5, 0)]   # 1-D nuisance only; N<=1000
    for sz, sb in combos:
        ch = Channel(lam_bg=0.05, lam_act=3.0, p_d=0.3, sigma_z=sz, sigma_b=sb)
        out["cc"][f"{sz}|{sb}"] = curve(ch, [1, 10, 100, 1000], trials=300)
        print("  e4 cc", sz, sb, flush=True)
    for sz, sb in [(0, 0), (0, .35), (0, .7), (0, 1.4)]:
        ch = l1_channel(3.0, 0.5, p_d=1.0).but(sigma_z=sz, sigma_b=sb)
        out["l1"][f"{sz}|{sb}"] = curve(ch, [1, 10, 100, 1000], trials=300)
        print("  e4 l1", sz, sb, flush=True)
    save("e4", out)
    return out


def e5():
    """Multiplexing: two colours, two site-sets; site-resolved vs pooled readout."""
    out = {"chis": [0.0, 0.1, 0.3, 0.6], "Ns": [1, 3, 10, 30, 100], "resolved": {}, "pooled": {}, "single": {}}
    for chi in out["chis"]:
        ch = mx.TwoColour(chi=chi)
        L2, Lp = mx.tables(ch)
        out["single"][str(chi)] = {"resolved": mx.mi_exact(L2), "pooled": mx.mi_exact(Lp[:, None, :])}
        for ro in ("resolved", "pooled"):
            out[ro][str(chi)] = [list(mx.mc_mi(ch, N, trials=400, seed=4, readout=ro)) for N in out["Ns"]]
        print("  e5 chi", chi, flush=True)
    save("e5", out)
    return out


def e6():
    """Extrinsic-noise ceiling: single-cell capacity vs insertion rate for several expression-noise levels."""
    peaks = list(np.round(np.logspace(-1, 2.5, 15), 4))
    out = {"peaks": peaks, "sigmas": [0.0, 0.25, 0.5, 1.0], "cap": {}}
    for sg in out["sigmas"]:
        out["cap"][str(sg)] = [cap_single(Channel(lam_bg=0.05, lam_act=pk, p_d=0.3, sigma_o=sg, sigma_t=sg))
                               for pk in peaks]
        print("  e6 sigma", sg, flush=True)
    save("e6", out)
    return out


ALL = {"e1": e1, "e2": e2, "e3a": e3a, "e3b": e3b, "e4": e4, "e5": e5, "e6": e6}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        t0 = time.time()
        print(f"running {n} ...", flush=True)
        ALL[n]()
        print(f"{n} done in {time.time()-t0:.0f}s", flush=True)
