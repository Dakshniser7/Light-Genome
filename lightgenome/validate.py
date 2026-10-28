"""Validation suite: every check compares the model with an independent analytic or
brute-force answer.  Run:  python validate.py   (writes results/validation.json)"""
import json, sys, time
import numpy as np
from scipy.stats import poisson, binom
from model import (Channel, lik_table, single_cell_table, simulate, population_loglik,
                   mc_mutual_information, dose_response, pois_pmf)
from infotheory import (mi_from_channel, blahut_arimoto, mi_plugin, mi_miller_madow,
                        mi_shuffle_corrected)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def t_rows_sum():
    for ch in [Channel(), Channel(kind="biphasic", sigma_o=0.8), Channel(obs="binomial", M=4, p_d=0.5)]:
        L = lik_table(ch)
        check("likelihood rows sum to 1", np.allclose(L.sum(1), 1, atol=1e-6), f"min={L.sum(1).min():.9f}")


def t_sampler_vs_table():
    ch = Channel()
    rng = np.random.default_rng(1)
    k, x = simulate(ch, 1, 400000, rng)
    L = lik_table(ch)
    worst = 0
    for kk in range(ch.K):
        emp = np.bincount(x[k == kk, 0], minlength=L.shape[1])[: L.shape[1]] / (k == kk).sum()
        worst = max(worst, np.abs(emp - L[kk, : len(emp)]).max())
    check("mechanistic sampler matches likelihood table (poisson)", worst < 4e-3, f"max|diff|={worst:.4f}")
    chb = Channel(obs="binomial", M=4, p_d=0.5, lam_act=0.8, lam_bg=0.05)
    k, x = simulate(chb, 1, 1200000, rng)
    L = lik_table(chb)
    worst = 0
    for kk in range(chb.K):
        n_k = (k == kk).sum()
        emp = np.bincount(x[k == kk, 0], minlength=chb.M + 1) / n_k
        se = np.sqrt(np.maximum(L[kk] * (1 - L[kk]), 1e-12) / n_k)
        worst = max(worst, (np.abs(emp - L[kk]) / se).max())
    check("mechanistic sampler matches likelihood table (site-capped)", worst < 4.5, f"max|diff|={worst:.2f} SE")


def t_thinning():
    ch = Channel(sigma_o=0, sigma_t=0, p_d=0.3, lam_bg=0, lam_act=10, protocols=(100.0,), xi=1e-9)
    rng = np.random.default_rng(2)
    _, x = simulate(ch, 1, 300000, rng)
    x = x[:, 0]
    # h ~ 1 -> lam = 10 ; thinning of Poisson(10) by 0.3 is Poisson(3)
    check("Poisson thinning: mean = p_d*lam", abs(x.mean() - 3.0) < 0.02, f"mean={x.mean():.3f}")
    check("Poisson thinning: Fano factor = 1", abs(x.var() / x.mean() - 1) < 0.02, f"Fano={x.var()/x.mean():.3f}")


def t_zero_info_and_deterministic():
    ch0 = Channel(lam_act=0.0)
    L = lik_table(ch0)
    check("no activity coupling -> exact MI = 0", mi_from_channel(L) < 1e-9, f"{mi_from_channel(L):.2e}")
    I, se = mc_mutual_information(ch0, N=10, trials=400)
    check("no activity coupling -> MC MI(N=10) ~ 0", abs(I) < 4 * se + 0.02, f"{I:.4f}+-{se:.4f}")
    chd = Channel(protocols=(0.0, 1.0, 5.0, 100.0), lam_bg=0, lam_act=2000.0, p_d=1.0, sigma_o=0, sigma_t=0)
    I = mi_from_channel(lik_table(chd))
    check("deterministic channel -> MI -> log2 K", I > 1.99, f"{I:.4f} bits (max 2)")


def t_ba():
    p = 0.1
    W = np.array([[1 - p, p], [p, 1 - p]])
    C, _ = blahut_arimoto(W)
    ref = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
    check("Blahut-Arimoto: binary symmetric channel", abs(C - ref) < 1e-6, f"{C:.6f} vs {ref:.6f}")
    p = 0.5
    W = np.array([[1, 0], [p, 1 - p]])
    C, _ = blahut_arimoto(W)
    ref = np.log2(1 + (1 - p) * p ** (p / (1 - p)))
    check("Blahut-Arimoto: Z-channel", abs(C - ref) < 1e-6, f"{C:.6f} vs {ref:.6f}")


def t_mc_vs_exact():
    ch = Channel()
    exact = mi_from_channel(lik_table(ch))
    I, se = mc_mutual_information(ch, N=1, trials=40000, seed=3)
    check("MC MI (N=1) matches exact single-cell MI", abs(I - exact) < 4 * se + 1e-3, f"MC {I:.4f}+-{se:.4f}, exact {exact:.4f}")


def t_estimators():
    ch = Channel(lam_act=1.0, p_d=0.5)
    L = lik_table(ch)
    exact = mi_from_channel(L)
    rng = np.random.default_rng(4)
    n = 200
    res = {"plugin": [], "miller_madow": [], "shuffle": []}
    for _ in range(60):
        k = np.arange(n) % ch.K
        x = np.array([rng.choice(L.shape[1], p=L[kk] / L[kk].sum()) for kk in k])
        res["plugin"].append(mi_plugin(k, x)[0])
        res["miller_madow"].append(mi_miller_madow(k, x))
        res["shuffle"].append(mi_shuffle_corrected(k, x, nshuf=20))
    b = {m: np.mean(v) - exact for m, v in res.items()}
    check("plug-in MI is biased upward at n=200", b["plugin"] > 0.01, f"bias={b['plugin']:+.4f} (exact {exact:.4f})")
    check("Miller-Madow reduces |bias|", abs(b["miller_madow"]) < abs(b["plugin"]), f"bias={b['miller_madow']:+.4f}")
    check("shuffle correction reduces |bias|", abs(b["shuffle"]) < abs(b["plugin"]), f"bias={b['shuffle']:+.4f}")
    return b, exact


def t_sufficiency():
    # No extrinsic noise: total count over N cells is sufficient, ~Poisson(N*p_d*lam_k).
    N = 20
    ch = Channel(sigma_o=0, sigma_t=0, lam_act=0.5, lam_bg=0.05, p_d=0.3)
    lam = ch.lam_bg + ch.lam_act * dose_response(np.array(ch.protocols), ch.kind, ch.xi)
    n = np.arange(0, 200)
    W = np.stack([poisson.pmf(n, N * ch.p_d * l) for l in lam])
    exact = mi_from_channel(W)
    I, se = mc_mutual_information(ch, N=N, trials=20000, seed=5)
    check("population MI equals MI of the sufficient statistic (total count)", abs(I - exact) < 4 * se + 2e-3,
          f"MC {I:.4f}+-{se:.4f}, exact {exact:.4f}")


def t_clonal_bruteforce():
    ch = Channel(mu_c=2.0, clone_size=2, sigma_o=0.4, sigma_t=0.4)
    La = lik_table(ch)
    pairs = [(0, 0), (1, 3), (4, 2), (5, 5), (2, 7)]
    counts = np.array([list(p) for p in pairs])
    ll = population_loglik(ch, counts, "optimal")
    ok = True
    worst = 0
    for i, (a, b) in enumerate(pairs):
        for k in range(ch.K):
            s = 0.0
            for m in range(0, min(a, b) + 1):
                s += pois_pmf(m, ch.mu_c) * La[k, a - m] * La[k, b - m]
            worst = max(worst, abs(np.log(s) - ll[i, k]))
    check("clone-aware likelihood matches brute-force sum", worst < 1e-8, f"max|dlogL|={worst:.2e}")


def t_clonal_decoders():
    ch = Channel(mu_c=3.0, clone_size=5, lam_act=3.0)
    Io, so = mc_mutual_information(ch, N=100, trials=800, seed=6, decoder="optimal")
    In, sn = mc_mutual_information(ch, N=100, trials=800, seed=6, decoder="naive")
    check("clone-aware decoder >= naive decoder (paired data)", Io >= In - 0.02,
          f"optimal {Io:.3f}, naive {In:.3f}")
    check("MI never exceeds log2 K", Io <= np.log2(ch.K) + 4 * so, f"{Io:.3f} <= 2")
    chn = ch.but(mu_c=0.0, clone_size=1)
    a, _ = mc_mutual_information(chn, N=100, trials=800, seed=7, decoder="optimal")
    b, _ = mc_mutual_information(chn, N=100, trials=800, seed=7, decoder="naive")
    check("without clonality both decoders coincide", abs(a - b) < 1e-9, f"{a:.4f} vs {b:.4f}")


def t_shared_gain_marginal():
    # shared z and per-cell eps_o both multiply dose -> N=1 channel equals one with sigma_o' = sqrt(so^2+sz^2)
    ch = Channel(sigma_o=0.0, sigma_z=0.6, sigma_t=0.5)
    ref = Channel(sigma_o=0.6, sigma_z=0.0, sigma_t=0.5)
    exact = mi_from_channel(lik_table(ref, nq=41))
    I, se = mc_mutual_information(ch, N=1, trials=40000, seed=8)
    check("shared-gain marginalisation (N=1) matches single-cell exact", abs(I - exact) < 4 * se + 2e-3,
          f"MC {I:.4f}+-{se:.4f}, exact {exact:.4f}")


def t_shared_grid_converged():
    import model
    ch = Channel(lam_bg=3.0, lam_act=1.5, p_d=1.0, sigma_b=0.6)
    res = []
    for scale in (1.0, 0.5):
        model.GRID_SCALE = scale
        res.append(mc_mutual_information(ch, N=300, trials=300, seed=11)[0])
    model.GRID_SCALE = 1.0
    check("shared-nuisance integration converged (grid spacing halved, N=300)", abs(res[0] - res[1]) < 0.02,
          f"{res[0]:.4f} vs {res[1]:.4f}")


def t_shapes_and_bounds():
    x = np.linspace(0, 20, 2001)
    hb = dose_response(x, "biphasic", 1.5)
    hs = dose_response(x, "saturating", 1.5)
    check("biphasic response peaks at x = xi with value 1", abs(x[hb.argmax()] - 1.5) < 0.02 and abs(hb.max() - 1) < 1e-3)
    check("saturating response is monotone", np.all(np.diff(hs) >= 0))
    ch = Channel()
    grid = np.linspace(0, 3, 61)
    W = lik_table(ch.but(protocols=tuple(grid)))
    C, _ = blahut_arimoto(W)
    I = mi_from_channel(lik_table(ch))
    check("capacity (dose grid) >= MI of uniform 4-protocol input", C >= I - 1e-9, f"C={C:.3f} >= I={I:.3f}")
    prev = -1
    vals = []
    for N in (1, 3, 10, 30):
        v, _ = mc_mutual_information(ch, N=N, trials=800, seed=9)
        vals.append(v)
    check("MI non-decreasing in N", all(np.diff(vals) > -0.03), str(np.round(vals, 3)))


if __name__ == "__main__":
    t0 = time.time()
    t_rows_sum(); t_sampler_vs_table(); t_thinning(); t_zero_info_and_deterministic(); t_ba()
    t_mc_vs_exact(); bias, exact = t_estimators(); t_sufficiency(); t_clonal_bruteforce()
    t_clonal_decoders(); t_shared_gain_marginal(); t_shared_grid_converged(); t_shapes_and_bounds()
    npass = sum(r["pass"] for r in RESULTS)
    print(f"\n{npass}/{len(RESULTS)} checks passed in {time.time()-t0:.0f}s")
    json.dump({"checks": RESULTS, "estimator_bias_bits": bias, "estimator_exact_bits": exact},
              open("results/validation.json", "w"), indent=1)
    sys.exit(0 if npass == len(RESULTS) else 1)
