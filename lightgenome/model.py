"""Layered channel model: optogenetic stimulus -> transposase activity -> insertions
-> persistence (clonal inheritance) -> sequencing readout (thinning).

Layers
  1 stimulus     : protocol k has intensity S_k (arbitrary dose units)
  2 encoding     : dose x = S_k * eps_o * z   (eps_o per-cell opsin gain, z shared gain)
                   activity h(x): 'saturating' x/(x+xi)  or 'biphasic' 2u/(1+u^2), u=x/xi
                   (biphasic = overproduction inhibition, peak 1 at x = xi)
                   insertion rate  lam = b*lam_bg + lam_act * eps_t * h(x)
                   eps_t = per-cell transposase expression gain, b = shared baseline scale
  3 persistence  : mu_c extra insertions shared by every cell in a clone (no stimulus info)
  4 readout      : detection thinning p_d.
                   obs='poisson'  : counts ~ Poisson(p_d * lam)   (L1, Calling Cards)
                   obs='binomial' : M sites, each hit with prob 1-exp(-lam_site), detected
                                    with prob p_d (CAST-like, at most one insertion per site)

lam_bg / lam_act are totals per cell for 'poisson' and per site for 'binomial'.
All per-cell gains are lognormal with mean 1 (log-sd sigma_*).
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.special import gammaln, logsumexp
from numpy.polynomial.hermite_e import hermegauss


@dataclass(frozen=True)
class Channel:
    protocols: tuple = (0.0, 1.0, 2.0, 3.0)
    kind: str = "saturating"
    xi: float = 1.0
    lam_bg: float = 0.05
    lam_act: float = 3.0
    p_d: float = 0.3
    obs: str = "poisson"
    M: int = 1
    sigma_o: float = 0.5
    sigma_t: float = 0.5
    sigma_z: float = 0.0     # shared (per animal/trial) gain on dose
    sigma_b: float = 0.0     # shared (per animal/trial) scale on baseline
    mu_c: float = 0.0        # observed clonal insertions shared within a clone
    clone_size: int = 1

    def but(self, **kw):
        return replace(self, **kw)

    @property
    def K(self):
        return len(self.protocols)


def gh_nodes(n, sigma):
    """Quadrature nodes/weights for a mean-one lognormal gain with log-sd sigma."""
    if sigma <= 0 or n <= 1:
        return np.array([1.0]), np.array([1.0])
    x, w = hermegauss(n)
    return np.exp(sigma * x - 0.5 * sigma ** 2), w / w.sum()


def dose_response(x, kind, xi):
    if kind == "saturating":
        return x / (x + xi)
    if kind == "biphasic":
        u = x / xi
        return 2 * u / (1 + u * u)
    raise ValueError(kind)


def _rates(ch, eo, et, z, b):
    """lam[k, o, t] for quadrature gains."""
    S = np.asarray(ch.protocols, float)[:, None, None]
    x = S * eo[None, :, None] * z
    h = dose_response(x, ch.kind, ch.xi)
    return b * ch.lam_bg + ch.lam_act * et[None, None, :] * h


def _pmf_grid(ch, lam, nmax):
    n = np.arange(nmax + 1)
    if ch.obs == "poisson":
        mean = ch.p_d * lam[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = n * np.log(mean) - mean - gammaln(n + 1)
        lp = np.where(mean == 0, np.where(n == 0, 0.0, -np.inf), lp)
        return np.exp(lp)
    if ch.obs == "binomial":
        q = ch.p_d * (1 - np.exp(-lam))[..., None]
        M = ch.M
        n_ = n[None, None, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lc = gammaln(M + 1) - gammaln(n_ + 1) - gammaln(M - n_ + 1)
            lp = lc + n_ * np.log(q) + (M - n_) * np.log1p(-q)
        valid = n_ <= M
        lp = np.where(q == 0, np.where(n_ == 0, 0.0, -np.inf), lp)
        return np.where(valid, np.exp(lp), 0.0)
    raise ValueError(ch.obs)


def lik_table(ch, z=1.0, b=1.0, nq=15, nmax=None):
    """Activity-part likelihood L[k, n] = P(n observed | protocol k), marginalised over
    per-cell gains (no clonal term)."""
    eo, wo = gh_nodes(nq, ch.sigma_o)
    et, wt = gh_nodes(nq, ch.sigma_t)
    lam = _rates(ch, eo, et, z, b)
    w = wo[:, None] * wt[None, :]
    if ch.obs == "binomial":
        nmax = ch.M
    if nmax is None:
        nmax = 64
        while True:
            L = np.einsum("ot,kotn->kn", w, _pmf_grid(ch, lam, nmax))
            if 1 - L.sum(1).min() < 1e-7:
                break
            if nmax >= 8192:
                if 1 - L.sum(1).min() < 1e-6:     # extreme quadrature nodes only; negligible
                    break
                raise ValueError(f"count distribution tail not captured (missing mass "
                                 f"{1 - L.sum(1).min():.2e}); rates too large for the grid")
            nmax *= 2
    else:
        L = np.einsum("ot,kotn->kn", w, _pmf_grid(ch, lam, nmax))
    return L


def pois_pmf(m, mu):
    m = np.asarray(m)
    if mu == 0:
        return (m == 0).astype(float)
    return np.exp(m * np.log(mu) - mu - gammaln(m + 1))


def single_cell_table(ch, z=1.0, b=1.0, nq=15):
    """Table the single-cell observer sees: activity part convolved with clonal part."""
    L = lik_table(ch, z, b, nq)
    if ch.mu_c > 0 and ch.obs == "poisson":
        n = L.shape[1]
        extra = int(ch.mu_c + 10 * np.sqrt(ch.mu_c) + 20)
        L = lik_table(ch, z, b, nq, nmax=n + extra)
        c = pois_pmf(np.arange(L.shape[1]), ch.mu_c)
        L = np.stack([np.convolve(L[k], c)[: L.shape[1]] for k in range(L.shape[0])])
    return L


GRID_SCALE = 1.0   # multiplies the shared-nuisance grid spacing (used by the convergence test)


def shared_nodes(ch, N):
    """Integration grid over the shared (per-animal) nuisance gains z and b.
    Dense uniform grid in log-gain whose spacing shrinks as 1/sqrt(N): with N cells the
    posterior over a shared parameter has width ~1/sqrt(N), so a fixed coarse grid would
    miss it (this failed visibly at N=1e4 with 7 Gauss-Hermite nodes).
    If both z and b are random the grid is 2-D and is coarsened, so use it for small N only."""
    dlog = GRID_SCALE * float(np.clip(0.5 / np.sqrt(N), 0.012, 0.1))

    def axis(sigma, d):
        if sigma <= 0:
            return np.array([1.0]), np.array([1.0])
        x = np.arange(-5.0, 5.0 + d / sigma / 2, d / sigma)
        w = np.exp(-0.5 * x ** 2)
        return np.exp(sigma * x - 0.5 * sigma ** 2), w / w.sum()

    zs, wz = axis(ch.sigma_z, dlog)
    bs, wb = axis(ch.sigma_b, dlog)
    if len(zs) > 1 and len(bs) > 1:
        zs, wz = axis(ch.sigma_z, max(dlog, 0.2))
        bs, wb = axis(ch.sigma_b, max(dlog, 0.2))
    return [(z, b, a * c) for z, a in zip(zs, wz) for b, c in zip(bs, wb)]


# ----------------------------------------------------------------------------
# mechanistic population simulator
# ----------------------------------------------------------------------------
def simulate(ch, N, trials, rng, k_true=None):
    """Simulate `trials` independent experiments, each with N cells under one protocol.
    Returns k_true (trials,), counts (trials, N) of observed insertions per cell.
    Mechanism is sampled explicitly: insertions ~ Poisson(lam), detection ~ Binomial thinning."""
    K = ch.K
    if k_true is None:
        k_true = np.arange(trials) % K
    S = np.asarray(ch.protocols, float)[k_true][:, None]
    z = np.exp(ch.sigma_z * rng.standard_normal((trials, 1)) - 0.5 * ch.sigma_z ** 2) if ch.sigma_z > 0 else 1.0
    b = np.exp(ch.sigma_b * rng.standard_normal((trials, 1)) - 0.5 * ch.sigma_b ** 2) if ch.sigma_b > 0 else 1.0
    eo = np.exp(ch.sigma_o * rng.standard_normal((trials, N)) - 0.5 * ch.sigma_o ** 2) if ch.sigma_o > 0 else 1.0
    et = np.exp(ch.sigma_t * rng.standard_normal((trials, N)) - 0.5 * ch.sigma_t ** 2) if ch.sigma_t > 0 else 1.0
    h = dose_response(S * eo * z, ch.kind, ch.xi)
    lam = np.broadcast_to(b * ch.lam_bg + ch.lam_act * et * h, (trials, N))
    if ch.obs == "poisson":
        ins = rng.poisson(lam)
        det = rng.binomial(ins, ch.p_d)
        if ch.mu_c > 0:
            c = ch.clone_size
            assert N % c == 0
            nc = rng.poisson(ch.mu_c / 1.0, size=(trials, N // c))
            nc_det = nc  # mu_c is already an *observed* mean
            det = det + np.repeat(nc_det, c, axis=1)
        return k_true, det
    # binomial / site-capped
    ins = rng.poisson(np.broadcast_to(lam[..., None], (trials, N, ch.M)))
    hit = ins > 0
    seen = hit & (rng.random(hit.shape) < ch.p_d)
    return k_true, seen.sum(-1)


# ----------------------------------------------------------------------------
# optimal (Bayes) decoders
# ----------------------------------------------------------------------------
def population_loglik(ch, counts, decoder="optimal", nq=15):
    """log-likelihood (trials, K) of the observed counts under each protocol.
    decoder='optimal': knows clone membership (contiguous blocks of clone_size cells).
    decoder='naive'  : treats cells as independent (ignores clonal sharing)."""
    T, N = counts.shape
    K = ch.K
    pairs = shared_nodes(ch, N)
    clonal = ch.mu_c > 0 and ch.obs == "poisson" and ch.clone_size > 1
    out = []
    logw = []
    for z, b, w in pairs:
        if clonal and decoder == "optimal":
            La = lik_table(ch, z, b, nq)
            nmaxL = La.shape[1]
            logL = np.log(La + 1e-300)
            c = ch.clone_size
            C = N // c
            cc = counts.reshape(T, C, c)
            mmax = int(min(cc.min(axis=2).max(), ch.mu_c + 10 * np.sqrt(ch.mu_c) + 20))
            terms = []
            for m in range(mmax + 1):
                idx = cc - m
                ok = (idx >= 0).all(axis=2)
                idx_c = np.clip(idx, 0, None)
                idx_c = np.minimum(idx_c, nmaxL - 1)
                lp = logL[:, idx_c]                     # (K,T,C,c)
                ll = lp.sum(axis=3) + np.log(pois_pmf(m, ch.mu_c) + 1e-300)
                ll = np.where(ok[None], ll, -np.inf)
                terms.append(ll)
            ll = logsumexp(np.stack(terms), axis=0).sum(axis=2)   # (K,T)
        else:
            Ls = single_cell_table(ch, z, b, nq)
            idx = np.minimum(counts, Ls.shape[1] - 1)
            logL = np.log(Ls + 1e-300)
            ll = logL[:, idx].sum(axis=2)                          # (K,T)
        out.append(ll.T)
        logw.append(np.log(w))
    out = np.stack(out)                                            # (P,T,K)
    return logsumexp(out + np.array(logw)[:, None, None], axis=0)


def mc_mutual_information(ch, N, trials=600, seed=0, decoder="optimal", nq=15):
    """Monte Carlo estimate of I(protocol; N-cell insertion data) in bits using the
    known-likelihood Bayes posterior (unbiased for decoder='optimal'; a valid lower
    bound for any other decoder).  Returns (bits, standard error)."""
    rng = np.random.default_rng(seed)
    k, counts = simulate(ch, N, trials, rng)
    ll = population_loglik(ch, counts, decoder, nq)
    lp = ll - logsumexp(ll, axis=1, keepdims=True)               # uniform prior cancels
    lp_true = lp[np.arange(trials), k] / np.log(2)
    vals = np.log2(ch.K) + lp_true
    mean, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(trials))
    if decoder == "optimal" and mean < -5 * se - 0.05:
        # the exact-likelihood posterior cannot have negative expected information
        raise RuntimeError(f"negative MI {mean:.3f}+-{se:.3f}: likelihood integration is inaccurate")
    return mean, se
