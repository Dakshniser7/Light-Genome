"""Two-colour multiplexed recorder: colour j drives its own insertion site-set j
(guide set / barcode).  Protocols are (colour1, colour2) in {0,1}^2.

Observation per cell is (n1, n2) = insertions detected at site-set 1 and 2.
Compare a SITE-RESOLVED readout (n1, n2) with a POOLED readout (n1 + n2), which is what a
random-insertion recorder without site identity would give.
chi = optical/spectral cross-talk: colour j excites opsin j' != j with relative efficiency chi.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.special import gammaln, logsumexp
from model import gh_nodes, dose_response


@dataclass(frozen=True)
class TwoColour:
    S: float = 3.0
    xi: float = 1.0
    lam_bg: float = 0.05     # per site-set
    lam_act: float = 3.0     # peak per site-set
    p_d: float = 0.3
    chi: float = 0.0
    sigma_o: float = 0.5     # independent per colour per cell
    sigma_t: float = 0.5     # shared by both site-sets in a cell
    kind: str = "saturating"

    def but(self, **kw):
        return replace(self, **kw)


PROTOCOLS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)


def _rates(ch, o1, o2, t):
    a, b = PROTOCOLS[:, 0][:, None], PROTOCOLS[:, 1][:, None]
    x1 = ch.S * o1[None] * (a + ch.chi * b)
    x2 = ch.S * o2[None] * (b + ch.chi * a)
    l1 = ch.lam_bg + ch.lam_act * t[None] * dose_response(x1, ch.kind, ch.xi)
    l2 = ch.lam_bg + ch.lam_act * t[None] * dose_response(x2, ch.kind, ch.xi)
    return l1, l2


def _pois(n, mean):
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = n * np.log(mean[..., None]) - mean[..., None] - gammaln(n + 1)
    lp = np.where(mean[..., None] == 0, np.where(n == 0, 0.0, -np.inf), lp)
    return np.exp(lp)


def tables(ch, nq=9, nmax=40):
    """L2[k, n1, n2] and pooled Lp[k, n]."""
    xo, wo = gh_nodes(nq, ch.sigma_o)
    xt, wt = gh_nodes(nq, ch.sigma_t)
    O1, O2, T = np.meshgrid(xo, xo, xt, indexing="ij")
    W = (wo[:, None, None] * wo[None, :, None] * wt[None, None, :]).ravel()
    l1, l2 = _rates(ch, O1.ravel(), O2.ravel(), T.ravel())
    n = np.arange(nmax + 1)
    P1 = _pois(n, ch.p_d * l1)     # (k,q,n)
    P2 = _pois(n, ch.p_d * l2)
    L2 = np.einsum("q,kqa,kqb->kab", W, P1, P2)
    assert 1 - L2.sum((1, 2)).min() < 1e-6, "increase nmax"
    Lp = np.zeros((4, 2 * nmax + 1))
    for a in range(nmax + 1):
        for b in range(nmax + 1):
            Lp[:, a + b] += L2[:, a, b]
    return L2, Lp


def mi_exact(L):
    from infotheory import mi_from_channel
    return mi_from_channel(L.reshape(L.shape[0], -1))


def simulate(ch, N, trials, rng):
    k = np.arange(trials) % 4
    a = PROTOCOLS[k][:, 0][:, None]
    b = PROTOCOLS[k][:, 1][:, None]
    sh = (trials, N)
    o1 = np.exp(ch.sigma_o * rng.standard_normal(sh) - 0.5 * ch.sigma_o ** 2)
    o2 = np.exp(ch.sigma_o * rng.standard_normal(sh) - 0.5 * ch.sigma_o ** 2)
    t = np.exp(ch.sigma_t * rng.standard_normal(sh) - 0.5 * ch.sigma_t ** 2)
    x1 = ch.S * o1 * (a + ch.chi * b)
    x2 = ch.S * o2 * (b + ch.chi * a)
    l1 = ch.lam_bg + ch.lam_act * t * dose_response(x1, ch.kind, ch.xi)
    l2 = ch.lam_bg + ch.lam_act * t * dose_response(x2, ch.kind, ch.xi)
    n1 = rng.binomial(rng.poisson(l1), ch.p_d)
    n2 = rng.binomial(rng.poisson(l2), ch.p_d)
    return k, n1, n2


def mc_mi(ch, N, trials=600, seed=0, readout="resolved"):
    rng = np.random.default_rng(seed)
    L2, Lp = tables(ch)
    k, n1, n2 = simulate(ch, N, trials, rng)
    if readout == "resolved":
        logL = np.log(L2 + 1e-300)
        ll = logL[:, n1, n2].sum(axis=2).T
    else:
        logL = np.log(Lp + 1e-300)
        ll = logL[:, n1 + n2].sum(axis=2).T
    lp = ll - logsumexp(ll, axis=1, keepdims=True)
    v = 2.0 + lp[np.arange(trials), k] / np.log(2)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(trials))
