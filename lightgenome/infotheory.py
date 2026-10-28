"""Information-theoretic helpers: exact MI, Blahut-Arimoto capacity, sample estimators."""
import numpy as np


def mi_from_channel(W, prior=None):
    """Exact I(X;Y) in bits for channel W[x, y] (rows sum to 1)."""
    W = np.asarray(W, float)
    nx = W.shape[0]
    p = np.full(nx, 1.0 / nx) if prior is None else np.asarray(prior, float)
    py = p @ W
    mask = (W > 1e-290) & (py[None, :] > 1e-290)   # ignore subnormal entries (contribute ~0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = W * (np.log2(W) - np.log2(py)[None, :])
    t = np.where(mask, t, 0.0)
    return float((p[:, None] * t).sum())


def blahut_arimoto(W, tol=1e-10, maxit=20000):
    """Channel capacity (bits) and optimal input distribution for W[x, y]."""
    W = np.asarray(W, float)
    nx = W.shape[0]
    p = np.full(nx, 1.0 / nx)
    for _ in range(maxit):
        py = p @ W
        mask = (W > 1e-290) & (py[None, :] > 1e-290)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(mask, W * np.log(W / py[None, :]), 0.0).sum(1)
        pn = p * np.exp(d)
        pn /= pn.sum()
        lo = np.log(np.sum(p * np.exp(d)))
        hi = d.max()
        p = pn
        if hi - lo < tol:
            break
    return mi_from_channel(W, p), p


def entropy_plugin(counts):
    c = np.asarray(counts, float)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mi_plugin(x, y):
    """Plug-in MI (bits) from paired integer samples."""
    x = np.asarray(x); y = np.asarray(y)
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    J = np.zeros((len(xs), len(ys)))
    np.add.at(J, (xi, yi), 1)
    n = J.sum()
    px = J.sum(1, keepdims=True) / n
    py = J.sum(0, keepdims=True) / n
    pj = J / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(pj > 0, pj * np.log2(pj / (px * py)), 0.0)
    return float(t.sum()), J


def mi_miller_madow(x, y):
    """Miller-Madow bias-corrected MI: plug-in + (Kx-1)(Ky-1)/(2 n ln2) correction on nonzero bins."""
    mi, J = mi_plugin(x, y)
    n = J.sum()
    kxy = (J > 0).sum()
    kx = (J.sum(1) > 0).sum()
    ky = (J.sum(0) > 0).sum()
    # plug-in H is biased by -(K-1)/(2n ln2); MI = Hx + Hy - Hxy is therefore biased by
    # +(Kxy - Kx - Ky + 1)/(2n ln2), which we subtract.
    corr = (kxy - kx - ky + 1) / (2 * n * np.log(2))
    return mi - corr


def mi_shuffle_corrected(x, y, nshuf=50, seed=0):
    """Plug-in MI minus the mean plug-in MI of label-shuffled data."""
    rng = np.random.default_rng(seed)
    mi, _ = mi_plugin(x, y)
    nulls = [mi_plugin(rng.permutation(x), y)[0] for _ in range(nshuf)]
    return mi - float(np.mean(nulls))
