"""Lipschitz quantities: the bound everyone cites and the number it fails to
predict.

The product-of-spectral-norms bound is what papers use because it is the only
thing that is cheap and provable.  It is also, on trained networks, loose by
orders of magnitude.  The point of keeping both here is to *measure the gap*,
which is a fact about the network, not a footnote.
"""

from __future__ import annotations

import numpy as np

from .jacobian import input_jacobian, jvp, vjp
from .nets import ACTIVATIONS, MLP

# sup |phi'| for the supported activations; None means "computed, not assumed"
_SUP_DPHI = {
    "tanh": 1.0, "relu": 1.0, "erf": 2.0 / np.sqrt(np.pi), "sin": 1.0,
    "softplus": 1.0, "hardtanh": 1.0, "identity": 1.0,
}


def spectral_norm(M: np.ndarray, iters: int = 200, tol: float = 1e-12,
                  seed: int = 0) -> float:
    """Largest singular value by power iteration on M^T M (no full SVD)."""
    M = np.asarray(M, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1])
    v /= np.linalg.norm(v)
    s_old = 0.0
    for _ in range(iters):
        u = M @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = M.T @ u
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
        if abs(s - s_old) < tol * max(s, 1.0):
            break
        s_old = s
    return float(s)


def spectral_norm_matrix_free(net: MLP, x: np.ndarray, iters: int = 100,
                              tol: float = 1e-10, seed: int = 0) -> float:
    """||J(x)||_2 using only jvp/vjp -- never forms J.

    This is how you get the local Lipschitz constant for a net whose Jacobian
    would not fit in RAM.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(net.dims[0])
    v /= np.linalg.norm(v)
    s_old = 0.0
    s = 0.0
    for _ in range(iters):
        u = jvp(net, x, v)
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = vjp(net, x, u)
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
        if abs(s - s_old) < tol * max(s, 1.0):
            break
        s_old = s
    return float(s)


def product_norm_bound(net: MLP) -> dict:
    """L <= prod_l ||W_l||_2 * (sup|phi'|)^(#nonlinear layers).

    Global, data-independent, provable, and usually enormous.
    """
    norms = [spectral_norm(W) for W in net.W]
    n_nonlin = net.depth - (1 if net.readout_linear else 0)
    sup = _SUP_DPHI.get(net.act, 1.0)
    bound = float(np.prod(norms) * (sup ** n_nonlin))
    return {
        "layer_spectral_norms": [float(x) for x in norms],
        "sup_dphi": float(sup),
        "n_nonlinear_layers": int(n_nonlin),
        "bound": bound,
        "log10_bound": float(np.log10(bound)) if bound > 0 else None,
    }


def empirical_lipschitz(net: MLP, X: np.ndarray, matrix_free: bool = False,
                        seed: int = 0) -> dict:
    """max_x ||J(x)||_2 over the given points, plus the distribution.

    A lower bound on the true Lipschitz constant (the max over the whole
    domain can only be larger), which is exactly why the gap to
    product_norm_bound is informative in both directions.
    """
    X = np.atleast_2d(np.asarray(X, dtype=net.dtype))
    vals = []
    for i in range(X.shape[0]):
        if matrix_free:
            vals.append(spectral_norm_matrix_free(net, X[i], seed=seed + i))
        else:
            vals.append(spectral_norm(input_jacobian(net, X[i]), seed=seed + i))
    v = np.asarray(vals)
    return {
        "n": int(v.size),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "p95": float(np.quantile(v, 0.95)),
        "values": [float(x) for x in v],
    }


def looseness(net: MLP, X: np.ndarray) -> dict:
    """How many orders of magnitude the certified bound overshoots reality."""
    b = product_norm_bound(net)
    e = empirical_lipschitz(net, X)
    ratio = b["bound"] / e["max"] if e["max"] > 0 else float("inf")
    return {
        "bound": b["bound"],
        "empirical_max": e["max"],
        "ratio": float(ratio),
        "log10_ratio": float(np.log10(ratio)) if np.isfinite(ratio) and ratio > 0 else None,
    }
