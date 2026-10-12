"""Marchenko-Pastur law and Fuss-Catalan moments -- the closed forms.

MP describes the eigenvalues of W W^T for a single Gaussian layer.  Fuss-
Catalan describes the *squared singular values* of a product of L independent
square Ginibre matrices, which is the null model for a deep linear network.
Both are exact; they are the yardsticks the numerical machinery is measured
against.
"""

from __future__ import annotations

import math

import numpy as np


def mp_edges(ratio: float, sigma2: float = 1.0) -> tuple:
    """Support [lambda_-, lambda_+] of the MP law with aspect ratio q = p/n."""
    q = float(ratio)
    a = sigma2 * (1.0 - math.sqrt(q)) ** 2
    b = sigma2 * (1.0 + math.sqrt(q)) ** 2
    return a, b


def mp_pdf(x, ratio: float, sigma2: float = 1.0):
    """Density of the Marchenko-Pastur law (continuous part).

    For q > 1 there is additionally an atom of mass 1 - 1/q at zero, which is
    not part of this density and is reported by mp_atom_at_zero().
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    q = float(ratio)
    a, b = mp_edges(q, sigma2)
    out = np.zeros_like(x)
    inside = (x > a) & (x < b)
    xi = x[inside]
    out[inside] = np.sqrt((b - xi) * (xi - a)) / (2.0 * np.pi * sigma2 * q * xi)
    return out


def mp_atom_at_zero(ratio: float) -> float:
    q = float(ratio)
    return max(0.0, 1.0 - 1.0 / q) if q > 1 else 0.0


def mp_sample_eigs(n: int, p: int, sigma2: float = 1.0, seed: int = 0) -> np.ndarray:
    """Eigenvalues of the sample covariance (1/n) X^T X, X is n x p Gaussian."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * math.sqrt(sigma2)
    S = (X.T @ X) / n
    return np.linalg.eigvalsh(S)


def esd_histogram(eigs, bins: int = 80, range_=None) -> dict:
    """Empirical spectral density as a normalised histogram."""
    e = np.asarray(eigs, dtype=np.float64)
    hist, edges = np.histogram(e, bins=bins, range=range_, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"centers": centers.tolist(), "density": hist.tolist(),
            "edges": edges.tolist(), "n": int(e.size)}


def ks_distance_to_mp(eigs, ratio: float, sigma2: float = 1.0,
                      grid: int = 2000) -> float:
    """Kolmogorov-Smirnov distance between the empirical CDF and MP's.

    Not a hypothesis test (the eigenvalues are strongly dependent, so the
    usual KS p-value does not apply) -- it is used only as a monotone
    goodness number, and is reported as such.
    """
    e = np.sort(np.asarray(eigs, dtype=np.float64))
    a, b = mp_edges(ratio, sigma2)
    xs = np.linspace(max(a, 0.0), b, grid)
    pdf = mp_pdf(xs, ratio, sigma2)
    cdf = np.cumsum(pdf) * (xs[1] - xs[0])
    atom = mp_atom_at_zero(ratio)
    cdf = atom + (1.0 - atom) * cdf / max(cdf[-1], 1e-300)
    emp = np.searchsorted(e, xs, side="right") / e.size
    return float(np.abs(emp - cdf).max())


def fuss_catalan_moments(L: int, k_max: int = 8) -> list:
    """Moments of the squared-singular-value law of a product of L Ginibres.

    m_k = FC(L, k) = binom((L+1)k, k) / (L k + 1).  L = 1 recovers the Catalan
    numbers, i.e. the moments of the Marchenko-Pastur law at q = 1.
    """
    out = []
    for k in range(1, k_max + 1):
        out.append(math.comb((L + 1) * k, k) / (L * k + 1))
    return out


def ginibre_product_eigs(L: int, N: int, seed: int = 0, normalise: bool = True) -> np.ndarray:
    """Squared singular values of a product of L independent N x N Ginibres.

    Each factor is scaled by 1/sqrt(N) so the limiting law is the standard
    Fuss-Catalan one with first moment 1.
    """
    rng = np.random.default_rng(seed)
    M = np.eye(N)
    for _ in range(L):
        G = rng.standard_normal((N, N)) / math.sqrt(N)
        M = G @ M
    s = np.linalg.svd(M, compute_uv=False)
    lam = s ** 2
    if normalise and lam.mean() > 0:
        pass  # already unit-mean in expectation; do not rescale, that would hide bias
    return lam
