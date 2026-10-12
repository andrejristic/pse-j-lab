"""Numerical free probability: predict the spectrum of a product of layers
from the measured spectra of the individual layers.

The chain, all in truncated formal power series:

    moments m_k  ->  psi(z) = sum_k m_k z^k
                 ->  chi = psi^{-1}   (compositional inverse)
                 ->  S(z) = chi(z)(1+z)/z
    S_product = prod_l S_l                     (free multiplication)
    S -> chi = z S(z)/(1+z) -> psi = chi^{-1} -> moments of the product

Freeness is an *assumption* here, not a theorem about a particular trained
network: the layer matrices of a real net after training need not be free.
That is precisely why the labs report predicted-vs-measured side by side
instead of plotting the prediction alone.
"""

from __future__ import annotations

import numpy as np

# A series is a plain list/array of coefficients [c0, c1, c2, ...] for
# c0 + c1 z + c2 z^2 + ...


def _mul(a, b, K: int) -> np.ndarray:
    """Truncated product of two power series, keeping orders 0..K."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros(K + 1)
    for i in range(min(len(a), K + 1)):
        if a[i] == 0.0:
            continue
        jmax = min(len(b), K + 1 - i)
        out[i:i + jmax] += a[i] * b[:jmax]
    return out


def _pow(a, k: int, K: int) -> np.ndarray:
    out = np.zeros(K + 1)
    out[0] = 1.0
    for _ in range(k):
        out = _mul(out, a, K)
    return out


def moments_from_spectrum(eigs, k_max: int = 8) -> list:
    """m_k = (1/N) sum lambda_i^k of an empirical (eigenvalue) spectrum."""
    e = np.asarray(eigs, dtype=np.float64)
    return [float((e ** k).mean()) for k in range(1, k_max + 1)]


def psi_series(moments) -> np.ndarray:
    """psi(z) = sum_{k>=1} m_k z^k, returned with a leading zero coefficient."""
    m = np.asarray(moments, dtype=np.float64)
    return np.concatenate([[0.0], m])


def compositional_inverse(f, K: int) -> np.ndarray:
    """Inverse under composition of a series f with f[0]=0, f[1]!=0.

    Solves g(f(z)) = z order by order.  This is the workhorse: both psi -> chi
    and chi -> psi are compositional inversions.
    """
    f = np.asarray(f, dtype=np.float64)
    if len(f) < 2 or f[0] != 0.0:
        raise ValueError("series must have zero constant term")
    if f[1] == 0.0:
        raise ValueError("series must have non-zero linear term to be invertible")
    g = np.zeros(K + 1)
    g[1] = 1.0 / f[1]
    for n in range(2, K + 1):
        # coefficient of z^n in f(g(z)) must vanish; g_n enters only via f_1 g_n
        acc = 0.0
        for k in range(2, n + 1):
            if k >= len(f) or f[k] == 0.0:
                continue
            gk = _pow(g, k, n)          # g_n does not affect orders < n of g^k, k>=2
            acc += f[k] * gk[n]
        g[n] = -acc / f[1]
    return g


def s_transform(moments, K: int = 8) -> np.ndarray:
    """S(z) = chi(z) (1+z) / z, a series in z starting at order 0."""
    psi = psi_series(moments)
    chi = compositional_inverse(psi, K + 1)
    chi_over_z = chi[1:K + 2]                       # chi = a1 z + a2 z^2 + ...
    one_plus_z = np.zeros(K + 1)
    one_plus_z[0] = 1.0
    if K >= 1:
        one_plus_z[1] = 1.0
    return _mul(chi_over_z[:K + 1], one_plus_z, K)


def free_multiply(*s_series, K: int | None = None) -> np.ndarray:
    """S of a free product = product of the S transforms."""
    series = [np.asarray(s, dtype=np.float64) for s in s_series]
    if K is None:
        K = min(len(s) for s in series) - 1
    out = np.zeros(K + 1)
    out[0] = 1.0
    for s in series:
        out = _mul(out, s[:K + 1], K)
    return out


def moments_from_s(S, K: int = 8) -> list:
    """Invert S back to moments: chi = z S/(1+z), psi = chi^{-1}, read m_k."""
    S = np.asarray(S, dtype=np.float64)
    inv_1pz = np.array([(-1.0) ** i for i in range(K + 2)])     # 1/(1+z)
    chi_over_z = _mul(S[:K + 2], inv_1pz, K + 1)
    chi = np.concatenate([[0.0], chi_over_z])[:K + 2]
    psi = compositional_inverse(chi, K + 1)
    return [float(psi[k]) for k in range(1, K + 1)]


def predict_product_moments(spectra, k_max: int = 6) -> dict:
    """Given the eigen-spectra of each factor, predict the product's moments.

    `spectra` is a list of eigenvalue arrays -- for a Jacobian J = prod A_l one
    passes the spectra of A_l A_l^T.  The returned moments are those of the
    product's J J^T under the freeness assumption.
    """
    K = k_max + 2
    ss = []
    for e in spectra:
        m = moments_from_spectrum(e, K)
        if m[0] <= 0:
            raise ValueError("spectrum with non-positive first moment")
        ss.append(s_transform(m, K))
    S = free_multiply(*ss, K=K)
    pred = moments_from_s(S, K)
    return {
        "k_max": k_max,
        "predicted_moments": pred[:k_max],
        "n_factors": len(spectra),
    }


def moment_report(predicted, measured) -> dict:
    """Side-by-side with relative errors -- the only honest way to show this."""
    p = np.asarray(predicted, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)[:p.size]
    rel = np.where(np.abs(m) > 0, np.abs(p - m) / np.abs(m), np.inf)
    return {
        "k": list(range(1, p.size + 1)),
        "predicted": p.tolist(),
        "measured": m.tolist(),
        "rel_error": rel.tolist(),
        "max_rel_error": float(np.max(rel)) if rel.size else None,
    }
