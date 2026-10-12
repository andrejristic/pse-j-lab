"""Spectral descriptors of a Jacobian.

The quantities here are the ones the deep-learning literature actually argues
about: chi (mean squared singular value, the criticality order parameter),
the max/mean ratio (which is what "dynamical isometry" is really about), and
rank-like summaries that survive on ill-conditioned matrices.
"""

from __future__ import annotations

import numpy as np


def singular_values(J: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)


def chi_statistic(J: np.ndarray) -> float:
    """chi = mean of squared singular values = ||J||_F^2 / min(shape).

    chi = 1 is the edge of chaos for a square Jacobian: signals neither
    vanish nor explode in the mean.  Depth-independence of chi is necessary
    for trainability, but *not* sufficient -- see max/mean below.
    """
    J = np.asarray(J, dtype=np.float64)
    m = min(J.shape)
    return float((J * J).sum() / m)


def effective_rank(s: np.ndarray, eps: float = 1e-12) -> float:
    """exp(H(p)) with p = s / sum(s)  (Roy & Vetterli).  Continuous rank."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    h = -(p * np.log(p)).sum()
    return float(np.exp(h))


def participation_ratio(s: np.ndarray) -> float:
    """(sum s^2)^2 / sum s^4 -- how many singular directions carry the energy."""
    s2 = np.asarray(s, dtype=np.float64) ** 2
    denom = (s2 ** 2).sum()
    if denom <= 0:
        return 0.0
    return float((s2.sum() ** 2) / denom)


def stable_rank(s: np.ndarray) -> float:
    """||J||_F^2 / ||J||_2^2 -- lower bound on rank, robust to tiny values."""
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] <= 0:
        return 0.0
    return float((s ** 2).sum() / (s.max() ** 2))


def spectrum_summary(J: np.ndarray | None = None, s: np.ndarray | None = None,
                     shape: tuple | None = None) -> dict:
    """One dict with everything the labs report about a Jacobian spectrum."""
    if s is None:
        if J is None:
            raise ValueError("need J or s")
        s = singular_values(J)
        shape = np.asarray(J).shape
    s = np.asarray(s, dtype=np.float64)
    s_sorted = np.sort(s)[::-1]
    nz = s_sorted[s_sorted > 0]
    smax = float(s_sorted[0]) if s_sorted.size else 0.0
    smin = float(s_sorted[-1]) if s_sorted.size else 0.0
    smean = float(s_sorted.mean()) if s_sorted.size else 0.0
    chi = float((s_sorted ** 2).mean()) if s_sorted.size else 0.0
    return {
        "shape": list(shape) if shape else None,
        "n_sv": int(s_sorted.size),
        "s_max": smax,
        "s_min": smin,
        "s_mean": smean,
        "s_median": float(np.median(s_sorted)) if s_sorted.size else 0.0,
        "chi": chi,
        "max_over_mean": float(smax / smean) if smean > 0 else float("inf"),
        "condition": float(smax / smin) if smin > 0 else float("inf"),
        "log_condition": (float(np.log10(smax / smin)) if smin > 0 else None),
        "effective_rank": effective_rank(s_sorted),
        "participation_ratio": participation_ratio(s_sorted),
        "stable_rank": stable_rank(s_sorted),
        "frobenius": float(np.sqrt((s_sorted ** 2).sum())),
        "log_mean_sv": float(np.log(nz).mean()) if nz.size else None,
        "log_std_sv": float(np.log(nz).std()) if nz.size > 1 else None,
    }


def isometry_gap(s: np.ndarray) -> float:
    """RMS distance of the spectrum from 1.  Zero iff J is exactly orthogonal.

    This is the honest scalar for "dynamical isometry": chi=1 can hold while
    the spectrum is wildly spread, which is exactly the ReLU/Gaussian failure.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return float("inf")
    return float(np.sqrt(((s - 1.0) ** 2).mean()))
