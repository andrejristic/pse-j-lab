"""Synthetic manifolds with *known* intrinsic dimension.

Lab L4 estimates the dimension of the data manifold from the rank profile of
the Jacobian.  Any such estimator can be made to produce a plausible number;
these generators exist so that the number can be checked against an answer
that was fixed before the estimator ran.
"""

from __future__ import annotations

import numpy as np


def _embed(Z: np.ndarray, ambient: int, seed: int, noise: float) -> np.ndarray:
    """Rotate a low-dim cloud into `ambient` dimensions and add ambient noise."""
    rng = np.random.default_rng(seed + 7919)
    d = Z.shape[1]
    if ambient < d:
        raise ValueError("ambient dimension smaller than intrinsic")
    A = rng.standard_normal((ambient, ambient))
    Q, _ = np.linalg.qr(A)
    X = Z @ Q[:d, :]                        # isometric embedding into R^ambient
    if noise > 0:
        X = X + noise * rng.standard_normal(X.shape)
    return X


def sphere(n: int = 1000, dim: int = 2, ambient: int = 10, noise: float = 0.0,
           seed: int = 0) -> dict:
    """Uniform points on the `dim`-sphere S^dim embedded in R^ambient.

    Intrinsic dimension is `dim` (the sphere S^d lives in R^{d+1}).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, dim + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    X = _embed(Z, ambient, seed, noise)
    return {"X": X, "intrinsic_dim": dim, "name": f"sphere(S^{dim})",
            "ambient": ambient, "noise": noise}


def swiss_roll(n: int = 1000, ambient: int = 10, noise: float = 0.0,
               seed: int = 0) -> dict:
    """The classic 2-D sheet curled in 3-D, then embedded higher."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21.0 * rng.random(n)
    Z = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1) / 10.0
    X = _embed(Z, ambient, seed, noise)
    return {"X": X, "intrinsic_dim": 2, "name": "swiss_roll",
            "ambient": ambient, "noise": noise}


def linear_subspace(n: int = 1000, dim: int = 3, ambient: int = 20,
                    noise: float = 0.0, seed: int = 0) -> dict:
    """Gaussian cloud on a `dim`-dimensional linear subspace -- the easy case.

    If an estimator cannot get this one right, it does not get to be trusted
    on anything curved.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, dim))
    X = _embed(Z, ambient, seed, noise)
    return {"X": X, "intrinsic_dim": dim, "name": f"linear({dim})",
            "ambient": ambient, "noise": noise}


def torus(n: int = 1000, ambient: int = 10, R: float = 2.0, r: float = 0.6,
          noise: float = 0.0, seed: int = 0) -> dict:
    """2-torus in R^3, embedded higher.  Intrinsic dimension 2."""
    rng = np.random.default_rng(seed)
    u = 2 * np.pi * rng.random(n)
    v = 2 * np.pi * rng.random(n)
    Z = np.stack([(R + r * np.cos(v)) * np.cos(u),
                  (R + r * np.cos(v)) * np.sin(u),
                  r * np.sin(v)], axis=1)
    X = _embed(Z, ambient, seed, noise)
    return {"X": X, "intrinsic_dim": 2, "name": "torus",
            "ambient": ambient, "noise": noise}


def product_manifold(n: int = 1000, dims=(1, 2), ambient: int = 12,
                     noise: float = 0.0, seed: int = 0) -> dict:
    """Product of spheres, e.g. S^1 x S^2 -- intrinsic dim = sum(dims).

    Included because products are where naive PCA-style estimators quietly
    overestimate.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for d in dims:
        Z = rng.standard_normal((n, d + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        parts.append(Z)
    Z = np.concatenate(parts, axis=1)
    X = _embed(Z, ambient, seed, noise)
    return {"X": X, "intrinsic_dim": int(sum(dims)),
            "name": "x".join(f"S^{d}" for d in dims),
            "ambient": ambient, "noise": noise}


MANIFOLDS = {
    "sphere": sphere,
    "swiss_roll": swiss_roll,
    "linear": linear_subspace,
    "torus": torus,
    "product": product_manifold,
}
