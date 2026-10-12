"""L5 -- log|det J| in a coupling flow: three computations of one number.

    analytic    sum of the coupling log-scales  ("the determinant is free")
    assembled   slogdet of the exact Jacobian built from the layer blocks
    numeric     slogdet of a finite-difference Jacobian

All three must agree.  This lab is the strictest test of the whole engine:
it exercises the MLP Jacobian inside the coupling nets, the chain rule across
layers, and the determinant, and it has an answer that is known independently
of any of them.
"""

from __future__ import annotations

import numpy as np

from ..flow.coupling import Flow
from ..jac.jacobian import finite_difference_jacobian


def logdet_consistency(dims=(4, 6, 8), n_layers=4, hidden=32, act="tanh",
                       n_points=8, scale_cap=2.0, seed=0, progress=None) -> dict:
    """Analytic vs assembled vs finite-difference log|det J|, per dimension."""
    rows = []
    for di, dim in enumerate(dims):
        rng = np.random.default_rng(seed + di)
        fl = Flow(dim, n_layers=n_layers, hidden=hidden, act=act,
                  scale_cap=scale_cap, seed=seed + 7 * di)
        e_asm, e_fd, e_inv = [], [], []
        for _ in range(n_points):
            x = rng.standard_normal(dim)
            _, ld_analytic = fl.forward(x)
            J = fl.jacobian(x)
            sign, ld_assembled = np.linalg.slogdet(J)
            Jn = finite_difference_jacobian(lambda z: fl.forward(z)[0], x, eps=1e-6)
            _, ld_numeric = np.linalg.slogdet(Jn)
            z, _ = fl.forward(x)
            x_back = fl.inverse(z)
            e_asm.append(abs(ld_analytic - ld_assembled))
            e_fd.append(abs(ld_analytic - ld_numeric))
            e_inv.append(float(np.abs(x - x_back).max()))
        rows.append({
            "dim": int(dim),
            "n_layers": n_layers,
            "max_err_analytic_vs_assembled": float(max(e_asm)),
            "max_err_analytic_vs_numeric": float(max(e_fd)),
            "max_inversion_err": float(max(e_inv)),
            "n_points": n_points,
        })
        if progress:
            progress((di + 1) / len(dims), f"dim {dim}")
    return {
        "kind": "l5_logdet_consistency",
        "config": {"dims": list(dims), "n_layers": n_layers, "hidden": hidden,
                   "act": act, "n_points": n_points, "scale_cap": scale_cap,
                   "seed": seed},
        "series": rows,
        "worst_assembled": float(max(r["max_err_analytic_vs_assembled"] for r in rows)),
        "worst_numeric": float(max(r["max_err_analytic_vs_numeric"] for r in rows)),
        "worst_inversion": float(max(r["max_inversion_err"] for r in rows)),
        "claim": ("the free determinant equals the true determinant; if this "
                  "fails, no other lab's numbers can be trusted"),
    }


def change_of_variables(dim=2, n_layers=4, hidden=32, act="tanh", n_samples=20000,
                        grid=40, extent=3.0, scale_cap=1.0, seed=0,
                        progress=None) -> dict:
    """Push a Gaussian through the flow and check the density it induces.

    The analytic density comes from the change-of-variables formula; the
    empirical one from a histogram of samples.  Agreement on a grid is the
    statement that log|det J| is doing its job.
    """
    rng = np.random.default_rng(seed)
    fl = Flow(dim, n_layers=n_layers, hidden=hidden, act=act,
              scale_cap=scale_cap, seed=seed)
    if dim != 2:
        raise ValueError("the histogram check is written for dim = 2")
    Z = rng.standard_normal((n_samples, dim))
    X = np.stack([fl.inverse(Z[i]) for i in range(n_samples)])
    if progress:
        progress(0.6, "sampled; building histogram")
    edges = np.linspace(-extent, extent, grid + 1)
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=[edges, edges], density=True)
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    analytic = np.zeros_like(H)
    for i, a in enumerate(cx):
        for j, b in enumerate(cy):
            analytic[i, j] = np.exp(fl.log_prob(np.array([a, b])))
    mask = H > 0
    if mask.sum() < 10:
        rel = float("nan")
    else:
        rel = float(np.median(np.abs(analytic[mask] - H[mask]) / H[mask]))
    # total mass of the analytic density on the grid (should approach 1)
    cell = (edges[1] - edges[0]) ** 2
    mass = float(analytic.sum() * cell)
    corr = float(np.corrcoef(analytic[mask], H[mask])[0, 1]) if mask.sum() > 2 else None
    return {
        "kind": "l5_change_of_variables",
        "config": {"dim": dim, "n_layers": n_layers, "hidden": hidden, "act": act,
                   "n_samples": n_samples, "grid": grid, "extent": extent,
                   "scale_cap": scale_cap, "seed": seed},
        "median_rel_error": rel,
        "correlation": corr,
        "analytic_mass_on_grid": mass,
        "occupied_cells": int(mask.sum()),
        "grid_x": cx.tolist(),
        "grid_y": cy.tolist(),
        "analytic_density": analytic.tolist(),
        "empirical_density": H.tolist(),
        "claim": ("the density predicted by the change of variables matches the "
                  "histogram of actual samples"),
    }
