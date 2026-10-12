"""L1 -- dynamical isometry: what the spectrum of J does as depth grows.

The claim under test is the one that made this literature useful: chi = 1
(criticality) keeps the *mean* squared singular value from exploding, but only
orthogonal weights with a bounded activation keep the whole *spectrum* near 1.
Gaussian weights at exactly chi = 1 still produce a Jacobian whose condition
number grows exponentially with depth, which is why very deep tanh networks
train from orthogonal init and not from Gaussian.

Mean-field predictions here are computed with Gauss-Hermite quadrature, not
copied from a paper, so the comparison is between two things this machine
computed.
"""

from __future__ import annotations

import numpy as np

from ..jac.jacobian import input_jacobian
from ..jac.nets import ACTIVATIONS, MLP
from ..jac.spectra import isometry_gap, spectrum_summary
from ..rmt.stransform import (moment_report, moments_from_spectrum,
                              predict_product_moments)

# --- mean field ---------------------------------------------------------------

def _gauss_hermite(f, q: float, order: int = 101) -> float:
    """E[f(z)] for z ~ N(0, q) via Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(order)
    z = np.sqrt(2.0 * max(q, 0.0)) * x
    return float((w * f(z)).sum() / np.sqrt(np.pi))


def variance_fixed_point(act: str, sigma_w: float, sigma_b: float,
                         iters: int = 200, q0: float = 1.0) -> float:
    """q* solving q = sigma_w^2 E_{z~N(0,q)}[phi(z)^2] + sigma_b^2."""
    phi, _ = ACTIVATIONS[act]
    q = q0
    for _ in range(iters):
        q_new = sigma_w ** 2 * _gauss_hermite(lambda z: phi(z) ** 2, q) + sigma_b ** 2
        if abs(q_new - q) < 1e-14:
            q = q_new
            break
        q = q_new
    return float(q)


def chi_mean_field(act: str, sigma_w: float, sigma_b: float) -> dict:
    """chi_1 = sigma_w^2 E[phi'(z)^2] at the variance fixed point."""
    _, dphi = ACTIVATIONS[act]
    q = variance_fixed_point(act, sigma_w, sigma_b)
    chi1 = sigma_w ** 2 * _gauss_hermite(lambda z: dphi(z) ** 2, q)
    return {"q_star": q, "chi_1": float(chi1), "sigma_w": sigma_w,
            "sigma_b": sigma_b, "act": act}


# --- experiments --------------------------------------------------------------

def depth_sweep(depths=(2, 4, 8, 16, 32, 64), width=64, act="tanh",
                init="gaussian", sigma_w=1.0, sigma_b=0.0, n_probe=8,
                seed=0, progress=None) -> dict:
    """Spectrum descriptors of J as a function of depth, at fixed width."""
    rng = np.random.default_rng(seed)
    rows = []
    for di, L in enumerate(depths):
        net = MLP(dims=[width] * (L + 1), act=act, init=init, sigma_w=sigma_w,
                  sigma_b=sigma_b, readout_linear=False, seed=seed + 1000 * di)
        q = variance_fixed_point(act, sigma_w, sigma_b)
        X = rng.standard_normal((n_probe, width)) * np.sqrt(max(q, 1e-12))
        stats, gaps = [], []
        for i in range(n_probe):
            J = input_jacobian(net, X[i])
            s = np.linalg.svd(J, compute_uv=False)
            stats.append(spectrum_summary(s=s, shape=J.shape))
            gaps.append(isometry_gap(s))
        row = {
            "depth": int(L),
            "chi": float(np.mean([s["chi"] for s in stats])),
            "s_max": float(np.mean([s["s_max"] for s in stats])),
            "s_mean": float(np.mean([s["s_mean"] for s in stats])),
            "max_over_mean": float(np.mean([s["max_over_mean"] for s in stats])),
            "log10_condition": float(np.mean([
                s["log_condition"] for s in stats if s["log_condition"] is not None])
                if any(s["log_condition"] is not None for s in stats) else np.inf),
            "isometry_gap": float(np.mean(gaps)),
            "effective_rank": float(np.mean([s["effective_rank"] for s in stats])),
            "log_std_sv": float(np.mean([s["log_std_sv"] for s in stats
                                         if s["log_std_sv"] is not None]))
            if any(s["log_std_sv"] is not None for s in stats) else None,
        }
        rows.append(row)
        if progress:
            progress((di + 1) / len(depths), f"depth {L}")
    mf = chi_mean_field(act, sigma_w, sigma_b)
    return {
        "kind": "l1_depth_sweep",
        "config": {"width": width, "act": act, "init": init, "sigma_w": sigma_w,
                   "sigma_b": sigma_b, "n_probe": n_probe, "seed": seed},
        "mean_field": mf,
        "series": rows,
        "predicted_chi_by_depth": [
            {"depth": r["depth"], "chi_pred": float(mf["chi_1"] ** r["depth"])}
            for r in rows],
        "claim": ("chi controls the mean; only the isometry gap and the condition "
                  "number see the difference between orthogonal and Gaussian"),
    }


def chi_critical(sigma_ws=None, act="tanh", sigma_b=0.05, width=64, depth=1,
                 n_probe=16, seed=0, progress=None) -> dict:
    """Locate the chi = 1 critical line and check it against measurement."""
    if sigma_ws is None:
        sigma_ws = list(np.round(np.linspace(0.5, 2.5, 21), 4))
    rows = []
    for i, sw in enumerate(sigma_ws):
        mf = chi_mean_field(act, float(sw), sigma_b)
        rng = np.random.default_rng(seed + i)
        net = MLP(dims=[width] * (depth + 1), act=act, init="gaussian",
                  sigma_w=float(sw), sigma_b=sigma_b, readout_linear=False,
                  seed=seed + 100 * i)
        X = rng.standard_normal((n_probe, width)) * np.sqrt(max(mf["q_star"], 1e-12))
        chis = []
        for j in range(n_probe):
            J = input_jacobian(net, X[j])
            chis.append(float((J * J).sum() / min(J.shape)))
        rows.append({
            "sigma_w": float(sw),
            "q_star": mf["q_star"],
            "chi_1_theory": mf["chi_1"],
            "chi_theory_depth": float(mf["chi_1"] ** depth),
            "chi_measured": float(np.mean(chis)),
            "chi_measured_std": float(np.std(chis)),
        })
        if progress:
            progress((i + 1) / len(sigma_ws), f"sigma_w={sw}")
    # critical sigma_w by linear interpolation of chi_1 - 1
    crit = None
    for a, b in zip(rows, rows[1:]):
        if (a["chi_1_theory"] - 1.0) * (b["chi_1_theory"] - 1.0) <= 0:
            fa, fb = a["chi_1_theory"] - 1.0, b["chi_1_theory"] - 1.0
            if fb != fa:
                crit = a["sigma_w"] + (b["sigma_w"] - a["sigma_w"]) * (-fa) / (fb - fa)
            break
    err = [abs(r["chi_measured"] - r["chi_theory_depth"]) /
           max(r["chi_theory_depth"], 1e-12) for r in rows]
    return {
        "kind": "l1_chi_critical",
        "config": {"act": act, "sigma_b": sigma_b, "width": width,
                   "depth": depth, "n_probe": n_probe, "seed": seed},
        "series": rows,
        "sigma_w_critical": crit,
        "max_rel_error_theory_vs_measured": float(max(err)),
        "claim": "mean-field chi_1 predicts the measured mean squared singular value",
    }


def free_probability(depth=6, width=200, act="tanh", init="gaussian",
                     sigma_w=1.0, sigma_b=0.0, k_max=5, n_probe=4, seed=0,
                     progress=None) -> dict:
    """Predict the moments of J J^T by free multiplication of the layer spectra.

    Freeness is an assumption; the report shows how well it holds at this
    width rather than asserting that it does.
    """
    from ..jac.nets import ACTIVATIONS as _A
    _, dphi = _A[act]
    rng = np.random.default_rng(seed)
    q = variance_fixed_point(act, sigma_w, sigma_b)
    net = MLP(dims=[width] * (depth + 1), act=act, init=init, sigma_w=sigma_w,
              sigma_b=sigma_b, readout_linear=False, seed=seed)
    reports = []
    for p in range(n_probe):
        x = rng.standard_normal(width) * np.sqrt(max(q, 1e-12))
        _, c = net.forward(x, cache=True)
        factor_spectra = []
        for l in range(depth):
            d = dphi(c["z"][l][0])
            A = d[:, None] * net.W[l]                 # D_l W_l
            s = np.linalg.svd(A, compute_uv=False)
            factor_spectra.append(s ** 2)
        pred = predict_product_moments(factor_spectra, k_max=k_max)["predicted_moments"]
        J = input_jacobian(net, x)
        sJ = np.linalg.svd(J, compute_uv=False)
        meas = moments_from_spectrum(sJ ** 2, k_max)
        reports.append(moment_report(pred, meas))
        if progress:
            progress((p + 1) / n_probe, f"probe {p+1}")
    avg_pred = np.mean([r["predicted"] for r in reports], axis=0)
    avg_meas = np.mean([r["measured"] for r in reports], axis=0)
    return {
        "kind": "l1_free_probability",
        "config": {"depth": depth, "width": width, "act": act, "init": init,
                   "sigma_w": sigma_w, "sigma_b": sigma_b, "k_max": k_max,
                   "n_probe": n_probe, "seed": seed},
        "per_probe": reports,
        "averaged": moment_report(avg_pred, avg_meas),
        "claim": ("free multiplicative convolution of the measured layer spectra "
                  "reproduces the moments of J J^T"),
    }


def trainability(depths=(4, 16, 32), inits=("orthogonal", "gaussian"),
                 width=64, act="tanh", sigma_w=1.0, n_train=1000, steps=300,
                 lr=0.02, batch=32, dataset="mnist", seed=0, progress=None) -> dict:
    """Does the isometric initialisation actually train deeper nets faster?

    Small by design: 1000 MNIST images, a few hundred steps.  The comparison
    is between initialisations under an identical budget, not an attempt at
    state of the art.
    """
    from ..data.idx import load_dataset, subset
    from ..train.trainer import TrainConfig, Trainer

    d = load_dataset(dataset, "train", n=max(n_train * 4, 4000), dtype=np.float64)
    d = subset(d, n_train, seed=seed)
    X, y = d["X"], d["y"]
    total = len(depths) * len(inits)
    rows, k = [], 0
    for L in depths:
        for init in inits:
            net = MLP(dims=[X.shape[1]] + [width] * L + [10], act=act, init=init,
                      sigma_w=sigma_w, sigma_b=0.0, readout_linear=True,
                      seed=seed + 17 * L)
            J0 = input_jacobian(net, X[0])
            s0 = np.linalg.svd(J0, compute_uv=False)
            epochs = max(1, int(steps * batch / max(len(X), 1)))
            cfg = TrainConfig(epochs=epochs, batch_size=batch, lr=lr, opt="sgd",
                              seed=seed, probe_n=4, eval_every=max(1, epochs // 5))
            res = Trainer(net, cfg).fit(X, y)
            rows.append({
                "depth": int(L), "init": init,
                "init_isometry_gap": float(isometry_gap(s0)),
                "init_log10_condition": (float(np.log10(s0[0] / s0[-1]))
                                          if s0[-1] > 0 else None),
                "epochs": epochs,
                "final_train_loss": res.final["train"]["loss"],
                "final_train_acc": res.final["train"]["acc"],
                "loss_curve": [h["train_loss"] for h in res.history],
                "seconds": res.seconds,
            })
            k += 1
            if progress:
                progress(k / total, f"depth {L} / {init}")
    return {
        "kind": "l1_trainability",
        "config": {"width": width, "act": act, "sigma_w": sigma_w,
                   "n_train": n_train, "steps": steps, "lr": lr, "batch": batch,
                   "dataset": dataset, "seed": seed},
        "series": rows,
        "claim": ("under an identical budget, isometric init trains deeper nets "
                  "further than Gaussian init"),
    }
