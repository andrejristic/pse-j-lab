"""L2 -- the parameter Jacobian and the neural tangent kernel.

Theta(x, x') = <df(x)/dtheta, df(x')/dtheta>.  Three questions, each cheap
enough for a 4 GB laptop if the width is kept honest:

  1. what does the NTK spectrum look like, and how aligned is it with labels
  2. how far does Theta move during training (lazy vs rich), as a function of
     width -- the prediction is that the relative drift falls like 1/sqrt(width)
  3. does the *frozen* kernel actually predict the trained network's residual

Question 3 is the one that can fail loudly, so it is the one worth running.
"""

from __future__ import annotations

import numpy as np

from ..jac.jacobian import param_jacobian
from ..jac.nets import MLP
from ..jac.spectra import effective_rank, participation_ratio


def param_jacobian_stack(net: MLP, X: np.ndarray, output: int = 0,
                         chunk: int = 64) -> np.ndarray:
    """(n, n_params) stack of df_output/dtheta rows, built in chunks."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    rows = np.empty((X.shape[0], net.n_params), dtype=np.float64)
    for i in range(0, X.shape[0], chunk):
        for j in range(i, min(i + chunk, X.shape[0])):
            rows[j] = param_jacobian(net, X[j], outputs=[output])[0]
    return rows


def ntk_gram(net: MLP, X: np.ndarray, output: int = 0,
             chunk: int = 64, max_bytes: int = 400 * 1024 * 1024) -> np.ndarray:
    n = np.atleast_2d(X).shape[0]
    need = n * net.n_params * 8
    if need > max_bytes:
        raise MemoryError(f"NTK needs a {need/2**20:.0f} MiB Jacobian stack; "
                          f"reduce n or width")
    J = param_jacobian_stack(net, X, output=output, chunk=chunk)
    return J @ J.T


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """<K, yy^T>_F / (||K||_F ||yy^T||_F) -- Cristianini's alignment in [0,1]."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    Y = y @ y.T
    num = float((K * Y).sum())
    den = float(np.linalg.norm(K) * np.linalg.norm(Y))
    return num / den if den > 0 else 0.0


# --- experiments --------------------------------------------------------------

def ntk_spectrum(n=128, width=64, depth=2, act="tanh", dataset="mnist",
                 classes=(0, 1), sigma_w=1.4, seed=0, progress=None) -> dict:
    """Eigen-spectrum of the empirical NTK and its alignment with the labels."""
    from ..data.idx import load_dataset

    d = load_dataset(dataset, "train", n=4000, dtype=np.float64)
    mask = np.isin(d["y"], list(classes))
    X, yl = d["X"][mask][:n], d["y"][mask][:n]
    y = np.where(yl == classes[0], -1.0, 1.0)
    net = MLP(dims=[X.shape[1]] + [width] * depth + [1], act=act,
              sigma_w=sigma_w, readout_linear=True, seed=seed)
    if progress:
        progress(0.2, "building parameter Jacobian")
    K = ntk_gram(net, X)
    if progress:
        progress(0.8, "eigendecomposition")
    ev = np.linalg.eigvalsh(K)[::-1]
    ev = np.clip(ev, 0.0, None)
    tot = ev.sum()
    return {
        "kind": "l2_ntk_spectrum",
        "config": {"n": int(X.shape[0]), "width": width, "depth": depth,
                   "act": act, "dataset": dataset, "classes": list(classes),
                   "sigma_w": sigma_w, "seed": seed,
                   "n_params": int(net.n_params)},
        "eigenvalues": ev.tolist(),
        "top10_share": float(ev[:10].sum() / tot) if tot > 0 else None,
        "effective_rank": effective_rank(ev),
        "participation_ratio": participation_ratio(np.sqrt(ev)),
        "condition": float(ev[0] / ev[-1]) if ev[-1] > 0 else None,
        "alignment_with_labels": kernel_alignment(K, y),
        "trace": float(np.trace(K)),
        "claim": ("the NTK spectrum is dominated by a few directions; alignment "
                  "with the labels is what makes those directions useful"),
    }


def lazy_rich(widths=(16, 32, 64, 128, 256), n=64, depth=2, act="tanh",
              steps=200, lr=0.5, dataset="mnist", classes=(0, 1),
              sigma_w=1.4, seed=0, progress=None) -> dict:
    """Relative NTK drift during training, as a function of width.

    Lazy (kernel) regime: the drift should vanish as the width grows.  The
    scaling to look for is ||Theta_t - Theta_0||_F / ||Theta_0||_F ~ 1/sqrt(w).
    """
    from ..data.idx import load_dataset
    from ..train.losses import mse_loss

    d = load_dataset(dataset, "train", n=4000, dtype=np.float64)
    mask = np.isin(d["y"], list(classes))
    X, yl = d["X"][mask][:n], d["y"][mask][:n]
    y = np.where(yl == classes[0], -1.0, 1.0).reshape(-1, 1)
    rows = []
    for wi, w in enumerate(widths):
        net = MLP(dims=[X.shape[1]] + [w] * depth + [1], act=act,
                  sigma_w=sigma_w, readout_linear=True, seed=seed + wi)
        K0 = ntk_gram(net, X)
        n0 = np.linalg.norm(K0)
        f0 = net.forward(X)
        for _ in range(steps):
            out, cache = net.forward(X, cache=True)
            _, dout = mse_loss(out, y)
            gW, gb, _ = net.backward(cache, dout)
            for l in range(net.depth):
                net.W[l] -= lr * gW[l]
                net.b[l] -= lr * gb[l]
        K1 = ntk_gram(net, X)
        out = net.forward(X)
        loss, _ = mse_loss(out, y)
        rows.append({
            "width": int(w),
            "n_params": int(net.n_params),
            "kernel_drift_rel": float(np.linalg.norm(K1 - K0) / n0) if n0 > 0 else None,
            "kernel_trace_0": float(np.trace(K0)),
            "kernel_trace_1": float(np.trace(K1)),
            "final_mse": float(loss),
            "output_change": float(np.linalg.norm(out - f0)),
            "alignment_0": kernel_alignment(K0, y.ravel()),
            "alignment_1": kernel_alignment(K1, y.ravel()),
        })
        if progress:
            progress((wi + 1) / len(widths), f"width {w}")
    # fit log(drift) = a + b log(width); b close to -0.5 is the lazy prediction
    ws = np.array([r["width"] for r in rows], dtype=float)
    dr = np.array([r["kernel_drift_rel"] for r in rows], dtype=float)
    slope = None
    good = dr > 0
    if good.sum() >= 2:
        slope = float(np.polyfit(np.log(ws[good]), np.log(dr[good]), 1)[0])
    return {
        "kind": "l2_lazy_rich",
        "config": {"n": int(X.shape[0]), "depth": depth, "act": act,
                   "steps": steps, "lr": lr, "dataset": dataset,
                   "classes": list(classes), "sigma_w": sigma_w, "seed": seed},
        "series": rows,
        "log_log_slope": slope,
        "lazy_prediction_slope": -0.5,
        "claim": ("kernel drift decays with width; slope near -1/2 means the wide "
                  "network is training in the lazy regime"),
    }


def kernel_predicts_training(widths=(32, 128, 512), n=48, depth=1, act="tanh",
                             steps=150, lr="auto", stability=0.5, dataset="mnist",
                             classes=(0, 1), sigma_w=1.4, seed=0,
                             progress=None) -> dict:
    """Does the frozen NTK predict the real residual trajectory?

    Full-batch gradient descent on (1/2n)|f-y|^2 has the linearised solution
    r_{t+1} = (I - lr*Theta_0/n) r_t.  The test is how far the real network
    departs from that, and whether the departure shrinks with width.

    The learning rate must be chosen *per width*: the iteration is stable only
    for lr < 2n/lambda_max(Theta_0), and lambda_max grows with width.  With a
    fixed lr the narrow networks diverge, both prediction and reality blow up,
    and the comparison measures nothing.  lr="auto" takes
    `stability` * 2n/lambda_max and records what it used.
    """
    from ..data.idx import load_dataset
    from ..train.losses import mse_loss

    d = load_dataset(dataset, "train", n=4000, dtype=np.float64)
    mask = np.isin(d["y"], list(classes))
    X, yl = d["X"][mask][:n], d["y"][mask][:n]
    y = np.where(yl == classes[0], -1.0, 1.0).reshape(-1, 1)
    rows = []
    for wi, w in enumerate(widths):
        net = MLP(dims=[X.shape[1]] + [w] * depth + [1], act=act,
                  sigma_w=sigma_w, readout_linear=True, seed=seed + wi)
        K0 = ntk_gram(net, X)
        n_eff = X.shape[0]
        lam_max = float(np.linalg.eigvalsh(K0)[-1])
        lr_stable = 2.0 * n_eff / lam_max if lam_max > 0 else 1.0
        lr_used = stability * lr_stable if lr == "auto" else float(lr)
        f0 = net.forward(X)
        r = (f0 - y).copy()
        M = np.eye(n_eff) - lr_used * K0 / n_eff
        pred_curve, real_curve = [], []
        for t in range(steps):
            r = M @ r
            pred_curve.append(float(np.linalg.norm(r)))
            out, cache = net.forward(X, cache=True)
            _, dout = mse_loss(out, y)
            gW, gb, _ = net.backward(cache, dout)
            for l in range(net.depth):
                net.W[l] -= lr_used * gW[l]
                net.b[l] -= lr_used * gb[l]
            real_curve.append(float(np.linalg.norm(net.forward(X) - y)))
        rel = [abs(p - q) / max(q, 1e-12) for p, q in zip(pred_curve, real_curve)]
        rows.append({
            "width": int(w),
            "lr_used": lr_used,
            "lr_stability_limit": lr_stable,
            "ntk_lambda_max": lam_max,
            "diverged": bool(not np.isfinite(real_curve[-1]) or real_curve[-1] > 1e6),
            "predicted_residual": pred_curve,
            "actual_residual": real_curve,
            "final_rel_gap": float(rel[-1]),
            "max_rel_gap": float(max(rel)),
        })
        if progress:
            progress((wi + 1) / len(widths), f"width {w}")
    return {
        "kind": "l2_kernel_predicts_training",
        "config": {"n": int(X.shape[0]), "depth": depth, "act": act,
                   "steps": steps, "lr": lr, "stability": stability,
                   "dataset": dataset, "classes": list(classes),
                   "sigma_w": sigma_w, "seed": seed},
        "series": rows,
        "any_diverged": any(r["diverged"] for r in rows),
        "claim": ("the frozen NTK predicts the residual trajectory ever better "
                  "as width grows -- and visibly fails at small width"),
    }
