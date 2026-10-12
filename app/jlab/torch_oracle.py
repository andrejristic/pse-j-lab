"""Torch as an independent oracle -- never as a dependency.

Everything in the five labs runs without torch. This module exists for one
purpose: to rebuild the same network in a library we did not write and check
that its autodiff agrees with our hand-derived Jacobians. If they disagree,
our engine is wrong, and no amount of internal consistency would have caught it.

Three quantities are compared:

    input Jacobian      ours  vs  torch.func.jacrev
    parameter Jacobian  ours  vs  torch.autograd per output
    ||J||_F^2 penalty   ours (exact double backprop)  vs  torch double backward

The third is the strongest check, because it is the one piece of mathematics
in this app that was derived by hand rather than taken from a textbook.
"""

from __future__ import annotations

import numpy as np

from .jac.nets import MLP
from .jac.jacobian import input_jacobian, param_jacobian
from .train.jacreg import jacobian_penalty_grad


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _to_torch(net: MLP):
    import torch
    import torch.nn as nn
    acts = {"tanh": torch.tanh, "relu": torch.relu, "sin": torch.sin,
            "softplus": nn.functional.softplus, "erf": torch.erf,
            "identity": lambda z: z}
    if net.act not in acts:
        raise ValueError(f"activation {net.act} not mapped for the torch oracle")
    phi = acts[net.act]
    Ws = [torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in net.W]
    bs = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in net.b]

    def f(x, Ws=Ws, bs=bs):
        h = x
        last = len(Ws) - 1
        for l, (W, b) in enumerate(zip(Ws, bs)):
            z = h @ W.T + b
            h = z if (l == last and net.readout_linear) else phi(z)
        return h

    return f, Ws, bs


def check_input_jacobian(net: MLP, x: np.ndarray) -> dict:
    import torch
    f, _, _ = _to_torch(net)
    xt = torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=True)
    Jt = torch.func.jacrev(f)(xt).detach().numpy()
    Jo = input_jacobian(net, x)
    err = float(np.abs(Jt - Jo).max())
    return {"quantity": "input Jacobian", "max_abs_err": err,
            "scale": float(np.abs(Jt).max()), "ok": err < 1e-10}


def check_param_jacobian(net: MLP, x: np.ndarray) -> dict:
    import torch
    f, Ws, bs = _to_torch(net)
    xt = torch.tensor(np.asarray(x, dtype=np.float64).reshape(-1))
    out = f(xt)
    rows = []
    for k in range(out.numel()):
        grads = torch.autograd.grad(out[k], Ws + bs, retain_graph=True,
                                    allow_unused=False)
        rows.append(np.concatenate([g.detach().numpy().ravel() for g in grads]))
    Jt = np.stack(rows)
    Jo = param_jacobian(net, x)
    err = float(np.abs(Jt - Jo).max())
    return {"quantity": "parameter Jacobian", "max_abs_err": err,
            "scale": float(np.abs(Jt).max()), "ok": err < 1e-10}


def check_penalty_gradient(net: MLP, X: np.ndarray, n_proj: int = 3,
                           seed: int = 11) -> dict:
    """The hand-derived double backprop against torch's second derivatives.

    The random projections must be identical on both sides, so they are drawn
    once here with the same generator jacreg uses.
    """
    import torch
    from .train.jacreg import _projections
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, d_out = X.shape[0], net.dims[-1]
    rng = np.random.default_rng(seed)
    V = _projections(rng, n, d_out, n_proj)
    scale = d_out / (n_proj * n)

    f, Ws, bs = _to_torch(net)
    Xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    for p in range(n_proj):
        v = torch.tensor(V[p], dtype=torch.float64)
        Xp = Xt.clone().requires_grad_(True)
        out = f(Xp)
        # v^T J via a vector-Jacobian product, then square it
        (a,) = torch.autograd.grad((out * v).sum(), Xp, create_graph=True)
        total = total + 0.5 * scale * (a * a).sum()
    # allow_unused: with a linear readout the last layer's bias enters the
    # network additively and therefore does not affect J at all, so it is
    # genuinely absent from the penalty's graph. Our analytic code returns
    # exact zeros for it; torch returning None is the same statement.
    params = Ws + bs
    grads = torch.autograd.grad(total, params, allow_unused=True)
    gt = np.concatenate([
        (np.zeros(tuple(p.shape)) if g is None else g.detach().numpy()).ravel()
        for g, p in zip(grads, params)])
    n_unused = sum(1 for g in grads if g is None)

    P, g = jacobian_penalty_grad(net, X, n_proj=n_proj, seed=seed)
    go = np.concatenate([a.ravel() for a in g[0]] + [a.ravel() for a in g[1]])
    val_err = float(abs(float(total.item()) - P))
    grad_err = float(np.abs(gt - go).max())
    return {"quantity": "||J||_F^2 penalty and its gradient",
            "value_err": val_err, "max_abs_grad_err": grad_err,
            "scale": float(np.abs(gt).max()),
            "params_not_in_penalty_graph": n_unused,
            "ok": val_err < 1e-10 and grad_err < 1e-9}


def run_all(seed: int = 0) -> dict:
    """Every oracle check across activations and both readout styles."""
    if not available():
        return {"available": False,
                "note": "torch is not installed; the labs do not need it"}
    rng = np.random.default_rng(seed)
    results = []
    for act in ("tanh", "relu", "erf", "sin", "softplus"):
        for readout in (True, False):
            net = MLP(dims=[5, 7, 6, 4], act=act, sigma_w=1.3, sigma_b=0.2,
                      readout_linear=readout, seed=seed + 3)
            x = rng.standard_normal(5)
            X = rng.standard_normal((4, 5))
            for r in (check_input_jacobian(net, x),
                      check_param_jacobian(net, x),
                      check_penalty_gradient(net, X)):
                r.update({"act": act, "readout_linear": readout})
                results.append(r)
    bad = [r for r in results if not r["ok"]]
    return {"available": True, "n_checks": len(results), "n_failed": len(bad),
            "ok": not bad, "failures": bad, "results": results}
