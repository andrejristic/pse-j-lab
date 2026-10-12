"""Jacobian regularisation with an *exact* gradient (double backprop, numpy).

The penalty is the Frobenius norm of the input Jacobian, estimated with random
projections (Hoffman, Roberts & Yaida 2019):

    ||J(x)||_F^2 = d_out * E_v[ ||v^T J(x)||^2 ],   v uniform on the sphere

Computing dP/dtheta requires differentiating *through* the reverse pass, so
second derivatives of the activation appear.  The derivation is spelled out in
the code because a hand-derived double backprop that is merely asserted to be
correct is worthless -- and it is checked against finite differences in the
selftest, where an error of 1e-9 or worse fails the build.
"""

from __future__ import annotations

import numpy as np

from ..jac.nets import ACTIVATIONS, D2_ACTIVATIONS, MLP


def _projections(rng, n: int, d_out: int, n_proj: int, sphere: bool = True):
    """n_proj random directions per sample, shape (n_proj, n, d_out)."""
    V = rng.standard_normal((n_proj, n, d_out))
    if sphere:
        V /= np.linalg.norm(V, axis=2, keepdims=True)
    return V


def _forward_cache(net: MLP, X: np.ndarray):
    _, c = net.forward(X, cache=True)
    return c


def jacobian_penalty(net: MLP, X: np.ndarray, n_proj: int = 1, seed: int = 0,
                     exact: bool = False) -> float:
    """P = (1/2n) sum_i ||J(x_i)||_F^2, estimated or exact."""
    X = np.atleast_2d(np.asarray(X, dtype=net.dtype))
    if exact:
        from ..jac.jacobian import input_jacobian
        tot = sum(float((input_jacobian(net, X[i]) ** 2).sum())
                  for i in range(X.shape[0]))
        return 0.5 * tot / X.shape[0]
    val, _ = jacobian_penalty_grad(net, X, n_proj=n_proj, seed=seed, want_grad=False)
    return val


def jacobian_penalty_grad(net: MLP, X: np.ndarray, n_proj: int = 1,
                          seed: int = 0, want_grad: bool = True) -> tuple:
    """Returns (P, (gW, gb)) with P = (1/2n) sum_i d_out * E_v ||v^T J(x_i)||^2.

    Forward of the penalty (row-vector convention, batch in the first axis):

        s_L = v                                  (readout linear)
        s_L = v * phi'(z_L)                      (readout nonlinear)
        t_l = s_{l+1} @ W_{l+1};  s_l = t_l * phi'(z_l)      l = L-1 .. 1
        a   = s_1 @ W_1                          == v^T J(x)
        P   = c/2 * ||a||^2,   c = d_out / n_proj / n

    Reverse of that computation gives the gradient; the terms marked (2nd)
    are the ones that only exist because we are differentiating a derivative.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    L = net.depth
    last = L - 1
    phi, dphi = ACTIVATIONS[net.act]
    d2phi = D2_ACTIVATIONS[net.act]

    c = _forward_cache(net, X)
    zs, hs = c["z"], c["h"]
    dphis = [dphi(z) for z in zs]

    rng = np.random.default_rng(seed)
    V = _projections(rng, n, net.dims[-1], n_proj)
    scale = net.dims[-1] / (n_proj * n)

    gW = [np.zeros_like(W) for W in net.W]
    gb = [np.zeros_like(b) for b in net.b]
    total = 0.0

    for p in range(n_proj):
        v = V[p]                                    # (n, d_out)

        # ---- forward pass of the penalty ---------------------------------
        s = [None] * (L + 1)                        # s[l] lives after layer l
        t = [None] * (L + 1)
        if net.readout_linear:
            s[L] = v
        else:
            t[L] = v
            s[L] = v * dphis[last]
        for l in range(L - 1, 0, -1):
            t[l] = s[l + 1] @ net.W[l]              # (n, n_l)
            s[l] = t[l] * dphis[l - 1]
        a = s[1] @ net.W[0]                         # (n, d_in) == v^T J
        total += 0.5 * scale * float((a * a).sum())
        if not want_grad:
            continue

        # ---- reverse pass ------------------------------------------------
        abar = scale * a                            # dP/da
        gW[0] += s[1].T @ abar                      # a = s_1 @ W_1
        sbar = [None] * (L + 1)
        sbar[1] = abar @ net.W[0].T                 # (n, n_1)
        zbar = [np.zeros_like(z) for z in zs]

        for l in range(1, L):
            # s_l = t_l * phi'(z_{l-1});  t_l = s_{l+1} @ W_{l+1}
            tbar = sbar[l] * dphis[l - 1]
            dbar = sbar[l] * t[l]                   # (2nd) d/d phi'(z_{l-1})
            zbar[l - 1] += dbar * d2phi(zs[l - 1])  # (2nd) chain through phi''
            gW[l] += s[l + 1].T @ tbar
            contrib = tbar @ net.W[l].T
            sbar[l + 1] = contrib if sbar[l + 1] is None else sbar[l + 1] + contrib

        if not net.readout_linear:
            # s_L = v * phi'(z_L)
            zbar[last] += (sbar[L] * v) * d2phi(zs[last])

        # ---- propagate zbar back through the ordinary forward graph ------
        hbar = None
        for l in range(L - 1, -1, -1):
            zb = zbar[l] if hbar is None else zbar[l] + hbar * (
                np.ones_like(zs[l]) if (l == last and net.readout_linear) else dphis[l])
            gW[l] += zb.T @ hs[l]
            gb[l] += zb.sum(axis=0)
            hbar = zb @ net.W[l]

    return total, (gW, gb)


def frobenius_estimate(net: MLP, X: np.ndarray, n_proj: int = 8,
                       seed: int = 0) -> dict:
    """Random-projection estimate of mean ||J||_F^2 vs the exact value.

    Reported together so the estimator's own error is visible rather than
    assumed small.
    """
    est = 2.0 * jacobian_penalty(net, X, n_proj=n_proj, seed=seed)
    exact = 2.0 * jacobian_penalty(net, X, exact=True)
    return {
        "n_proj": n_proj,
        "estimate": est,
        "exact": exact,
        "rel_error": float(abs(est - exact) / exact) if exact > 0 else None,
    }
