"""Exact Jacobians, matrix-free products, and honest cross-checks.

Three ways to get the same object, kept side by side on purpose:

  exact       -- multiply the layer factors (no autodiff, no approximation)
  matrix-free -- jvp / vjp, which never form J and cost one pass each
  numeric     -- central finite differences, slow, used only to falsify us

If `exact` ever disagrees with `numeric` beyond tolerance, the engine is wrong
and every downstream lab result is void.  That check is in the selftest.
"""

from __future__ import annotations

import numpy as np

from .nets import ACTIVATIONS, MLP


def input_jacobian(net: MLP, x: np.ndarray, mode: str = "auto") -> np.ndarray:
    """df/dx at a single point x.  Shape (d_out, d_in).

    mode 'forward' accumulates from the input side (cost ~ d_in), 'reverse'
    from the output side (cost ~ d_out), 'auto' picks the cheaper one.  Both
    are exact and must agree to machine precision.
    """
    x = np.asarray(x, dtype=net.dtype).reshape(-1)
    _, c = net.forward(x, cache=True)
    _, dphi = ACTIVATIONS[net.act]
    L = net.depth
    last = L - 1
    d_in, d_out = net.dims[0], net.dims[-1]

    if mode == "auto":
        mode = "reverse" if d_out <= d_in else "forward"

    if mode == "forward":
        M = net.W[0]                                    # (n1, d_in)
        for l in range(L - 1):
            d = dphi(c["z"][l][0])                      # (n_{l+1},)
            M = d[:, None] * M
            M = net.W[l + 1] @ M
        if not net.readout_linear:
            M = dphi(c["z"][last][0])[:, None] * M
        return M

    if mode == "reverse":
        M = net.W[last]                                 # (d_out, n_{L-1})
        if not net.readout_linear:
            M = dphi(c["z"][last][0])[:, None] * M
        for l in range(L - 2, -1, -1):
            d = dphi(c["z"][l][0])
            M = (M * d[None, :]) @ net.W[l]
        return M

    raise ValueError(f"unknown mode {mode!r}")


def input_jacobian_batch(net: MLP, X: np.ndarray, mode: str = "auto",
                         max_bytes: int = 256 * 1024 * 1024) -> np.ndarray:
    """Stack of Jacobians, shape (n, d_out, d_in).

    Refuses to allocate more than `max_bytes` -- on a 4 GB machine a silent
    784x512 x batch allocation is the difference between a result and a swap
    death.  Use `for_each` below if the stack genuinely does not fit.
    """
    X = np.atleast_2d(np.asarray(X, dtype=net.dtype))
    n = X.shape[0]
    need = n * net.dims[-1] * net.dims[0] * np.dtype(net.dtype).itemsize
    if need > max_bytes:
        raise MemoryError(
            f"stack of {n} Jacobians needs {need/2**20:.0f} MiB > "
            f"{max_bytes/2**20:.0f} MiB budget; iterate with for_each() instead")
    out = np.empty((n, net.dims[-1], net.dims[0]), dtype=net.dtype)
    for i in range(n):
        out[i] = input_jacobian(net, X[i], mode=mode)
    return out


def for_each(net: MLP, X: np.ndarray, fn, mode: str = "auto"):
    """Apply fn(J, i) per sample without ever holding the stack."""
    X = np.atleast_2d(np.asarray(X, dtype=net.dtype))
    return [fn(input_jacobian(net, X[i], mode=mode), i) for i in range(X.shape[0])]


def jvp(net: MLP, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """J(x) @ v without forming J (forward mode, one pass)."""
    x = np.asarray(x, dtype=net.dtype).reshape(-1)
    v = np.asarray(v, dtype=net.dtype).reshape(-1)
    _, c = net.forward(x, cache=True)
    _, dphi = ACTIVATIONS[net.act]
    last = net.depth - 1
    u = v
    for l in range(net.depth):
        u = net.W[l] @ u
        if not (l == last and net.readout_linear):
            u = dphi(c["z"][l][0]) * u
    return u


def vjp(net: MLP, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """u^T @ J(x) without forming J (reverse mode, one pass)."""
    x = np.asarray(x, dtype=net.dtype).reshape(-1)
    u = np.asarray(u, dtype=net.dtype).reshape(-1)
    _, c = net.forward(x, cache=True)
    _, dphi = ACTIVATIONS[net.act]
    last = net.depth - 1
    d = u
    for l in range(net.depth - 1, -1, -1):
        if not (l == last and net.readout_linear):
            d = d * dphi(c["z"][l][0])
        d = d @ net.W[l]
    return d


def finite_difference_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences.  O(2 d_in) forward passes -- for validation only."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f0 = np.atleast_1d(np.asarray(f(x), dtype=np.float64))
    J = np.empty((f0.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = eps
        fp = np.atleast_1d(np.asarray(f(x + e), dtype=np.float64))
        fm = np.atleast_1d(np.asarray(f(x - e), dtype=np.float64))
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def param_jacobian(net: MLP, x: np.ndarray, outputs=None) -> np.ndarray:
    """df/dtheta at one point.  Shape (n_out_selected, n_params).

    One reverse pass per selected output.  This is the object the NTK is built
    from: Theta(x, x') = J_theta(x) J_theta(x')^T.
    """
    x = np.asarray(x, dtype=net.dtype).reshape(1, -1)
    _, c = net.forward(x, cache=True)
    d_out = net.dims[-1]
    idx = range(d_out) if outputs is None else list(outputs)
    rows = []
    for k in idx:
        seed = np.zeros((1, d_out), dtype=net.dtype)
        seed[0, k] = 1.0
        gW, gb, _ = net.backward(c, seed)
        # backward() averages over the batch; batch is 1 here, so no rescale
        rows.append(np.concatenate([g.ravel() for g in gW] +
                                   [g.ravel() for g in gb]))
    return np.stack(rows, axis=0)


def param_jacobian_batch(net: MLP, X: np.ndarray, outputs=None,
                         max_bytes: int = 512 * 1024 * 1024) -> np.ndarray:
    """(n * n_out_selected, n_params).  Guarded against 4 GB suicide."""
    X = np.atleast_2d(np.asarray(X, dtype=net.dtype))
    n = X.shape[0]
    k = net.dims[-1] if outputs is None else len(list(outputs))
    need = n * k * net.n_params * np.dtype(net.dtype).itemsize
    if need > max_bytes:
        raise MemoryError(
            f"parameter Jacobian needs {need/2**20:.0f} MiB > "
            f"{max_bytes/2**20:.0f} MiB; reduce n, outputs, or width")
    return np.concatenate([param_jacobian(net, X[i], outputs) for i in range(n)], axis=0)
