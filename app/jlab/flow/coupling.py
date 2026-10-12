"""Affine coupling layers (RealNVP-style) with both the cheap analytic
log-determinant and the full, explicitly assembled Jacobian.

The reason both exist: the analytic form is the entire selling point of
normalising flows ("the determinant is free"), and the assembled Jacobian is
the only way to check that the free thing is the true thing.  If
sum(s) != logabsdet(J), one of the two is wrong.
"""

from __future__ import annotations

import numpy as np

from ..jac.jacobian import input_jacobian
from ..jac.nets import MLP


class AffineCoupling:
    """y = m*x + (1-m) * (x * exp(s(m*x)) + t(m*x)).

    s is squashed with tanh and scaled so the layer cannot explode; that is a
    modelling choice and it is recorded in the run parameters, not hidden.
    """

    def __init__(self, dim: int, mask: np.ndarray, hidden: int = 32,
                 act: str = "tanh", scale_cap: float = 2.0, seed: int = 0):
        self.dim = int(dim)
        self.mask = np.asarray(mask, dtype=np.float64).reshape(-1)
        if self.mask.size != dim:
            raise ValueError("mask length must equal dim")
        self.scale_cap = float(scale_cap)
        self.s_net = MLP(dims=[dim, hidden, dim], act=act, sigma_w=1.0,
                         sigma_b=0.1, readout_linear=True, seed=seed)
        self.t_net = MLP(dims=[dim, hidden, dim], act=act, sigma_w=1.0,
                         sigma_b=0.1, readout_linear=True, seed=seed + 1)

    # -- pieces -------------------------------------------------------------
    def _s(self, xm):
        return self.scale_cap * np.tanh(self.s_net.forward(xm) / self.scale_cap)

    def _t(self, xm):
        return self.t_net.forward(xm)

    def forward(self, x: np.ndarray) -> tuple:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        xm = x * self.mask
        s = self._s(xm)
        t = self._t(xm)
        y = self.mask * x + (1 - self.mask) * (x * np.exp(s) + t)
        logdet = float(((1 - self.mask) * s).sum())
        return y, logdet

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        ym = y * self.mask                      # masked part is unchanged
        s = self._s(ym)
        t = self._t(ym)
        x = self.mask * y + (1 - self.mask) * ((y - t) * np.exp(-s))
        return x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Exact dy/dx, assembled.  Block-triangular by construction."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        m = self.mask
        xm = x * m
        raw_s = self.s_net.forward(xm)
        s = self.scale_cap * np.tanh(raw_s / self.scale_cap)
        # d/draw of the cap: sech^2(raw/cap)
        dcap = 1.0 - np.tanh(raw_s / self.scale_cap) ** 2
        Js = (dcap[:, None] * input_jacobian(self.s_net, xm)) * m[None, :]
        Jt = input_jacobian(self.t_net, xm) * m[None, :]
        e = np.exp(s)
        J = np.diag(m) + np.diag((1 - m) * e)
        J += ((1 - m) * x * e)[:, None] * Js
        J += (1 - m)[:, None] * Jt
        return J


class Flow:
    """A stack of couplings with alternating masks."""

    def __init__(self, dim: int, n_layers: int = 4, hidden: int = 32,
                 act: str = "tanh", scale_cap: float = 2.0, seed: int = 0):
        self.dim = int(dim)
        self.layers = []
        base = np.zeros(dim)
        base[::2] = 1.0
        for k in range(n_layers):
            mask = base if k % 2 == 0 else 1.0 - base
            self.layers.append(AffineCoupling(dim, mask, hidden=hidden, act=act,
                                              scale_cap=scale_cap, seed=seed + 10 * k))

    def forward(self, x: np.ndarray) -> tuple:
        z = np.asarray(x, dtype=np.float64).reshape(-1)
        total = 0.0
        for lay in self.layers:
            z, ld = lay.forward(z)
            total += ld
        return z, total

    def inverse(self, y: np.ndarray) -> np.ndarray:
        x = np.asarray(y, dtype=np.float64).reshape(-1)
        for lay in reversed(self.layers):
            x = lay.inverse(x)
        return x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Full dy/dx by chaining the per-layer Jacobians."""
        z = np.asarray(x, dtype=np.float64).reshape(-1)
        J = np.eye(self.dim)
        for lay in self.layers:
            J = lay.jacobian(z) @ J
            z, _ = lay.forward(z)
        return J

    def log_prob(self, x: np.ndarray) -> float:
        """Standard normal base density pushed through the change of variables."""
        z, logdet = self.forward(x)
        base = -0.5 * float(z @ z) - 0.5 * self.dim * np.log(2 * np.pi)
        return base + logdet
