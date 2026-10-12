"""Small feed-forward networks with explicit, inspectable internals.

Deliberately not a framework.  Every quantity the labs need -- pre-activations,
activation derivatives, per-layer Jacobian factors, parameter gradients -- is
returned as a plain numpy array, because the point of this app is to look at
those arrays, not to hide them.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass, field

import numpy as np

# --- activations --------------------------------------------------------------
# each entry: (phi, phi', name for plots)


def _tanh(z):
    return np.tanh(z)


def _dtanh(z):
    t = np.tanh(z)
    return 1.0 - t * t


def _relu(z):
    return np.maximum(z, 0.0)


def _drelu(z):
    # derivative at 0 taken as 0; the measure-zero choice is recorded, not hidden
    return (z > 0.0).astype(z.dtype)


# numpy has no erf and scipy is not a dependency.  An A&S 7.1.26 polynomial
# would be fast but only 1e-7 accurate, and its analytic derivative below would
# then disagree with the implemented function at 1e-6 -- which the
# finite-difference selftest correctly flags as an engine error.  math.erf is
# correctly rounded, so we vectorise it and keep phi and phi' consistent to
# machine precision.  Cost is ~0.1 us/element; irrelevant at our widths.
_erf_uf = np.frompyfunc(_math.erf, 1, 1)


def _erf(z):
    return _erf_uf(z).astype(z.dtype if hasattr(z, "dtype") else float)


def _derf(z):
    return 2.0 / np.sqrt(np.pi) * np.exp(-z * z)


def _sin(z):
    return np.sin(z)


def _dsin(z):
    return np.cos(z)


def _softplus(z):
    return np.logaddexp(0.0, z)


def _dsoftplus(z):
    return 1.0 / (1.0 + np.exp(-z))


def _identity(z):
    return z


def _didentity(z):
    return np.ones_like(z)


def _hardtanh(z):
    return np.clip(z, -1.0, 1.0)


def _dhardtanh(z):
    return ((z > -1.0) & (z < 1.0)).astype(z.dtype)


ACTIVATIONS = {
    "tanh": (_tanh, _dtanh),
    "relu": (_relu, _drelu),
    "erf": (_erf, _derf),
    "sin": (_sin, _dsin),
    "softplus": (_softplus, _dsoftplus),
    "hardtanh": (_hardtanh, _dhardtanh),
    "identity": (_identity, _didentity),
}


# second derivatives -- needed only by the exact Jacobian-penalty gradient
# (double backprop) in train/jacreg.py


def _d2tanh(z):
    t = np.tanh(z)
    return -2.0 * t * (1.0 - t * t)


def _d2erf(z):
    return -4.0 / np.sqrt(np.pi) * z * np.exp(-z * z)


def _d2softplus(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 - s)


D2_ACTIVATIONS = {
    "tanh": _d2tanh,
    "relu": lambda z: np.zeros_like(z),       # a.e.; the kink is measure zero
    "erf": _d2erf,
    "sin": lambda z: -np.sin(z),
    "softplus": _d2softplus,
    "hardtanh": lambda z: np.zeros_like(z),   # a.e.
    "identity": lambda z: np.zeros_like(z),
}

INITS = ("gaussian", "orthogonal", "uniform")


def _orthogonal(rng, rows, cols, gain=1.0):
    """Uniform (Haar) orthogonal / semi-orthogonal matrix via QR with sign fix."""
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))          # make QR unique -> Haar measure
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


@dataclass
class MLP:
    """Fully connected net.  dims = [n_in, h1, ..., n_out].

    sigma_w, sigma_b follow the mean-field convention: weights are drawn with
    variance sigma_w^2 / fan_in, biases with variance sigma_b^2.  With
    `readout_linear` the last layer has no nonlinearity (a classifier); without
    it, every layer is nonlinear (the isometry setting).
    """

    dims: list
    act: str = "tanh"
    init: str = "gaussian"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    readout_linear: bool = True
    seed: int = 0
    dtype: type = np.float64
    W: list = field(default_factory=list, repr=False)
    b: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.act not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.act!r}")
        if self.init not in INITS:
            raise ValueError(f"unknown init {self.init!r}")
        if len(self.dims) < 2:
            raise ValueError("need at least [n_in, n_out]")
        if not self.W:
            self.reset(self.seed)

    # -- construction -------------------------------------------------------
    def reset(self, seed: int | None = None) -> "MLP":
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for i in range(len(self.dims) - 1):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]
            if self.init == "gaussian":
                w = rng.standard_normal((fan_out, fan_in)) * (self.sigma_w / np.sqrt(fan_in))
            elif self.init == "uniform":
                lim = np.sqrt(3.0) * self.sigma_w / np.sqrt(fan_in)
                w = rng.uniform(-lim, lim, size=(fan_out, fan_in))
            else:  # orthogonal: gain sigma_w, exactly isometric when square
                w = _orthogonal(rng, fan_out, fan_in, gain=self.sigma_w)
            self.W.append(np.ascontiguousarray(w, dtype=self.dtype))
            self.b.append(np.zeros(fan_out, dtype=self.dtype)
                          if self.sigma_b == 0.0 else
                          (rng.standard_normal(fan_out) * self.sigma_b).astype(self.dtype))
        return self

    @property
    def depth(self) -> int:
        return len(self.W)

    @property
    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(bb.size for bb in self.b)

    def copy(self) -> "MLP":
        m = MLP(dims=list(self.dims), act=self.act, init=self.init,
                sigma_w=self.sigma_w, sigma_b=self.sigma_b,
                readout_linear=self.readout_linear, seed=self.seed, dtype=self.dtype,
                W=[w.copy() for w in self.W], b=[bb.copy() for bb in self.b])
        return m

    # -- parameter vector ---------------------------------------------------
    def get_params(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.W] + [bb.ravel() for bb in self.b])

    def set_params(self, theta: np.ndarray) -> "MLP":
        i = 0
        for k, w in enumerate(self.W):
            n = w.size
            self.W[k] = theta[i:i + n].reshape(w.shape).astype(self.dtype)
            i += n
        for k, bb in enumerate(self.b):
            n = bb.size
            self.b[k] = theta[i:i + n].reshape(bb.shape).astype(self.dtype)
            i += n
        if i != theta.size:
            raise ValueError("parameter vector length mismatch")
        return self

    def param_slices(self) -> list:
        """[(name, start, stop, shape)] in the same order as get_params()."""
        out, i = [], 0
        for k, w in enumerate(self.W):
            out.append((f"W{k+1}", i, i + w.size, w.shape)); i += w.size
        for k, bb in enumerate(self.b):
            out.append((f"b{k+1}", i, i + bb.size, bb.shape)); i += bb.size
        return out

    # -- forward ------------------------------------------------------------
    def forward(self, X: np.ndarray, cache: bool = False):
        """X: (n, d_in) or (d_in,).  Returns (n, d_out) [and a cache dict]."""
        single = (X.ndim == 1)
        H = np.atleast_2d(np.asarray(X, dtype=self.dtype))
        zs, hs = [], [H]
        phi, _ = ACTIVATIONS[self.act]
        last = self.depth - 1
        for l in range(self.depth):
            Z = H @ self.W[l].T + self.b[l]
            zs.append(Z)
            if l == last and self.readout_linear:
                H = Z
            else:
                H = phi(Z)
            hs.append(H)
        out = H[0] if single else H
        if cache:
            return out, {"z": zs, "h": hs, "single": single}
        return out

    def __call__(self, X):
        return self.forward(X)

    # -- per-layer Jacobian factors ----------------------------------------
    def layer_factors(self, x: np.ndarray) -> list:
        """[W_1, D_1, W_2, ..., D_{L-1}, W_L] as a list of arrays for one x.

        Returned in *application order* (right to left in the matrix product),
        i.e. J = factors[-1] @ ... @ factors[0].
        """
        _, c = self.forward(np.asarray(x, dtype=self.dtype), cache=True)
        _, dphi = ACTIVATIONS[self.act]
        facs = []
        last = self.depth - 1
        for l in range(self.depth):
            facs.append(self.W[l])
            nonlinear_here = not (l == last and self.readout_linear)
            if nonlinear_here:
                facs.append(dphi(c["z"][l][0]))          # stored as a 1-D diagonal
        return facs

    # -- gradients (for training) ------------------------------------------
    def backward(self, cache: dict, dL_dout: np.ndarray) -> tuple:
        """Reverse pass.  dL_dout: (n, d_out).  Returns (gW, gb, dL_dx)."""
        _, dphi = ACTIVATIONS[self.act]
        zs, hs = cache["z"], cache["h"]
        n = dL_dout.shape[0]
        gW = [None] * self.depth
        gb = [None] * self.depth
        delta = np.asarray(dL_dout, dtype=self.dtype)
        last = self.depth - 1
        for l in range(self.depth - 1, -1, -1):
            if not (l == last and self.readout_linear):
                delta = delta * dphi(zs[l])
            gW[l] = delta.T @ hs[l] / n
            gb[l] = delta.mean(axis=0)
            delta = delta @ self.W[l]
        return gW, gb, delta
