"""SGD / momentum / Adam in numpy, with Jacobian instrumentation.

The trainer exists to answer questions about J during and after training, so
it records more than loss curves: at every checkpoint it can store a parameter
snapshot (for NTK drift), the spectrum of J at fixed probe points, and the
empirical local Lipschitz constant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..jac.jacobian import input_jacobian
from ..jac.nets import MLP
from ..jac.spectra import spectrum_summary
from .jacreg import jacobian_penalty_grad
from .losses import accuracy, mse_loss, softmax_cross_entropy


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.05
    opt: str = "sgd"                 # sgd | momentum | adam
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    loss: str = "ce"                 # ce | mse
    jac_lambda: float = 0.0          # Jacobian regularisation strength
    jac_nproj: int = 1
    seed: int = 0
    eval_every: int = 1              # epochs
    probe_n: int = 16                # probe points for J metrics
    snapshot_params: bool = False    # keep theta_t for NTK drift
    max_seconds: float = 0.0         # 0 = no wall-clock limit

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TrainResult:
    history: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    stopped_early: bool = False
    seconds: float = 0.0


class Trainer:
    def __init__(self, net: MLP, cfg: TrainConfig | None = None):
        self.net = net
        self.cfg = cfg or TrainConfig()
        self._init_opt()

    def _init_opt(self):
        self._vW = [np.zeros_like(W) for W in self.net.W]
        self._vb = [np.zeros_like(b) for b in self.net.b]
        self._mW = [np.zeros_like(W) for W in self.net.W]
        self._mb = [np.zeros_like(b) for b in self.net.b]
        self._t = 0

    # -- one optimiser step --------------------------------------------------
    def _apply(self, gW, gb):
        cfg = self.cfg
        self._t += 1
        for l in range(self.net.depth):
            gw, gbb = gW[l], gb[l]
            if cfg.weight_decay:
                gw = gw + cfg.weight_decay * self.net.W[l]
            if cfg.opt == "sgd":
                self.net.W[l] -= cfg.lr * gw
                self.net.b[l] -= cfg.lr * gbb
            elif cfg.opt == "momentum":
                self._vW[l] = cfg.momentum * self._vW[l] + gw
                self._vb[l] = cfg.momentum * self._vb[l] + gbb
                self.net.W[l] -= cfg.lr * self._vW[l]
                self.net.b[l] -= cfg.lr * self._vb[l]
            elif cfg.opt == "adam":
                self._mW[l] = cfg.beta1 * self._mW[l] + (1 - cfg.beta1) * gw
                self._vW[l] = cfg.beta2 * self._vW[l] + (1 - cfg.beta2) * gw * gw
                mhat = self._mW[l] / (1 - cfg.beta1 ** self._t)
                vhat = self._vW[l] / (1 - cfg.beta2 ** self._t)
                self.net.W[l] -= cfg.lr * mhat / (np.sqrt(vhat) + cfg.eps)
                self._mb[l] = cfg.beta1 * self._mb[l] + (1 - cfg.beta1) * gbb
                self._vb[l] = cfg.beta2 * self._vb[l] + (1 - cfg.beta2) * gbb * gbb
                mhb = self._mb[l] / (1 - cfg.beta1 ** self._t)
                vhb = self._vb[l] / (1 - cfg.beta2 ** self._t)
                self.net.b[l] -= cfg.lr * mhb / (np.sqrt(vhb) + cfg.eps)
            else:
                raise ValueError(f"unknown optimiser {cfg.opt!r}")

    # -- loss + gradient on one batch ---------------------------------------
    def _batch_grads(self, Xb, yb, rng_seed: int):
        net, cfg = self.net, self.cfg
        out, cache = net.forward(Xb, cache=True)
        if cfg.loss == "ce":
            loss, dout = softmax_cross_entropy(out, yb)
        elif cfg.loss == "mse":
            loss, dout = mse_loss(out, yb)
        else:
            raise ValueError(f"unknown loss {cfg.loss!r}")
        gW, gb, _ = net.backward(cache, dout)
        extra = 0.0
        if cfg.jac_lambda > 0:
            p, (jW, jb) = jacobian_penalty_grad(net, Xb, n_proj=cfg.jac_nproj,
                                                seed=rng_seed)
            extra = cfg.jac_lambda * p
            for l in range(net.depth):
                gW[l] = gW[l] + cfg.jac_lambda * jW[l]
                gb[l] = gb[l] + cfg.jac_lambda * jb[l]
        return loss, extra, gW, gb, out

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, X, y, probe: np.ndarray | None = None) -> dict:
        out = self.net.forward(X)
        if self.cfg.loss == "ce":
            loss, _ = softmax_cross_entropy(out, y)
            acc = accuracy(out, y)
        else:
            loss, _ = mse_loss(out, y)
            acc = float("nan")
        rec = {"loss": loss, "acc": acc}
        if probe is not None and probe.size:
            svs, fro = [], []
            for i in range(probe.shape[0]):
                J = input_jacobian(self.net, probe[i])
                s = np.linalg.svd(J, compute_uv=False)
                svs.append(s)
                fro.append(float((J * J).sum()))
            S = np.stack(svs)
            rec["jac"] = {
                "s_max_mean": float(S[:, 0].mean()),
                "s_max_max": float(S[:, 0].max()),
                "frob_sq_mean": float(np.mean(fro)),
                "spectrum_mean": spectrum_summary(s=S.mean(axis=0)),
            }
        return rec

    # -- main loop -----------------------------------------------------------
    def fit(self, X, y, Xval=None, yval=None, probe=None,
            on_epoch=None) -> TrainResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        X = np.asarray(X, dtype=self.net.dtype)
        n = X.shape[0]
        if probe is None and cfg.probe_n > 0:
            probe = X[rng.choice(n, min(cfg.probe_n, n), replace=False)]
        res = TrainResult()
        t0 = time.perf_counter()
        step = 0
        for ep in range(cfg.epochs):
            order = rng.permutation(n)
            ep_loss, ep_extra, nb = 0.0, 0.0, 0
            for i in range(0, n, cfg.batch_size):
                idx = order[i:i + cfg.batch_size]
                loss, extra, gW, gb, _ = self._batch_grads(X[idx], y[idx], cfg.seed + step)
                self._apply(gW, gb)
                ep_loss += loss
                ep_extra += extra
                nb += 1
                step += 1
            rec = {"epoch": ep + 1, "train_loss": ep_loss / max(nb, 1),
                   "jac_penalty": ep_extra / max(nb, 1),
                   "seconds": time.perf_counter() - t0}
            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1:
                rec["train"] = self.evaluate(X, y, probe)
                if Xval is not None:
                    rec["val"] = self.evaluate(Xval, yval)
            if cfg.snapshot_params:
                res.snapshots.append({"epoch": ep + 1,
                                      "params": self.net.get_params().copy()})
            res.history.append(rec)
            if on_epoch is not None:
                on_epoch(rec)
            if cfg.max_seconds and (time.perf_counter() - t0) > cfg.max_seconds:
                res.stopped_early = True
                break
        res.seconds = time.perf_counter() - t0
        res.final = {"train": self.evaluate(X, y, probe)}
        if Xval is not None:
            res.final["val"] = self.evaluate(Xval, yval)
        return res
