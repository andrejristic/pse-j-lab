"""Losses returning (value, dL/dlogits) so nothing is recomputed."""

from __future__ import annotations

import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple:
    """Numerically stable CE.  y is an integer label vector."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = z.shape[0]
    loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-300, None)).mean())
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    return loss, g            # note: net.backward() divides by n


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    loss = float(0.5 * (d * d).sum(axis=1).mean())
    return loss, d


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(logits, axis=1) == y).mean())
