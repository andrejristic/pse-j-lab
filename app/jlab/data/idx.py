"""IDX (MNIST format) reader with integrity verification.

The datasets were fetched once, at home, on a trusted network, and their
hashes recorded in datasets/MANIFEST.json.  On the road the app verifies
against that manifest rather than re-downloading: an experiment whose data
silently changed is worse than one that refuses to run.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from ..core import config

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATASETS = ("mnist", "fashion")

_CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
}


def _root(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return config.datasets_dir() / dataset


def load_idx(path: Path) -> np.ndarray:
    """Parse one IDX file (gzipped or plain).  Returns uint8 array."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:                      # images
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(n * rows * cols)
            if len(buf) != n * rows * cols:
                raise ValueError(f"{path.name}: truncated image payload")
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
        if magic == 2049:                      # labels
            buf = f.read(n)
            if len(buf) != n:
                raise ValueError(f"{path.name}: truncated label payload")
            return np.frombuffer(buf, dtype=np.uint8)
        raise ValueError(f"{path.name}: bad IDX magic {magic}")


def verify_manifest(dataset: str | None = None) -> dict:
    """Check sha256 of every file against datasets/MANIFEST.json."""
    man_path = config.datasets_dir() / "MANIFEST.json"
    if not man_path.exists():
        return {"ok": False, "reason": "MANIFEST.json missing",
                "manifest": str(man_path)}
    man = json.loads(man_path.read_text())
    results, ok = {}, True
    for rel, rec in man.items():
        if dataset and not rel.startswith(dataset + "/"):
            continue
        p = config.datasets_dir() / rel
        if not p.exists():
            results[rel] = "missing"
            ok = False
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        good = (h == rec.get("sha256_gz"))
        results[rel] = "ok" if good else "HASH MISMATCH"
        ok = ok and good
    return {"ok": ok, "files": results, "n": len(results)}


def dataset_status() -> dict:
    """What is actually on this machine -- used by the GUI and the selftest."""
    out = {"root": str(config.datasets_dir()), "datasets": {}}
    for ds in DATASETS:
        r = _root(ds)
        present = {k: (r / v).exists() for k, v in FILES.items()}
        out["datasets"][ds] = {
            "path": str(r),
            "complete": all(present.values()),
            "files": present,
        }
    out["manifest"] = verify_manifest()
    return out


def load_dataset(dataset: str = "mnist", split: str = "train",
                 n: int | None = None, flatten: bool = True,
                 normalise: str = "unit", dtype=np.float32,
                 seed: int | None = None) -> dict:
    """Load a split into memory.

    normalise: 'unit'      -> [0,1]
               'standard'  -> zero mean / unit variance over the loaded subset
               'none'      -> raw uint8 as float

    On a 4 GB machine the full 60k x 784 float32 train split is 188 MB, which
    is fine; float64 would be 376 MB, which is not.  Hence float32 default.
    """
    r = _root(dataset)
    imgs = load_idx(r / FILES[f"{split}_images"])
    labs = load_idx(r / FILES[f"{split}_labels"])
    if n is not None and n < imgs.shape[0]:
        if seed is None:
            idx = np.arange(n)
        else:
            idx = np.random.default_rng(seed).choice(imgs.shape[0], n, replace=False)
            idx.sort()
        imgs, labs = imgs[idx], labs[idx]
    X = imgs.astype(dtype)
    if flatten:
        X = X.reshape(X.shape[0], -1)
    if normalise == "unit":
        X = X / 255.0
    elif normalise == "standard":
        X = X / 255.0
        X = (X - X.mean()) / max(float(X.std()), 1e-8)
    elif normalise != "none":
        raise ValueError(f"unknown normalise {normalise!r}")
    y = labs.astype(np.int64)
    return {
        "X": X, "y": y, "dataset": dataset, "split": split,
        "n": int(X.shape[0]), "d": int(X.shape[1]) if flatten else None,
        "shape": list(imgs.shape[1:]),
        "classes": _CLASS_NAMES.get(dataset, []),
        "normalise": normalise,
    }


def subset(d: dict, n: int, seed: int = 0, balanced: bool = True) -> dict:
    """Take n samples, by default with equal counts per class."""
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(seed)
    if not balanced:
        idx = rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    else:
        classes = np.unique(y)
        per = max(1, n // len(classes))
        picks = []
        for c in classes:
            where = np.flatnonzero(y == c)
            picks.append(rng.choice(where, min(per, where.size), replace=False))
        idx = np.concatenate(picks)
        rng.shuffle(idx)
    out = dict(d)
    out["X"], out["y"], out["n"] = X[idx], y[idx], int(idx.size)
    return out


def one_hot(y: np.ndarray, k: int = 10, dtype=np.float64) -> np.ndarray:
    o = np.zeros((y.size, k), dtype=dtype)
    o[np.arange(y.size), y] = 1.0
    return o
