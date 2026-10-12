"""L4 -- intrinsic dimension from the rank of a Jacobian.

An autoencoder trained on data that lies on a d-dimensional manifold can only
move its output along that manifold, so the Jacobian of the reconstruction map
should have about d large singular values and a sharp drop after them.  That
is the claim; the reason this lab exists is that the true d is known here, so
the claim can lose.

The estimator is deliberately simple (largest gap in the log spectrum) and is
reported alongside two alternatives, because "pick the biggest gap" is exactly
the kind of rule that looks decisive on clean data and falls apart with noise.
"""

from __future__ import annotations

import numpy as np

from ..data.manifolds import MANIFOLDS
from ..jac.jacobian import input_jacobian
from ..jac.nets import MLP
from ..jac.spectra import effective_rank, participation_ratio
from ..train.trainer import TrainConfig, Trainer


def estimate_dimension(s: np.ndarray, thresh_rel: float = 0.05,
                       rank_tol: float = 1e-6) -> dict:
    """Three estimators from one singular value spectrum.

    The gap search runs only over singular values above `rank_tol * s_1`.
    Without that guard it is useless: an autoencoder with a bottleneck of size
    b has *exact* numerical rank b, so the step from s_b to the 1e-16 tail is
    the largest log gap in the spectrum and the estimator would report the
    bottleneck width every time -- an artefact of the architecture, not a
    property of the data.
    """
    s = np.asarray(s, dtype=np.float64)
    s = np.sort(s)[::-1]
    if s.size == 0 or s[0] <= 0:
        return {"gap": None, "threshold": 0, "effective_rank": 0.0,
                "participation_ratio": 0.0, "largest_log_gap": None,
                "numerical_rank": 0}
    keep = s[s > rank_tol * s[0]]
    gap_dim = None
    largest = None
    if keep.size >= 2:
        logs = np.log(keep)
        gaps = logs[:-1] - logs[1:]
        gap_dim = int(np.argmax(gaps) + 1)
        largest = float(np.max(gaps))
    elif keep.size == 1:
        gap_dim = 1
    thresh_dim = int((s > thresh_rel * s[0]).sum())
    return {
        "gap": gap_dim,
        "threshold": thresh_dim,
        "effective_rank": effective_rank(s),
        "participation_ratio": participation_ratio(s),
        "largest_log_gap": largest,
        "numerical_rank": int(keep.size),
    }


def local_pca_dimension(X: np.ndarray, k: int = 40, n_probe: int = 24,
                        seed: int = 0, rank_tol: float = 1e-6) -> dict:
    """Data-only intrinsic dimension: PCA on each point's k nearest neighbours.

    This estimator never looks at the network.  It is here so that when the
    Jacobian estimate and the truth disagree, it is possible to say which of
    the two suspects is at fault -- the data, or the autoencoder.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(n_probe, X.shape[0]), replace=False)
    dims, spectra = [], []
    for i in idx:
        d2 = ((X - X[i]) ** 2).sum(axis=1)
        nb = np.argsort(d2)[:k + 1]
        P = X[nb] - X[nb].mean(axis=0)
        s = np.linalg.svd(P, compute_uv=False)
        spectra.append(s)
        dims.append(estimate_dimension(s, rank_tol=rank_tol)["gap"])
    dims = [d for d in dims if d is not None]
    return {
        "median": float(np.median(dims)) if dims else None,
        "k": k,
        "median_spectrum": np.median(np.stack(spectra), axis=0).tolist(),
    }


def subspace_alignment(J: np.ndarray, X: np.ndarray, i: int, d: int,
                       k: int = 40) -> float:
    """Mean cos of principal angles between J's top-d input directions and the
    local data span.  1.0 means the network's sensitive directions lie along
    the manifold; less means it is also sensitive off it.
    """
    _, _, Vt = np.linalg.svd(J)
    A = Vt[:d].T
    d2 = ((X - X[i]) ** 2).sum(axis=1)
    nb = np.argsort(d2)[:k + 1]
    P = X[nb] - X[nb].mean(axis=0)
    _, _, Wt = np.linalg.svd(P)
    B = Wt[:d].T
    sv = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.clip(sv, 0.0, 1.0).mean())


def manifold_dimension(manifolds=("linear", "sphere", "swiss_roll", "torus", "product"),
                       n=800, ambient=12, noise=0.0, hidden=64, bottleneck=8,
                       epochs=150, lr=0.01, act="tanh", n_probe=24,
                       train_noise=0.1, contractive=0.0, seed=0,
                       progress=None) -> dict:
    """Train an autoencoder per manifold; read the dimension off J's spectrum.

    The bottleneck is deliberately *wider* than every true dimension, so the
    estimate has to come from the data, not from the architecture.

    `train_noise` makes it a denoising autoencoder, and that is not cosmetic:
    a plain autoencoder never sees a point off the manifold, so its Jacobian
    in off-manifold directions is unconstrained by the loss and carries no
    information about the manifold.  Corrupting the input is what forces those
    directions to contract.  `contractive` adds the exact ||J||_F penalty
    instead (Rifai et al.); both routes are available, neither is assumed.
    """
    rows = []
    for mi, name in enumerate(manifolds):
        gen = MANIFOLDS[name]
        kw = dict(n=n, ambient=ambient, noise=noise, seed=seed)
        if name == "linear":
            kw["dim"] = 3
        elif name == "sphere":
            kw["dim"] = 2
        elif name == "product":
            kw["dims"] = (1, 2)
        m = gen(**kw)
        X = np.asarray(m["X"], dtype=np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        ae = MLP(dims=[ambient, hidden, bottleneck, hidden, ambient], act=act,
                 init="orthogonal", sigma_w=1.2, readout_linear=True,
                 seed=seed + 31 * mi)
        cfg = TrainConfig(epochs=epochs, batch_size=64, lr=lr, opt="adam",
                          loss="mse", jac_lambda=float(contractive), jac_nproj=2,
                          seed=seed, probe_n=0, eval_every=epochs)
        rng_tr = np.random.default_rng(seed + 991 + mi)
        Xin = (X + train_noise * rng_tr.standard_normal(X.shape)
               if train_noise > 0 else X)
        res = Trainer(ae, cfg).fit(Xin, X)
        rng = np.random.default_rng(seed + mi)
        idx = rng.choice(X.shape[0], min(n_probe, X.shape[0]), replace=False)
        spectra, ests = [], []
        for i in idx:
            J = input_jacobian(ae, X[i])
            s = np.linalg.svd(J, compute_uv=False)
            spectra.append(s)
            ests.append(estimate_dimension(s))
        S = np.stack(spectra)
        med = np.median(S, axis=0)
        pca = local_pca_dimension(X, k=min(40, max(10, n // 20)),
                                  n_probe=min(n_probe, 24), seed=seed)
        true_d = int(m["intrinsic_dim"])
        align = float(np.median([subspace_alignment(input_jacobian(ae, X[i]), X, i,
                                                    true_d,
                                                    k=min(40, max(10, n // 20)))
                                 for i in idx[:min(len(idx), 12)]]))
        rows.append({
            "local_pca_dim": pca["median"],
            "local_pca_spectrum": pca["median_spectrum"],
            "tangent_alignment_at_true_dim": align,
            "manifold": m["name"],
            "generator": name,
            "true_dim": int(m["intrinsic_dim"]),
            "ambient": ambient,
            "noise": noise,
            "recon_mse": res.final["train"]["loss"],
            "median_spectrum": med.tolist(),
            "gap_estimate_median": float(np.median([e["gap"] for e in ests
                                                    if e["gap"] is not None])),
            "threshold_estimate_median": float(np.median([e["threshold"] for e in ests])),
            "effective_rank_median": float(np.median([e["effective_rank"] for e in ests])),
            "gap_estimate_from_median_spectrum": estimate_dimension(med)["gap"],
            "numerical_rank_median": float(np.median([e["numerical_rank"] for e in ests])),
            "correct_by_gap": bool(round(float(np.median([e["gap"] for e in ests
                                                         if e["gap"] is not None])))
                                   == int(m["intrinsic_dim"])),
        })
        if progress:
            progress((mi + 1) / len(manifolds), name)
    n_ok = sum(1 for r in rows if r["correct_by_gap"])
    n_pca_ok = sum(1 for r in rows
                   if r["local_pca_dim"] is not None
                   and round(r["local_pca_dim"]) == r["true_dim"])
    return {
        "kind": "l4_manifold_dimension",
        "n_correct_local_pca": n_pca_ok,
        "config": {"n": n, "ambient": ambient, "noise": noise, "hidden": hidden,
                   "bottleneck": bottleneck, "epochs": epochs, "lr": lr,
                   "act": act, "n_probe": n_probe, "train_noise": train_noise,
                   "contractive": contractive, "seed": seed},
        "series": rows,
        "n_correct": n_ok,
        "n_total": len(rows),
        "verdict": (f"Jacobian rank correct on {n_ok}/{len(rows)}, "
                    f"local PCA correct on {n_pca_ok}/{len(rows)}"),
        "claim": ("the Jacobian rank of a denoising autoencoder recovers the known "
                  "intrinsic dimension on flat and spherical data, and tends to "
                  "overestimate it on curved manifolds; the data-only local-PCA "
                  "baseline is reported alongside so the blame can be assigned"),
    }


def noise_sensitivity(manifold="sphere", noises=(0.0, 0.01, 0.05, 0.1, 0.2),
                      n=800, ambient=12, hidden=64, bottleneck=8, epochs=150,
                      lr=0.01, act="tanh", n_probe=16, seed=0, progress=None) -> dict:
    """At what ambient noise level does the rank signature disappear?"""
    rows = []
    for ni, nz in enumerate(noises):
        r = manifold_dimension(manifolds=(manifold,), n=n, ambient=ambient,
                               noise=float(nz), hidden=hidden, bottleneck=bottleneck,
                               epochs=epochs, lr=lr, act=act, n_probe=n_probe,
                               seed=seed)
        row = dict(r["series"][0])
        row["noise"] = float(nz)
        rows.append(row)
        if progress:
            progress((ni + 1) / len(noises), f"noise={nz}")
    return {
        "kind": "l4_noise_sensitivity",
        "config": {"manifold": manifold, "n": n, "ambient": ambient,
                   "hidden": hidden, "bottleneck": bottleneck, "epochs": epochs,
                   "seed": seed},
        "series": rows,
        "breaks_at": next((r["noise"] for r in rows if not r["correct_by_gap"]), None),
        "claim": "the rank-gap estimator has a noise level above which it lies",
    }
