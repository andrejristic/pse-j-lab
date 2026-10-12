"""L3 -- the input Jacobian and adversarial robustness.

The folklore: a small local Lipschitz constant ||J(x)||_2 means a robust point.
The sharper linear theory says something more specific -- the distance to the
decision boundary is approximately

    margin(x) / || grad(f_y - f_k)(x) ||

which is a *row difference* of J, not its top singular value.  Both are tested
against a real attack here, because the interesting result is where the linear
picture stops being true, and that only shows up if you measure it.
"""

from __future__ import annotations

import numpy as np

from ..jac.jacobian import input_jacobian
from ..jac.lipschitz import product_norm_bound, spectral_norm
from ..jac.nets import MLP
from ..train.losses import accuracy, softmax_cross_entropy


# --- attacks ------------------------------------------------------------------

def _input_grad(net: MLP, x: np.ndarray, y: int) -> np.ndarray:
    out, cache = net.forward(x.reshape(1, -1), cache=True)
    _, dout = softmax_cross_entropy(out, np.array([y]))
    _, _, dx = net.backward(cache, dout)
    return dx.reshape(-1)


def pgd(net: MLP, x: np.ndarray, y: int, eps: float, steps: int = 20,
        norm: str = "l2", alpha: float | None = None,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Projected gradient descent inside an eps-ball.  Returns x_adv."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    rng = rng or np.random.default_rng(0)
    delta = rng.standard_normal(x.size) * 1e-4
    for _ in range(steps):
        g = _input_grad(net, x + delta, y)
        if norm == "linf":
            delta = delta + alpha * np.sign(g)
            delta = np.clip(delta, -eps, eps)
        else:
            gn = np.linalg.norm(g)
            if gn > 0:
                delta = delta + alpha * g / gn
            dn = np.linalg.norm(delta)
            if dn > eps:
                delta = delta * (eps / dn)
    return x + delta


def minimal_perturbation(net: MLP, x: np.ndarray, y: int, hi: float = 8.0,
                         tol: float = 0.02, steps: int = 20, norm: str = "l2",
                         rng=None) -> float:
    """Smallest eps (by bisection) at which PGD flips the prediction.

    An upper bound on the true minimal distance: PGD may fail where an attack
    exists.  Reported as such -- never called "the" distance.
    """
    lo = 0.0
    if int(np.argmax(net.forward(x.reshape(1, -1))[0])) != y:
        return 0.0
    found = None
    e = hi
    for _ in range(24):
        xa = pgd(net, x, y, e, steps=steps, norm=norm, rng=rng)
        if int(np.argmax(net.forward(xa.reshape(1, -1))[0])) != y:
            found = e
            break
        e *= 2.0
        if e > 1e4:
            return float("inf")
    if found is None:
        return float("inf")
    hi = found
    while hi - lo > tol * max(hi, 1e-9):
        mid = 0.5 * (lo + hi)
        xa = pgd(net, x, y, mid, steps=steps, norm=norm, rng=rng)
        if int(np.argmax(net.forward(xa.reshape(1, -1))[0])) != y:
            hi = mid
        else:
            lo = mid
    return float(hi)


def linear_distance_estimate(net: MLP, x: np.ndarray, y: int) -> dict:
    """margin / ||grad(f_y - f_k)|| minimised over the runner-up classes."""
    J = input_jacobian(net, x)
    f = net.forward(x.reshape(1, -1))[0]
    best, best_k = np.inf, None
    for k in range(f.size):
        if k == y:
            continue
        g = J[y] - J[k]
        gn = np.linalg.norm(g)
        if gn <= 0:
            continue
        d = (f[y] - f[k]) / gn
        if d < best:
            best, best_k = float(d), int(k)
    return {"linear_distance": best, "runner_up": best_k,
            "spectral_norm": float(spectral_norm(J)),
            "frobenius": float(np.sqrt((J * J).sum()))}


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / den) if den > 0 else float("nan")


# --- experiments --------------------------------------------------------------

def _train_small(dataset, n_train, width, depth, act, epochs, lr, jac_lambda,
                 seed, jac_nproj=1):
    from ..data.idx import load_dataset, subset
    from ..train.trainer import TrainConfig, Trainer
    d = load_dataset(dataset, "train", n=max(4 * n_train, 4000), dtype=np.float64)
    d = subset(d, n_train, seed=seed)
    dt = load_dataset(dataset, "test", n=2000, dtype=np.float64)
    net = MLP(dims=[d["X"].shape[1]] + [width] * depth + [10], act=act,
              init="orthogonal", sigma_w=1.2, readout_linear=True, seed=seed)
    cfg = TrainConfig(epochs=epochs, batch_size=64, lr=lr, opt="adam",
                      jac_lambda=jac_lambda, jac_nproj=jac_nproj, seed=seed,
                      probe_n=8)
    res = Trainer(net, cfg).fit(d["X"], d["y"], dt["X"], dt["y"])
    return net, d, dt, res


def lipschitz_vs_attack(dataset="mnist", n_train=2000, n_eval=60, width=64,
                        depth=2, act="tanh", epochs=6, lr=0.002, norm="l2",
                        pgd_steps=20, seed=0, progress=None) -> dict:
    """Per-sample: ||J||_2, the linear distance estimate, and a real attack."""
    net, d, dt, res = _train_small(dataset, n_train, width, depth, act, epochs,
                                   lr, 0.0, seed)
    if progress:
        progress(0.4, "trained; attacking")
    rng = np.random.default_rng(seed + 5)
    X, y = dt["X"][:n_eval], dt["y"][:n_eval]
    rows = []
    for i in range(X.shape[0]):
        pred = int(np.argmax(net.forward(X[i].reshape(1, -1))[0]))
        est = linear_distance_estimate(net, X[i], int(y[i]))
        dist = minimal_perturbation(net, X[i], int(y[i]), norm=norm,
                                    steps=pgd_steps, rng=rng)
        rows.append({"i": i, "label": int(y[i]), "pred": pred,
                     "correct": pred == int(y[i]),
                     "spectral_norm": est["spectral_norm"],
                     "frobenius": est["frobenius"],
                     "linear_distance": est["linear_distance"],
                     "attack_distance": dist})
        if progress and i % 5 == 0:
            progress(0.4 + 0.6 * (i + 1) / X.shape[0], f"sample {i+1}/{X.shape[0]}")
    ok = [r for r in rows if r["correct"] and np.isfinite(r["attack_distance"])
          and r["attack_distance"] > 0]
    sn = [r["spectral_norm"] for r in ok]
    ld = [r["linear_distance"] for r in ok]
    ad = [r["attack_distance"] for r in ok]
    ratio = [a / l for a, l in zip(ad, ld) if l > 0]
    return {
        "kind": "l3_lipschitz_vs_attack",
        "config": {"dataset": dataset, "n_train": n_train, "n_eval": n_eval,
                   "width": width, "depth": depth, "act": act, "epochs": epochs,
                   "lr": lr, "norm": norm, "pgd_steps": pgd_steps, "seed": seed},
        "test_acc": res.final["val"]["acc"],
        "n_usable": len(ok),
        "rows": rows,
        "spearman_specnorm_vs_attack": _spearman(sn, ad),
        "spearman_linear_vs_attack": _spearman(ld, ad),
        "attack_over_linear_median": float(np.median(ratio)) if ratio else None,
        "attack_over_linear_iqr": ([float(np.quantile(ratio, 0.25)),
                                    float(np.quantile(ratio, 0.75))]
                                   if ratio else None),
        "claim": ("the margin/gradient-difference estimate ranks robustness far "
                  "better than ||J||_2 alone; both are only local"),
    }


def jacreg_tradeoff(dataset="mnist", n_train=2000, n_eval=40, lambdas=(0.0, 0.01, 0.1),
                    width=64, depth=2, act="tanh", epochs=6, lr=0.002,
                    attack_eps=(0.25, 0.5, 1.0, 2.0), norm="l2", seed=0,
                    progress=None) -> dict:
    """Accuracy versus robustness as the Jacobian penalty is turned up.

    Robust accuracy is measured at several attack strengths, not one: at a
    single eps the answer is usually 100% (eps too small to matter) or 0%
    (eps large enough to break everything), and both hide the effect being
    measured.  The median attack distance is the summary that does not depend
    on choosing eps at all.
    """
    if isinstance(attack_eps, (int, float)):
        attack_eps = (float(attack_eps),)
    attack_eps = [float(e) for e in attack_eps]
    rows = []
    for li, lam in enumerate(lambdas):
        net, d, dt, res = _train_small(dataset, n_train, width, depth, act,
                                       epochs, lr, float(lam), seed, jac_nproj=1)
        rng = np.random.default_rng(seed + 3)
        X, y = dt["X"][:n_eval], dt["y"][:n_eval]
        adv_ok = {e: 0 for e in attack_eps}
        fro, dists = [], []
        for i in range(X.shape[0]):
            J = input_jacobian(net, X[i])
            fro.append(float(np.sqrt((J * J).sum())))
            for e in attack_eps:
                xa = pgd(net, X[i], int(y[i]), e, norm=norm, rng=rng)
                if int(np.argmax(net.forward(xa.reshape(1, -1))[0])) == int(y[i]):
                    adv_ok[e] += 1
            dists.append(minimal_perturbation(net, X[i], int(y[i]), norm=norm,
                                              steps=15, rng=rng))
        finite = [x for x in dists if np.isfinite(x)]
        rows.append({
            "jac_lambda": float(lam),
            "clean_test_acc": res.final["val"]["acc"],
            "clean_train_acc": res.final["train"]["acc"],
            "robust_acc": {str(e): adv_ok[e] / max(X.shape[0], 1) for e in attack_eps},
            "attack_eps": attack_eps,
            "mean_frobenius_J": float(np.mean(fro)),
            "median_attack_distance": float(np.median(finite)) if finite else None,
            "mean_attack_distance": float(np.mean(finite)) if finite else None,
            "n_unattackable": int(len(dists) - len(finite)),
            "certified_bound": product_norm_bound(net)["bound"],
            "seconds": res.seconds,
        })
        if progress:
            progress((li + 1) / len(lambdas), f"lambda={lam}")
    return {
        "kind": "l3_jacreg_tradeoff",
        "config": {"dataset": dataset, "n_train": n_train, "n_eval": n_eval,
                   "width": width, "depth": depth, "act": act, "epochs": epochs,
                   "lr": lr, "norm": norm, "seed": seed},
        "series": rows,
        "claim": ("penalising ||J||_F buys robustness and costs clean accuracy; "
                  "the exchange rate is measured here, not assumed"),
    }


def bound_looseness(dataset="mnist", n_train=2000, n_eval=40, width=64, depth=3,
                    act="tanh", epochs=6, lr=0.002, seed=0, progress=None) -> dict:
    """How many orders of magnitude the certified Lipschitz bound overshoots."""
    from ..data.idx import load_dataset
    net_init = MLP(dims=[784] + [width] * depth + [10], act=act, init="orthogonal",
                   sigma_w=1.2, readout_linear=True, seed=seed)
    dt = load_dataset(dataset, "test", n=n_eval, dtype=np.float64)
    before_emp = max(spectral_norm(input_jacobian(net_init, dt["X"][i]))
                     for i in range(min(n_eval, dt["X"].shape[0])))
    before = product_norm_bound(net_init)
    if progress:
        progress(0.3, "training")
    net, d, dt2, res = _train_small(dataset, n_train, width, depth, act, epochs,
                                    lr, 0.0, seed)
    after_emp = max(spectral_norm(input_jacobian(net, dt2["X"][i]))
                    for i in range(min(n_eval, dt2["X"].shape[0])))
    after = product_norm_bound(net)
    return {
        "kind": "l3_bound_looseness",
        "config": {"dataset": dataset, "n_train": n_train, "n_eval": n_eval,
                   "width": width, "depth": depth, "act": act, "epochs": epochs,
                   "seed": seed},
        "before_training": {"bound": before["bound"], "empirical_max": float(before_emp),
                            "log10_ratio": float(np.log10(before["bound"] / before_emp))
                            if before_emp > 0 else None,
                            "layer_norms": before["layer_spectral_norms"]},
        "after_training": {"bound": after["bound"], "empirical_max": float(after_emp),
                           "log10_ratio": float(np.log10(after["bound"] / after_emp))
                           if after_emp > 0 else None,
                           "layer_norms": after["layer_spectral_norms"]},
        "test_acc": res.final["val"]["acc"],
        "claim": ("how far the product-of-norms certificate sits above the "
                  "measured maximum, before and after training -- the size of "
                  "the gap is the result, and at small depth it can be nearly "
                  "tight"),
    }
