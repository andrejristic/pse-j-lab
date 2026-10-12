"""Figures for the runs.  Agg backend only -- these are written to disk and
displayed as images, never through an interactive canvas, so the GUI never
pays for a second rendering stack.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

def _dpi() -> int:
    """Rendering resolution for the PNGs.

    Raised from 110 to 150 because on a HiDPI screen at 200% scaling a
    770x484 image occupies about 385x242 points -- legible but soft. Bigger
    files (~90 KB instead of ~48 KB) are a fair trade for readability on the
    machine this runs on. Override with PSE_JLAB_PLOT_DPI.
    """
    import os
    try:
        return max(70, min(300, int(os.environ.get("PSE_JLAB_PLOT_DPI", "150"))))
    except ValueError:
        return 150


_STYLE = {"figure.figsize": (7.0, 4.4),
          "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
          "axes.titlesize": 10, "legend.fontsize": 8}


def _fig(outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{name}.png"


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=_dpi())
    plt.close(fig)
    return str(path)


def render(spec: str, result: dict, outdir: Path) -> list:
    fn = _PLOTS.get(spec)
    if fn is None:
        return []
    outdir = Path(outdir)
    with plt.rc_context(dict(_STYLE, **{"figure.dpi": _dpi()})):
        return fn(result, outdir)


def rerender_run(run_id: str) -> dict:
    """Redraw a finished run's figures from its sealed result.json.

    The numbers are not touched -- only the pictures are rewritten, so the
    run's sha256 (which covers result.json alone) stays valid. This is what
    lets an old run's plots be brought up to a new resolution without
    repeating the computation.
    """
    from .registry import get_experiment
    from .runner import load_run
    rec = load_run(run_id)
    if rec is None:
        return {"ok": False, "reason": "no such run"}
    if not rec.ok:
        return {"ok": False, "reason": "the run itself failed; nothing to draw"}
    spec = get_experiment(rec.name).get("plot")
    if not spec:
        return {"ok": True, "plots": [], "note": "this experiment draws no figures"}
    out = Path(rec.path) / "plots"
    try:
        paths = render(spec, rec.result, out)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "plots": paths, "dpi": _dpi()}


# --- L1 -----------------------------------------------------------------------

def _l1_depth_sweep(r, outdir):
    s = r["series"]
    d = [x["depth"] for x in s]
    fig, ax = plt.subplots(1, 2)
    ax[0].semilogy(d, [x["chi"] for x in s], "o-", label="measured chi")
    pred = {p["depth"]: p["chi_pred"] for p in r.get("predicted_chi_by_depth", [])}
    if pred:
        ax[0].semilogy(d, [pred[k] for k in d], "s--", label="mean field chi_1^L")
    ax[0].set_xlabel("depth"); ax[0].set_ylabel("chi"); ax[0].legend()
    ax[0].set_title("mean squared singular value")
    ax[1].semilogy(d, [x["isometry_gap"] for x in s], "o-", label="isometry gap")
    ax[1].semilogy(d, [x["max_over_mean"] for x in s], "^-", label="s_max / s_mean")
    ax[1].set_xlabel("depth"); ax[1].legend()
    ax[1].set_title(f"{r['config']['init']} / {r['config']['act']}")
    return [_save(fig, _fig(outdir, "depth_sweep"))]


def _l1_chi_critical(r, outdir):
    s = r["series"]
    sw = [x["sigma_w"] for x in s]
    fig, ax = plt.subplots()
    ax.plot(sw, [x["chi_1_theory"] for x in s], "-", label="mean field chi_1")
    ax.plot(sw, [x["chi_measured"] for x in s], "o", ms=3, label="measured")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    if r.get("sigma_w_critical"):
        ax.axvline(r["sigma_w_critical"], color="r", lw=0.8, ls="--",
                   label=f"critical sigma_w = {r['sigma_w_critical']:.3f}")
    ax.set_xlabel("sigma_w"); ax.set_ylabel("chi"); ax.legend()
    ax.set_title(f"criticality, {r['config']['act']}, sigma_b={r['config']['sigma_b']}")
    return [_save(fig, _fig(outdir, "chi_critical"))]


def _l1_freeprob(r, outdir):
    a = r["averaged"]
    fig, ax = plt.subplots()
    k = a["k"]
    ax.semilogy(k, a["predicted"], "o-", label="free-probability prediction")
    ax.semilogy(k, a["measured"], "s--", label="measured")
    ax.set_xlabel("moment order k"); ax.set_ylabel("m_k of J J^T")
    ax.legend(); ax.set_title("S-transform product vs measurement")
    return [_save(fig, _fig(outdir, "free_probability"))]


def _l1_trainability(r, outdir):
    fig, ax = plt.subplots()
    for row in r["series"]:
        ax.plot(row["loss_curve"], label=f"depth {row['depth']} / {row['init']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.legend()
    ax.set_title("equal budget, different initialisation")
    return [_save(fig, _fig(outdir, "trainability"))]


# --- L2 -----------------------------------------------------------------------

def _l2_spectrum(r, outdir):
    ev = np.asarray(r["eigenvalues"], dtype=float)
    ev = ev[ev > 0]
    fig, ax = plt.subplots()
    ax.loglog(np.arange(1, ev.size + 1), ev, "o-", ms=3)
    ax.set_xlabel("index"); ax.set_ylabel("NTK eigenvalue")
    ax.set_title(f"alignment with labels = {r['alignment_with_labels']:.3f}, "
                 f"eff. rank = {r['effective_rank']:.1f}")
    return [_save(fig, _fig(outdir, "ntk_spectrum"))]


def _l2_lazy_rich(r, outdir):
    s = r["series"]
    w = np.array([x["width"] for x in s], dtype=float)
    d = np.array([x["kernel_drift_rel"] for x in s], dtype=float)
    fig, ax = plt.subplots()
    ax.loglog(w, d, "o-", label="measured drift")
    if r.get("log_log_slope") is not None:
        ref = d[0] * (w / w[0]) ** -0.5
        ax.loglog(w, ref, "--", label="1/sqrt(width) reference")
    ax.set_xlabel("width"); ax.set_ylabel("||K_t - K_0|| / ||K_0||")
    ax.legend(); ax.set_title(f"fitted slope = {r.get('log_log_slope')}")
    return [_save(fig, _fig(outdir, "lazy_rich"))]


def _l2_predicts(r, outdir):
    fig, ax = plt.subplots()
    for row in r["series"]:
        line, = ax.plot(row["actual_residual"], label=f"width {row['width']} actual")
        ax.plot(row["predicted_residual"], ls="--", color=line.get_color(),
                label=f"width {row['width']} kernel")
    ax.set_xlabel("step"); ax.set_ylabel("||f - y||"); ax.legend(fontsize=7)
    ax.set_title("frozen NTK prediction vs real training")
    return [_save(fig, _fig(outdir, "kernel_predicts"))]


# --- L3 -----------------------------------------------------------------------

def _l3_scatter(r, outdir):
    rows = [x for x in r["rows"] if x["correct"] and np.isfinite(x["attack_distance"])]
    if not rows:
        return []
    fig, ax = plt.subplots(1, 2)
    ax[0].loglog([x["spectral_norm"] for x in rows],
                 [x["attack_distance"] for x in rows], "o", ms=3)
    ax[0].set_xlabel("||J(x)||_2"); ax[0].set_ylabel("PGD distance")
    ax[0].set_title(f"spearman = {r['spearman_specnorm_vs_attack']:.3f}")
    ld = [x["linear_distance"] for x in rows]
    ad = [x["attack_distance"] for x in rows]
    ax[1].loglog(ld, ad, "o", ms=3)
    lim = [min(min(ld), min(ad)), max(max(ld), max(ad))]
    ax[1].plot(lim, lim, "k:", lw=0.8)
    ax[1].set_xlabel("margin / ||grad diff||"); ax[1].set_ylabel("PGD distance")
    ax[1].set_title(f"spearman = {r['spearman_linear_vs_attack']:.3f}")
    return [_save(fig, _fig(outdir, "lipschitz_vs_attack"))]


def _l3_tradeoff(r, outdir):
    s = r["series"]
    lam = [x["jac_lambda"] for x in s]
    fig, ax = plt.subplots()
    ax.plot(lam, [x["clean_test_acc"] for x in s], "o-", label="clean test acc")
    for e in s[0].get("attack_eps", []):
        ax.plot(lam, [x["robust_acc"][str(e)] for x in s], "s-",
                label=f"robust acc @ eps={e}")
    ax.set_xlabel("jacobian penalty lambda"); ax.set_ylabel("accuracy")
    ax2 = ax.twinx()
    ax2.plot(lam, [x["mean_frobenius_J"] for x in s], "^--", color="gray",
             label="mean ||J||_F")
    ax2.set_ylabel("mean ||J||_F")
    ax.legend(loc="lower left"); ax2.legend(loc="upper right")
    ax.set_title("robustness bought with accuracy")
    return [_save(fig, _fig(outdir, "jacreg_tradeoff"))]


# --- L4 -----------------------------------------------------------------------

def _l4_spectra(r, outdir):
    fig, ax = plt.subplots()
    for row in r["series"]:
        s = np.asarray(row["median_spectrum"], dtype=float)
        s = s / max(s[0], 1e-30)
        ax.semilogy(np.arange(1, s.size + 1), s, "o-", ms=3,
                    label=f"{row['manifold']} (true {row['true_dim']}, "
                          f"est {row['gap_estimate_median']:.0f})")
    ax.set_xlabel("singular value index"); ax.set_ylabel("s_i / s_1")
    ax.legend(fontsize=7); ax.set_title("autoencoder Jacobian spectra")
    return [_save(fig, _fig(outdir, "manifold_spectra"))]


def _l4_noise(r, outdir):
    s = r["series"]
    fig, ax = plt.subplots()
    ax.plot([x["noise"] for x in s], [x["gap_estimate_median"] for x in s], "o-",
            label="estimated dim")
    ax.axhline(s[0]["true_dim"], color="k", ls=":", label="true dim")
    ax.set_xlabel("ambient noise"); ax.set_ylabel("dimension"); ax.legend()
    ax.set_title(f"estimator breaks at noise = {r.get('breaks_at')}")
    return [_save(fig, _fig(outdir, "noise_sensitivity"))]


# --- L5 -----------------------------------------------------------------------

def _l5_density(r, outdir):
    A = np.asarray(r["analytic_density"], dtype=float)
    H = np.asarray(r["empirical_density"], dtype=float)
    ext = [r["grid_x"][0], r["grid_x"][-1], r["grid_y"][0], r["grid_y"][-1]]
    fig, ax = plt.subplots(1, 2)
    for a, M, t in ((ax[0], A, "analytic (change of variables)"),
                    (ax[1], H, "empirical (histogram)")):
        im = a.imshow(M.T, origin="lower", extent=ext, aspect="auto")
        a.set_title(t, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    return [_save(fig, _fig(outdir, "change_of_variables"))]


_PLOTS = {
    "l1_depth_sweep": _l1_depth_sweep,
    "l1_chi_critical": _l1_chi_critical,
    "l1_freeprob": _l1_freeprob,
    "l1_trainability": _l1_trainability,
    "l2_spectrum": _l2_spectrum,
    "l2_lazy_rich": _l2_lazy_rich,
    "l2_predicts": _l2_predicts,
    "l3_scatter": _l3_scatter,
    "l3_tradeoff": _l3_tradeoff,
    "l4_spectra": _l4_spectra,
    "l4_noise": _l4_noise,
    "l5_density": _l5_density,
}
