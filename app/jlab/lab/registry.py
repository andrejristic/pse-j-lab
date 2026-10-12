"""The catalogue of experiments.

Defaults are chosen to finish in seconds-to-a-few-minutes on a 4 GB laptop.
Every one of them can be scaled up from the GUI; none of them needs to be.
"""

from __future__ import annotations

from . import l1_isometry, l2_ntk, l3_robustness, l4_dimension, l5_logdet

EXPERIMENTS = {
    # --- L1 -----------------------------------------------------------------
    "l1_depth_sweep": {
        "fn": l1_isometry.depth_sweep, "lab": "L1", "plot": "l1_depth_sweep",
        "title_en": "Spectrum vs depth",
        "title_sr": "Spektar u odnosu na dubinu",
        "doc": "How chi, the isometry gap and the condition number grow with depth.",
        "defaults": {"depths": [2, 4, 8, 16, 32], "width": 64, "act": "tanh",
                     "init": "gaussian", "sigma_w": 1.0, "sigma_b": 0.0,
                     "n_probe": 8, "seed": 0},
    },
    "l1_chi_critical": {
        "fn": l1_isometry.chi_critical, "lab": "L1", "plot": "l1_chi_critical",
        "title_en": "Critical line chi = 1",
        "title_sr": "Kriticna linija chi = 1",
        "doc": "Mean-field chi_1 from Gauss-Hermite quadrature vs measurement.",
        "defaults": {"act": "tanh", "sigma_b": 0.05, "width": 64, "depth": 1,
                     "n_probe": 16, "seed": 0},
    },
    "l1_free_probability": {
        "fn": l1_isometry.free_probability, "lab": "L1", "plot": "l1_freeprob",
        "title_en": "Free probability prediction",
        "title_sr": "Predikcija slobodne verovatnoce",
        "doc": "S-transform product of measured layer spectra vs measured J J^T.",
        "defaults": {"depth": 6, "width": 200, "act": "tanh", "init": "gaussian",
                     "sigma_w": 1.0, "sigma_b": 0.0, "k_max": 5, "n_probe": 4,
                     "seed": 0},
    },
    "l1_trainability": {
        "fn": l1_isometry.trainability, "lab": "L1", "plot": "l1_trainability",
        "title_en": "Does isometry help training?",
        "title_sr": "Da li izometrija pomaze treningu?",
        "doc": "Equal-budget training of deep nets from orthogonal vs Gaussian init.",
        "defaults": {"depths": [4, 16, 32], "inits": ["orthogonal", "gaussian"],
                     "width": 64, "act": "tanh", "sigma_w": 1.0, "n_train": 1000,
                     "steps": 300, "lr": 0.02, "batch": 32, "dataset": "mnist",
                     "seed": 0},
    },
    # --- L2 -----------------------------------------------------------------
    "l2_ntk_spectrum": {
        "fn": l2_ntk.ntk_spectrum, "lab": "L2", "plot": "l2_spectrum",
        "title_en": "NTK spectrum and label alignment",
        "title_sr": "NTK spektar i poravnanje sa labelama",
        "doc": "Eigenvalues of the empirical NTK; how much of it points at the labels.",
        "defaults": {"n": 128, "width": 64, "depth": 2, "act": "tanh",
                     "dataset": "mnist", "classes": [0, 1], "sigma_w": 1.4,
                     "seed": 0},
    },
    "l2_lazy_rich": {
        "fn": l2_ntk.lazy_rich, "lab": "L2", "plot": "l2_lazy_rich",
        "title_en": "Kernel drift vs width (lazy / rich)",
        "title_sr": "Pomeranje kernela u odnosu na sirinu",
        "doc": "Relative NTK movement during training; slope -1/2 means lazy.",
        "defaults": {"widths": [16, 32, 64, 128], "n": 64, "depth": 2,
                     "act": "tanh", "steps": 200, "lr": 0.5, "dataset": "mnist",
                     "classes": [0, 1], "sigma_w": 1.4, "seed": 0},
    },
    "l2_kernel_predicts_training": {
        "fn": l2_ntk.kernel_predicts_training, "lab": "L2", "plot": "l2_predicts",
        "title_en": "Does the frozen kernel predict training?",
        "title_sr": "Da li zamrznuti kernel predvidja trening?",
        "doc": "Linearised residual trajectory vs the real one, across widths.",
        "defaults": {"widths": [32, 128, 512], "n": 48, "depth": 1, "act": "tanh",
                     "steps": 150, "lr": "auto", "stability": 0.5,
                     "dataset": "mnist", "classes": [0, 1], "sigma_w": 1.4,
                     "seed": 0},
    },
    # --- L3 -----------------------------------------------------------------
    "l3_lipschitz_vs_attack": {
        "fn": l3_robustness.lipschitz_vs_attack, "lab": "L3", "plot": "l3_scatter",
        "title_en": "Local Lipschitz vs real attack distance",
        "title_sr": "Lokalni Lipschitz vs stvarni napad",
        "doc": "||J||_2 and margin/gradient estimates against PGD, per sample.",
        "defaults": {"dataset": "mnist", "n_train": 2000, "n_eval": 60,
                     "width": 64, "depth": 2, "act": "tanh", "epochs": 6,
                     "lr": 0.002, "norm": "l2", "pgd_steps": 20, "seed": 0},
    },
    "l3_jacreg_tradeoff": {
        "fn": l3_robustness.jacreg_tradeoff, "lab": "L3", "plot": "l3_tradeoff",
        "title_en": "Jacobian regularisation trade-off",
        "title_sr": "Kompromis Jakobijan-regularizacije",
        "doc": "Clean accuracy, robust accuracy and ||J||_F as lambda grows.",
        "defaults": {"dataset": "mnist", "n_train": 2000, "n_eval": 40,
                     "lambdas": [0.0, 0.01, 0.1], "width": 64, "depth": 2,
                     "act": "tanh", "epochs": 6, "lr": 0.002,
                     "attack_eps": [0.25, 0.5, 1.0, 2.0], "norm": "l2",
                     "seed": 0},
    },
    "l3_bound_looseness": {
        "fn": l3_robustness.bound_looseness, "lab": "L3", "plot": None,
        "title_en": "How loose is the certified bound?",
        "title_sr": "Koliko je sertifikovana granica labava?",
        "doc": "Product-of-norms bound vs measured max ||J||_2, before and after training.",
        "defaults": {"dataset": "mnist", "n_train": 2000, "n_eval": 40,
                     "width": 64, "depth": 3, "act": "tanh", "epochs": 6,
                     "lr": 0.002, "seed": 0},
    },
    # --- L4 -----------------------------------------------------------------
    "l4_manifold_dimension": {
        "fn": l4_dimension.manifold_dimension, "lab": "L4", "plot": "l4_spectra",
        "title_en": "Intrinsic dimension from Jacobian rank",
        "title_sr": "Unutrasnja dimenzija iz ranga Jakobijana",
        "doc": "Autoencoder Jacobian spectra on manifolds whose dimension is known.",
        "defaults": {"manifolds": ["linear", "sphere", "swiss_roll", "torus", "product"],
                     "n": 800, "ambient": 12, "noise": 0.0, "hidden": 64,
                     "bottleneck": 8, "epochs": 150, "lr": 0.01, "act": "tanh",
                     "n_probe": 24, "train_noise": 0.1, "contractive": 0.0,
                     "seed": 0},
    },
    "l4_noise_sensitivity": {
        "fn": l4_dimension.noise_sensitivity, "lab": "L4", "plot": "l4_noise",
        "title_en": "Where the dimension estimator breaks",
        "title_sr": "Gde procenitelj dimenzije puca",
        "doc": "Ambient noise level at which the rank signature stops being readable.",
        "defaults": {"manifold": "sphere", "noises": [0.0, 0.01, 0.05, 0.1, 0.2],
                     "n": 800, "ambient": 12, "hidden": 64, "bottleneck": 8,
                     "epochs": 150, "lr": 0.01, "act": "tanh", "n_probe": 16,
                     "seed": 0},
    },
    # --- L5 -----------------------------------------------------------------
    "l5_logdet_consistency": {
        "fn": l5_logdet.logdet_consistency, "lab": "L5", "plot": None,
        "title_en": "log|det J|: analytic = assembled = numeric",
        "title_sr": "log|det J|: analiticki = sklopljen = numericki",
        "doc": "The strictest engine check; a failure here voids every other lab.",
        "defaults": {"dims": [4, 6, 8], "n_layers": 4, "hidden": 32,
                     "act": "tanh", "n_points": 8, "scale_cap": 2.0, "seed": 0},
    },
    "l5_change_of_variables": {
        "fn": l5_logdet.change_of_variables, "lab": "L5", "plot": "l5_density",
        "title_en": "Change of variables against a histogram",
        "title_sr": "Promena promenljivih naspram histograma",
        "doc": "Analytic pushed-forward density vs the histogram of real samples.",
        "defaults": {"dim": 2, "n_layers": 4, "hidden": 32, "act": "tanh",
                     "n_samples": 20000, "grid": 40, "extent": 3.0,
                     "scale_cap": 1.0, "seed": 0},
    },
}

LABS = {
    "L1": {"en": "Isometry", "sr": "Izometrija"},
    "L2": {"en": "NTK", "sr": "NTK"},
    "L3": {"en": "Robustness", "sr": "Robusnost"},
    "L4": {"en": "Dimension", "sr": "Dimenzija"},
    "L5": {"en": "log|det J|", "sr": "log|det J|"},
}


def get_experiment(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; "
                       f"known: {', '.join(sorted(EXPERIMENTS))}")
    return EXPERIMENTS[name]


def list_experiments(lab: str | None = None) -> list:
    out = []
    for name, e in EXPERIMENTS.items():
        if lab and e["lab"] != lab:
            continue
        out.append({"name": name, "lab": e["lab"], "title_en": e["title_en"],
                    "title_sr": e["title_sr"], "doc": e["doc"],
                    "defaults": e["defaults"]})
    return sorted(out, key=lambda r: (r["lab"], r["name"]))
