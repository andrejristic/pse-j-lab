"""Seed content: the topic spine, the folklore worth attacking, and a list of
*proposed* citations that are deliberately not cards yet.

Note what is missing here: CITED cards. A citation is only admitted after the
reference resolves against arXiv or Crossref (see llm/refs.py), because a
plausible-looking arXiv id recalled from memory is exactly the kind of thing
this vault exists to keep out. Until they resolve they live in
reference_proposals.json and the GUI shows them as unverified.
"""

from __future__ import annotations

import json

from ..core import config, journal
from .cards import CardKind
from .gate import GateError, admit

TOPICS = [
    {"id": "tema-osnove", "sr": "Sta je Jakobijan mreze",
     "en": "What the Jacobian of a network is",
     "blurb": "Input Jacobian df/dx vs parameter Jacobian df/dtheta; why the "
              "two answer different questions."},
    {"id": "tema-izometrija", "sr": "Dinamicka izometrija",
     "en": "Dynamical isometry",
     "blurb": "Spectrum of J near 1 in a deep net; chi=1 criticality; "
              "orthogonal vs Gaussian initialisation.", "lab": "L1"},
    {"id": "tema-rmt", "sr": "Slucajne matrice i slobodna verovatnoca",
     "en": "Random matrices and free probability",
     "blurb": "Marchenko-Pastur, products of random matrices, S-transform, "
              "Fuss-Catalan.", "lab": "L1"},
    {"id": "tema-ntk", "sr": "Neuralni tangentni kernel",
     "en": "Neural tangent kernel",
     "blurb": "Theta = J_theta J_theta^T, its spectrum and its alignment with "
              "the labels.", "lab": "L2"},
    {"id": "tema-lenj-bogat", "sr": "Lenji i bogati rezim",
     "en": "Lazy and rich regimes",
     "blurb": "How far the kernel moves during training, and what width has to "
              "do with it.", "lab": "L2"},
    {"id": "tema-lipschitz", "sr": "Lipschitz i sertifikati",
     "en": "Lipschitz constants and certificates",
     "blurb": "Product-of-norms bounds, local vs global, how loose the "
              "certificate is in practice.", "lab": "L3"},
    {"id": "tema-adversarial", "sr": "Adversarijalna robusnost",
     "en": "Adversarial robustness",
     "blurb": "Margin over gradient norm as a distance estimate; where the "
              "linear picture stops holding.", "lab": "L3"},
    {"id": "tema-jacreg", "sr": "Jakobijan-regularizacija",
     "en": "Jacobian regularisation",
     "blurb": "Penalising ||J||_F, the random-projection estimator, and the "
              "accuracy it costs.", "lab": "L3"},
    {"id": "tema-dimenzija", "sr": "Unutrasnja dimenzija podataka",
     "en": "Intrinsic dimension of data",
     "blurb": "Manifold hypothesis; Jacobian rank of a denoising autoencoder "
              "vs local PCA.", "lab": "L4"},
    {"id": "tema-tokovi", "sr": "Normalizujuci tokovi i log|det J|",
     "en": "Normalising flows and log|det J|",
     "blurb": "Change of variables; why coupling layers make the determinant "
              "cheap.", "lab": "L5"},
    {"id": "tema-fisher", "sr": "Fisher, Gauss-Newton i prirodni gradijent",
     "en": "Fisher, Gauss-Newton and natural gradient",
     "blurb": "J^T H J as the curvature that second-order methods actually "
              "use; relation to the NTK."},
    {"id": "tema-implicitno", "sr": "Implicitni slojevi",
     "en": "Implicit layers",
     "blurb": "Implicit function theorem: differentiating a fixed point "
              "without unrolling it."},
    {"id": "tema-folklor", "sr": "Folklor koji treba oboriti",
     "en": "Folklore that should be attacked",
     "blurb": "Claims repeated without a source. Each one here is phrased so "
              "that a lab in this app can wound it."},
]

# Each of these is a real, commonly repeated claim, phrased so that one of the
# five labs can actually test it. They enter the vault as FOLKLORE -- neither
# endorsed nor dismissed -- and the point of the trip is to move them.
FOLKLORE = [
    {"claim": "A small local Lipschitz constant ||J(x)||_2 implies the point is "
              "hard to attack.",
     "topic": "tema-folklor", "tags": ["L3", "attackable"],
     "body": "Testable by l3_lipschitz_vs_attack: rank-correlate ||J||_2 with a "
             "real PGD distance and compare against the margin/gradient estimate."},
    {"claim": "Setting chi = 1 at initialisation is enough for a very deep "
              "network to be trainable.",
     "topic": "tema-folklor", "tags": ["L1", "attackable"],
     "body": "Testable by l1_depth_sweep: chi can sit at 1 while the condition "
             "number of J grows exponentially with depth."},
    {"claim": "The neural tangent kernel is constant during training.",
     "topic": "tema-folklor", "tags": ["L2", "attackable"],
     "body": "Testable by l2_lazy_rich: it is only asymptotically constant, and "
             "the drift at finite width is measurable."},
    {"claim": "The number of large singular values of a trained autoencoder's "
              "Jacobian equals the intrinsic dimension of the data.",
     "topic": "tema-folklor", "tags": ["L4", "attackable"],
     "body": "Testable by l4_manifold_dimension on manifolds with known "
             "dimension, against a local-PCA baseline."},
    {"claim": "The product of layer spectral norms is a useful estimate of a "
              "network's Lipschitz constant.",
     "topic": "tema-folklor", "tags": ["L3", "attackable"],
     "body": "Testable by l3_bound_looseness: measure the ratio to the observed "
             "maximum before and after training."},
    {"claim": "In a normalising flow the log-determinant is free, so it need "
              "not be checked.",
     "topic": "tema-folklor", "tags": ["L5", "attackable"],
     "body": "Testable by l5_logdet_consistency: analytic vs assembled vs "
             "numeric. It is free, but only if the implementation is right."},
]

# Proposed citations. Fields marked 'unverified' until the reference resolves.
REFERENCE_PROPOSALS = [
    {"key": "ntk", "arxiv": "1806.07572", "topic": "tema-ntk",
     "title_guess": "Neural Tangent Kernel: Convergence and Generalization in "
                    "Neural Networks",
     "claim": "The NTK of an infinitely wide network is constant during "
              "gradient-flow training."},
    {"key": "lazy", "arxiv": "1812.07956", "topic": "tema-lenj-bogat",
     "title_guess": "On Lazy Training in Differentiable Programming",
     "claim": "Lazy training is a scaling phenomenon, not a property of depth."},
    {"key": "wide-linear", "arxiv": "1902.06720", "topic": "tema-lenj-bogat",
     "title_guess": "Wide Neural Networks of Any Depth Evolve as Linear Models "
                    "Under Gradient Descent",
     "claim": "Wide networks follow their linearisation during training."},
    {"key": "saxe", "arxiv": "1312.6120", "topic": "tema-izometrija",
     "title_guess": "Exact solutions to the nonlinear dynamics of learning in "
                    "deep linear networks",
     "claim": "Orthogonal initialisation gives depth-independent learning times "
              "in deep linear networks."},
    {"key": "chaos", "arxiv": "1606.05340", "topic": "tema-izometrija",
     "title_guess": "Exponential expressivity in deep neural networks through "
                    "transient chaos",
     "claim": "There is an order-to-chaos transition governed by sigma_w."},
    {"key": "info-prop", "arxiv": "1611.01232", "topic": "tema-izometrija",
     "title_guess": "Deep Information Propagation",
     "claim": "Trainable depth is set by the correlation depth scale at "
              "initialisation."},
    {"key": "jacreg", "arxiv": "1908.02729", "topic": "tema-jacreg",
     "title_guess": "Robust Learning with Jacobian Regularization",
     "claim": "A random-projection estimator makes the Frobenius Jacobian "
              "penalty cheap enough to train with."},
    {"key": "realnvp", "arxiv": "1605.08803", "topic": "tema-tokovi",
     "title_guess": "Density estimation using Real NVP",
     "claim": "Coupling layers give a triangular Jacobian, so the determinant "
              "is the product of the scales."},
    {"key": "intriguing", "arxiv": "1312.6199", "topic": "tema-adversarial",
     "title_guess": "Intriguing properties of neural networks",
     "claim": "Imperceptible perturbations can change a network's prediction."},
    {"key": "fgsm", "arxiv": "1412.6572", "topic": "tema-adversarial",
     "title_guess": "Explaining and Harnessing Adversarial Examples",
     "claim": "Adversarial examples arise from locally linear behaviour."},
    {"key": "pgd", "arxiv": "1706.06083", "topic": "tema-adversarial",
     "title_guess": "Towards Deep Learning Models Resistant to Adversarial "
                    "Attacks",
     "claim": "PGD is a strong first-order adversary."},
    {"key": "deepfool", "arxiv": "1511.04599", "topic": "tema-adversarial",
     "title_guess": "DeepFool: a simple and accurate method to fool deep neural "
                    "networks",
     "claim": "The distance to the boundary is well approximated by "
              "margin / gradient norm, iterated."},
    {"key": "sensitivity", "arxiv": "1802.08760", "topic": "tema-osnove",
     "title_guess": "Sensitivity and Generalization in Neural Networks: an "
                    "Empirical Study",
     "claim": "The input-Jacobian norm correlates with generalisation across "
              "trained models."},
]


def topics_path():
    return config.vault_dir() / "topics.json"


def proposals_path():
    return config.vault_dir() / "reference_proposals.json"


def seed_vault(force: bool = False) -> dict:
    """Write the topic spine and admit the folklore cards.  Idempotent."""
    tp = topics_path()
    if force or not tp.exists():
        tp.write_text(json.dumps(TOPICS, indent=2, ensure_ascii=False))
    pp = proposals_path()
    if force or not pp.exists():
        pp.write_text(json.dumps(
            [dict(r, status="unverified") for r in REFERENCE_PROPOSALS],
            indent=2, ensure_ascii=False))
    admitted, skipped = [], []
    for f in FOLKLORE:
        try:
            c = admit(CardKind.FOLKLORE, f["claim"], topic=f["topic"],
                      body=f.get("body", ""), tags=f.get("tags", []),
                      author="seed")
            admitted.append(c.id)
        except GateError as e:
            skipped.append(str(e))
    journal.log("VAULT_SEED", {"topics": len(TOPICS), "admitted": len(admitted),
                               "skipped": len(skipped),
                               "reference_proposals": len(REFERENCE_PROPOSALS)},
                actor="vault")
    return {"topics": len(TOPICS), "admitted": admitted, "skipped": skipped,
            "reference_proposals": len(REFERENCE_PROPOSALS)}


def load_topics() -> list:
    p = topics_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return list(TOPICS)


def load_proposals() -> list:
    p = proposals_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return [dict(r, status="unverified") for r in REFERENCE_PROPOSALS]


def save_proposals(rows: list) -> None:
    proposals_path().write_text(json.dumps(rows, indent=2, ensure_ascii=False))
