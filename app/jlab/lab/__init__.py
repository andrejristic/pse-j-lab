"""The five laboratories.

  L1  isometry     spectrum of the input Jacobian vs depth, init, activation
  L2  ntk          parameter Jacobian: kernel spectrum, lazy/rich drift
  L3  robustness   ||J||_2 vs adversarial distance; Jacobian regularisation
  L4  dimension    Jacobian rank as an intrinsic-dimension estimator
  L5  logdet       exact log|det J| in a coupling flow, three-way cross-check

L4 and L5 exist as much for validation as for science: they are the two labs
whose correct answer is known in advance, so a wrong engine cannot pass them.
"""

from .registry import EXPERIMENTS, get_experiment, list_experiments   # noqa: F401
from .runner import run, RunRecord, load_run, list_runs               # noqa: F401
