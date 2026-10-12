"""Training in numpy, with the Jacobian instrumented at every checkpoint."""

from .losses import softmax_cross_entropy, mse_loss, accuracy      # noqa: F401
from .jacreg import (                                              # noqa: F401
    jacobian_penalty, jacobian_penalty_grad, frobenius_estimate,
)
from .trainer import Trainer, TrainConfig                          # noqa: F401
