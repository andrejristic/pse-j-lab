"""The Jacobian engine: pure numpy, exact, no autodiff library required.

For a feed-forward net with pre-activations z_l and activation phi,

    J(x) = W_L D_{L-1}(x) W_{L-1} ... D_1(x) W_1,   D_l = diag(phi'(z_l))

which is exact, cheap, and -- crucially -- is literally the product of random
matrices that random matrix theory says it is.  That identity is what makes
the isometry lab (L1) possible on a 4 GB machine.
"""

from .nets import MLP, ACTIVATIONS, INITS                     # noqa: F401
from .jacobian import (                                       # noqa: F401
    input_jacobian, input_jacobian_batch, jvp, vjp,
    finite_difference_jacobian, param_jacobian,
)
from .spectra import (                                        # noqa: F401
    singular_values, chi_statistic, effective_rank,
    participation_ratio, spectrum_summary,
)
from .lipschitz import (                                      # noqa: F401
    product_norm_bound, spectral_norm, empirical_lipschitz,
)
