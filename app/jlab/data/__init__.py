"""Data: real (MNIST / FashionMNIST) and synthetic manifolds with known
intrinsic dimension, which is what makes lab L4 falsifiable.
"""

from .idx import (                                          # noqa: F401
    load_idx, load_dataset, dataset_status, verify_manifest, subset,
)
from .manifolds import (                                    # noqa: F401
    sphere, swiss_roll, linear_subspace, torus, product_manifold, MANIFOLDS,
)
