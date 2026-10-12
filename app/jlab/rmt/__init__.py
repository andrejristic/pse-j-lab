"""Random matrix theory: what the Jacobian spectrum is *supposed* to look like.

Two independent routes to a prediction, so each can falsify the other:

  mp.py          closed forms -- Marchenko-Pastur, and Fuss-Catalan moments
                 for products of Ginibre matrices
  stransform.py  a numerical free-probability pipeline (moments -> psi ->
                 chi -> S, multiply, invert back) that takes *measured*
                 layer spectra and predicts the spectrum of their product

The S-transform pipeline is validated against Fuss-Catalan in the selftest:
if free multiplication of L Ginibre spectra does not reproduce the known
Fuss-Catalan numbers, the pipeline is wrong and its predictions are void.
"""

from .mp import (                                            # noqa: F401
    mp_pdf, mp_edges, mp_sample_eigs, esd_histogram, fuss_catalan_moments,
    ks_distance_to_mp,
)
from .stransform import (                                    # noqa: F401
    moments_from_spectrum, psi_series, compositional_inverse, s_transform,
    free_multiply, moments_from_s, predict_product_moments,
)
