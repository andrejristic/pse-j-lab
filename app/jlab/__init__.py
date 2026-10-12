"""PSE J-Lab — Jacobian geometry of small neural networks, under PSE governance.

Two Jacobians are studied here:

  * the input Jacobian   J(x) = df/dx      -- geometry, robustness, isometry
  * the parameter Jacobian J_th = df/dtheta -- NTK, training dynamics

The engine is pure numpy.  Torch, if installed, is used only as an independent
oracle: our exact Jacobian must agree with ``torch.func.jacrev`` to ~1e-10.
"""

__version__ = "0.1.0"
