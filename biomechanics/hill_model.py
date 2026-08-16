"""
Hill muscle model — the three force components as pure functions of
normalised fibre length and tendon strain.

This module contains no solver logic; it only defines the constitutive
relationships that make the equilibrium problem nonlinear (and therefore
expensive to invert analytically).

Three elements:
  CE  — contractile element: active, Gaussian peak at optimal length
  PEE — parallel elastic element: passive, exponential for L_M > L_M0
  SEE — series elastic element: tendon, quadratic in strain
"""

import numpy as np


def f_CE(l_norm: float, activation: float, F_max: float) -> float:
    """Active force produced by the contractile element."""
    return activation * F_max * np.exp(-((l_norm - 1.0) / 0.45) ** 2)


def f_PEE(l_norm: float, F_max: float) -> float:
    """Passive parallel elastic force (zero when fibre is shorter than L_M0)."""
    if l_norm <= 1.0:
        return 0.0
    return F_max * (np.exp(5.0 * (l_norm - 1.0)) - 1.0) / (np.exp(5.0) - 1.0)


def f_SEE(L_S: float, L_S0: float, F_max: float) -> float:
    """Tendon force — quadratic in tendon strain, zero for negative strain."""
    eps = (L_S - L_S0) / L_S0
    return F_max * 37.5 * eps ** 2 if eps > 0.0 else 0.0


def pennation_cosine(L_M: float, L_M0: float, alpha0: float) -> float:
    """
    cos(alpha) from the constant-width pennation constraint.
    For biceps longus alpha0 = 0, so this always returns 1.0.
    Included for anatomical correctness with pennate muscles.
    """
    sin_a = (L_M0 * np.sin(alpha0)) / L_M
    return float(np.sqrt(max(1.0 - sin_a ** 2, 1e-6)))
