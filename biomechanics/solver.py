"""
Newton-Raphson solver for Hill muscle equilibrium.

Solves the implicit force-balance equation at a given musculo-tendinous
length L_MT and activation level a:

    (f_CE(L_M, a) + f_PEE(L_M)) * cos(alpha(L_M)) = f_SEE(L_S)

where L_S = L_MT - L_M * cos(alpha).

This is the "expensive computation" the thesis proposes to replace with RL.
Each call requires 5–40 iterations of Newton-Raphson for a single muscle,
so controlling 9 muscles in real time means 45–360 iterations per timestep.
"""

import time
import numpy as np
from biomechanics.hill_model import f_CE, f_PEE, f_SEE, pennation_cosine


def solve(L_MT: float, activation: float, params: dict,
          tol: float = 1e-6, max_iter: int = 50):
    """
    Solve muscle equilibrium for one (L_MT, activation) point.

    Returns
    -------
    L_M        : float  — equilibrium fibre length (m)
    L_S        : float  — tendon length (m)
    F          : float  — muscle force (N)
    n_iter     : int    — iterations used
    elapsed_us : float  — wall-clock time in microseconds
    """
    L_M0   = params["L_M0"]
    L_S0   = params["L_S0"]
    alpha0 = params["alpha0"]
    F_max  = params["F_max"]

    # Initial guess: clamp to physiological range
    L_M = float(np.clip(L_MT - L_S0, 0.5 * L_M0, 1.6 * L_M0))
    n_iter = max_iter

    t0 = time.perf_counter()
    for it in range(max_iter):
        cos_a  = pennation_cosine(L_M, L_M0, alpha0)
        L_S    = L_MT - L_M * cos_a
        l_norm = L_M / L_M0

        fce  = f_CE(l_norm, activation, F_max)
        fpee = f_PEE(l_norm, F_max)
        fsee = f_SEE(L_S, L_S0, F_max)

        residual = (fce + fpee) * cos_a - fsee
        if abs(residual) < tol:
            n_iter = it + 1
            break

        # Approximate Jacobian: F_max / L_M0 acts as a stiffness scale
        L_M -= 0.5 * residual / (F_max / L_M0)
        L_M  = float(np.clip(L_M, 0.5 * L_M0, 1.6 * L_M0))

    elapsed_us = (time.perf_counter() - t0) * 1e6

    # Recompute final state at converged L_M
    cos_a  = pennation_cosine(L_M, L_M0, alpha0)
    L_S    = L_MT - L_M * cos_a
    l_norm = L_M / L_M0
    F      = (f_CE(l_norm, activation, F_max) + f_PEE(l_norm, F_max)) * cos_a

    return L_M, L_S, F, n_iter, elapsed_us


def workspace_sweep(L_MT_range: np.ndarray, activations, params: dict) -> dict:
    """
    Solve equilibrium over a grid of (L_MT, activation) pairs.

    Returns a dict keyed by activation with arrays of:
      F        : muscle force (N)
      L_M      : fibre length (m)
      iters    : N-R iterations used
      time_us  : wall-clock time per solve (µs)
    """
    results = {}
    for a in activations:
        forces, lengths, iters, times = [], [], [], []
        for L_MT in L_MT_range:
            L_M, L_S, F, n, t = solve(L_MT, a, params)
            forces.append(F)
            lengths.append(L_M)
            iters.append(n)
            times.append(t)
        results[a] = {
            "F":       np.array(forces),
            "L_M":     np.array(lengths),
            "iters":   np.array(iters, dtype=int),
            "time_us": np.array(times),
        }
    return results
