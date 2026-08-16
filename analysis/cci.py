"""
Co-contraction index (CCI) from logged activation traces.

Uses the Falconer & Winter (1985) formulation applied to the elbow
agonist/antagonist muscle groups defined in config:

    CCI = 2 * I_common / (I_agonist + I_antagonist)

where each I is the time-integral of the group's mean activation over an episode,
and I_common is the integral of the pointwise minimum of the two group envelopes
(the activation shared by both, i.e. the co-contraction). CCI in [0, 1]; higher
means more simultaneous agonist/antagonist activation.

This module is the ONLY definition of CCI in the codebase — the number reported
in the thesis/paper must come from here, run on real evaluation rollouts.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config


def _episode_cci(a: np.ndarray) -> float:
    """CCI for one (T, 9) activation trace."""
    ago = a[:, config.CCI_AGONIST_IDX].mean(axis=1)      # T
    ant = a[:, config.CCI_ANTAGONIST_IDX].mean(axis=1)   # T
    common = np.minimum(ago, ant)
    denom  = np.trapz(ago) + np.trapz(ant)
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * np.trapz(common) / denom)


def analyse_cci(activation_traces) -> dict:
    """Mean +/- std CCI across episodes."""
    vals = [_episode_cci(np.asarray(t)) for t in activation_traces if len(t) > 1]
    if not vals:
        return {"error": "no usable activation traces"}
    return {
        "cci_mean":     float(np.mean(vals)),
        "cci_std":      float(np.std(vals)),
        "cci_per_ep":   [float(v) for v in vals],
        "n_episodes":   len(vals),
        "definition":   "Falconer & Winter (1985), elbow flexor vs extensor groups",
    }
