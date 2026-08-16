"""
Muscle-synergy analysis via non-negative matrix factorisation.

Given the per-step activation traces a(t) in [0,1]^9 logged during evaluation,
this factorises the activation matrix X (samples x 9 muscles) into k synergies
W (k x 9) and time-varying coefficients H, and reports:

  - VAF(k)                 : variance accounted for at k synergies
  - n_synergies_90         : smallest k reaching 90% VAF
  - similarity_to_reference: best-match cosine similarity of the extracted
                             synergies to a reference upper-limb synergy set

Reference synergy set
---------------------
`REFERENCE_SYNERGIES` below is a DOCUMENTED APPROXIMATION of the four upper-limb
synergies reported by d'Avella et al. (2006), expressed over THIS model's 9-muscle
ordering (config.MUSCLE_NAMES). The exact loadings in the reference paper are not
reproduced verbatim; these are qualitative agonist-grouped prototypes. The cosine
similarity is therefore a coarse correspondence check, and the paper/thesis must
describe it as such (and cite the source for the reference set) — do not report it
as an exact quantitative match without digitising the original loadings.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.decomposition import NMF
import config

# Rows = synergies, columns follow config.MUSCLE_NAMES:
# [biceps_long, biceps_short, brachialis, triceps_long, triceps_lat,
#  triceps_med, deltoid_ant, deltoid_med, deltoid_post]
REFERENCE_SYNERGIES = np.array([
    [0.9, 0.8, 0.9, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0],  # elbow flexion
    [0.0, 0.0, 0.0, 0.9, 0.8, 0.9, 0.0, 0.1, 0.2],  # elbow extension
    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.2],  # anterior shoulder
    [0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.4, 0.9],  # posterior shoulder
], dtype=float)


def _stack(activation_traces) -> np.ndarray:
    X = np.vstack([np.asarray(t) for t in activation_traces if len(t) > 0])
    return np.clip(X, 0.0, None)


def _vaf(X: np.ndarray, k: int, seed: int = 0):
    """Fit NMF with k components; return (VAF, W, H)."""
    model = NMF(n_components=k, init="nndsvda", max_iter=1000,
                random_state=seed)
    W = model.fit_transform(X)           # samples x k
    H = model.components_                 # k x muscles
    Xhat = W @ H
    vaf = 1.0 - np.sum((X - Xhat) ** 2) / (np.sum(X ** 2) + 1e-12)
    return float(vaf), W, H


def _cosine_to_reference(H: np.ndarray) -> float:
    """Mean best-match cosine similarity of extracted synergies to the reference."""
    ref = REFERENCE_SYNERGIES / (np.linalg.norm(REFERENCE_SYNERGIES, axis=1,
                                                 keepdims=True) + 1e-12)
    ext = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    sims = ext @ ref.T                    # k_ext x k_ref
    return float(np.mean(np.max(sims, axis=1)))


def analyse_synergies(activation_traces, k: int = None, seed: int = 0) -> dict:
    """
    Full synergy analysis on logged activation traces.

    Returns VAF at k synergies, the VAF curve up to 6 synergies, the number of
    synergies needed for 90% VAF, and the cosine similarity to the reference set.
    """
    if not activation_traces:
        return {"error": "no activation traces provided"}
    k = k or config.N_SYNERGIES
    X = _stack(activation_traces)

    vaf_curve = {}
    for kk in range(1, min(config.N_MUSCLES, 6) + 1):
        vaf_curve[kk], _, _ = _vaf(X, kk, seed=seed)

    n_syn_90 = next((kk for kk in sorted(vaf_curve) if vaf_curve[kk] >= 0.90),
                    config.N_MUSCLES)
    vaf_k, _, H_k = _vaf(X, k, seed=seed)

    return {
        "k":                       k,
        "vaf_at_k":                vaf_k,
        "vaf_curve":               vaf_curve,
        "n_synergies_90":          int(n_syn_90),
        "similarity_to_reference": _cosine_to_reference(H_k),
        "n_samples":               int(X.shape[0]),
        "reference":               "d'Avella et al. (2006), approximate loadings",
    }
