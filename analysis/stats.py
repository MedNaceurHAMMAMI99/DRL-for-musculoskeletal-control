"""
Statistical comparison utilities.

Wilcoxon signed-rank test + BCa bootstrap confidence intervals.

Wilcoxon is used (rather than a t-test) because:
  - Sample sizes are small (n = 10 seeds per algorithm)
  - Performance distributions may not be Gaussian
  - It is a standard choice in RL benchmarking literature

BCa bootstrap is used for confidence intervals because it corrects for
both bias and skewness in the bootstrap distribution — important when
comparing algorithms near the ceiling (success rate close to 1.0).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.stats import wilcoxon, norm


def bootstrap_bca(data: np.ndarray, stat_fn=np.mean,
                  n_boot: int = 10_000, alpha: float = 0.05,
                  rng: np.random.Generator = None):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.
    Returns (lower, upper) at the given alpha level.
    """
    rng = rng or np.random.default_rng(0)
    n   = len(data)

    boot = np.array([stat_fn(rng.choice(data, n, replace=True))
                     for _ in range(n_boot)])

    # Bias-correction factor z0
    z0 = norm.ppf(np.clip(np.mean(boot < stat_fn(data)), 1e-6, 1 - 1e-6))

    # Acceleration factor via jackknife
    jack      = np.array([stat_fn(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jack)
    num       = np.sum((jack_mean - jack) ** 3)
    den       = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    acc       = num / den

    # BCa-adjusted percentile endpoints
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1.0 - alpha / 2)
    p_lo = float(np.clip(norm.cdf(z0 + (z0 + z_lo) / (1.0 - acc * (z0 + z_lo))), 1e-6, 1 - 1e-6))
    p_hi = float(np.clip(norm.cdf(z0 + (z0 + z_hi) / (1.0 - acc * (z0 + z_hi))), 1e-6, 1 - 1e-6))

    return float(np.percentile(boot, p_lo * 100)), float(np.percentile(boot, p_hi * 100))


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial correlation as effect size for paired Wilcoxon test.
    Range [-1, +1]; positive means x tends to be larger than y.
    """
    diffs = x - y
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks   = np.argsort(np.argsort(np.abs(nonzero))) + 1
    W_plus  = float(np.sum(ranks[nonzero > 0]))
    W_max   = len(nonzero) * (len(nonzero) + 1) / 2.0
    return 2.0 * W_plus / W_max - 1.0


def compare_algorithms(results: dict, rng_seed: int = 42):
    """
    Pairwise Wilcoxon tests (Bonferroni-corrected) + BCa CIs on mean.

    Parameters
    ----------
    results : {algorithm_name: np.ndarray of per-seed scores}

    Returns
    -------
    summaries   : {name: {"mean", "ci_lo", "ci_hi"}}
    comparisons : list of dicts with keys a, b, W, p, rb, significant, alpha_corrected
    """
    rng   = np.random.default_rng(rng_seed)
    algos = list(results.keys())
    n_cmp = len(algos) * (len(algos) - 1) // 2
    alpha_corrected = 0.05 / max(n_cmp, 1)

    summaries = {}
    for name, scores in results.items():
        lo, hi = bootstrap_bca(scores, rng=rng)
        summaries[name] = {
            "mean":  float(np.mean(scores)),
            "ci_lo": lo,
            "ci_hi": hi,
        }

    comparisons = []
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            diffs = np.asarray(results[a]) - np.asarray(results[b])
            if np.allclose(diffs, 0.0):
                # Every paired difference is zero (algorithms tie exactly on all
                # seeds). Wilcoxon is undefined here; report no significant
                # difference rather than crashing.
                W, p = 0.0, 1.0
            else:
                W, p = wilcoxon(results[a], results[b], alternative="two-sided",
                                zero_method="wilcox")
            rb   = rank_biserial(results[a], results[b])
            comparisons.append({
                "a": a, "b": b,
                "W": float(W), "p": float(p),
                "rb": float(rb),
                "significant":    bool(p < alpha_corrected),
                "alpha_corrected": alpha_corrected,
            })

    return summaries, comparisons
