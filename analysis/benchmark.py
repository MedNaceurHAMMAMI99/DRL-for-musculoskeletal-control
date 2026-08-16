"""
Computation time benchmark: Newton-Raphson solver vs SAC policy forward pass.

This is the core empirical argument of the thesis:
  - Physics-based control requires solving 9 muscle equilibria per timestep
    (9 × Newton-Raphson, 5–40 iterations each).
  - RL-based control requires one neural network forward pass per timestep
    ([256, 256] MLP, one matrix multiply chain).

Both produce control signals at each timestep, but the computational profile
is fundamentally different. The benchmark quantifies this trade-off.

Important nuance: for single-point CPU queries the NR solver can be faster
(pure NumPy arithmetic vs PyTorch overhead). The RL advantage emerges at
batch inference, on GPU, and in the amortised sense — training is expensive
once; deployment is cheap for the entire lifetime of the device.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import config
from biomechanics.solver import solve


def benchmark_solver(n_queries: int = 1000, params: dict = None, seed: int = 42) -> dict:
    """
    Time n_queries physics-solver steps.

    Each 'step' solves equilibrium for all 9 muscles simultaneously — this
    mirrors what would be required at each simulation timestep under a
    physics-based controller.
    """
    params    = params or config.BICEPS_LONGUS
    rng       = np.random.default_rng(seed)
    N_MUSCLES = 9

    L_MT_grid = rng.uniform(0.33, 0.43, (n_queries, N_MUSCLES))
    act_grid  = rng.uniform(0.0,  1.0,  (n_queries, N_MUSCLES))

    times_us = []
    for step in range(n_queries):
        t0 = time.perf_counter()
        for m in range(N_MUSCLES):
            solve(L_MT_grid[step, m], act_grid[step, m], params)
        times_us.append((time.perf_counter() - t0) * 1e6)

    arr = np.array(times_us)
    return {
        "method":    f"Newton-Raphson ({N_MUSCLES} muscles / step)",
        "n_queries": n_queries,
        "mean_us":   float(np.mean(arr)),
        "std_us":    float(np.std(arr)),
        "median_us": float(np.median(arr)),
        "p95_us":    float(np.percentile(arr, 95)),
    }


def benchmark_rl(model, envs, n_queries: int = 1000) -> dict:
    """
    Time n_queries SAC policy forward passes (obs → action).

    Each call is one control step: a 38-dim observation goes in,
    a 9-dim muscle activation vector comes out.
    """
    obs    = envs.reset()
    times  = []

    for _ in range(n_queries):
        t0 = time.perf_counter()
        model.predict(obs, deterministic=True)
        times.append((time.perf_counter() - t0) * 1e6)

    arr = np.array(times)
    return {
        "method":    "SAC policy (one forward pass)",
        "n_queries": n_queries,
        "mean_us":   float(np.mean(arr)),
        "std_us":    float(np.std(arr)),
        "median_us": float(np.median(arr)),
        "p95_us":    float(np.percentile(arr, 95)),
    }


def print_comparison(solver_stats: dict, rl_stats: dict) -> None:
    speedup = solver_stats["mean_us"] / max(rl_stats["mean_us"], 1e-9)

    print("\n" + "=" * 68)
    print("  COMPUTATION TIME BENCHMARK")
    print("=" * 68)
    header = f"  {'Method':<38} {'Mean µs':>8}  {'Median µs':>10}  {'P95 µs':>7}"
    print(header)
    print("  " + "-" * 64)
    for s in [solver_stats, rl_stats]:
        print(f"  {s['method']:<38} {s['mean_us']:>8.1f}  "
              f"{s['median_us']:>10.1f}  {s['p95_us']:>7.1f}")

    print(f"\n  Speedup (RL vs solver per step): {speedup:.1f}×")

    if speedup < 1.0:
        print("\n  Note: for single CPU queries, NumPy NR arithmetic is faster")
        print("  than the PyTorch dispatch overhead in the policy forward pass.")
        print("  The RL advantage emerges with:")
        print("    • Batch inference (many observations at once)")
        print("    • GPU deployment (network scales; NR does not)")
        print("    • Generalisation (no re-solve for new targets at runtime)")
    print()
