"""
A FAIR latency comparison: classical load sharing versus the trained network.

Why the existing 7.3x figure is not adequate
--------------------------------------------
The speed-up reported throughout this project compares a Python Newton--Raphson
implementation of the Hill fibre--tendon EQUILIBRIUM solve against a PyTorch
forward pass. Two problems:

  1. It benchmarks the wrong computation. The equilibrium solve is part of
     forward simulation. The question this thesis asks --- what force must each
     muscle produce --- is answered by LOAD SHARING (static optimisation), a
     different calculation entirely. The two have been conflated.
  2. Both sides are unoptimised Python. A deployment argument cannot rest on
     that: the load-sharing problem here has nine variables and four equality
     constraints, which a direct method solves in microseconds.

This script measures the honest comparison, including the possibility that it
overturns the deployment claim.

What is timed
-------------
  * NETWORK: one policy forward pass, observation -> nine activations.
  * CLASSICAL (SLSQP): the general-purpose solver used in force_comparison.py.
  * CLASSICAL (direct): the closed-form weighted minimum-norm solution
        f = W^-1 R (R^T W^-1 R)^-1 tau,     W = diag(Fmax^2),
    which is the exact optimum of min sum (f_i/Fmax_i)^2 subject to R^T f = tau
    when the box constraints are inactive. A 4x4 solve. The box-violation rate
    is measured and reported, since the closed form is only valid where the
    constraints do not bind.
  * MOMENT ARMS: extracting R, which the classical method needs and the network
    does not. Charged to the classical side.

Run:  python bench_loadsharing.py  (run on an OTHERWISE IDLE machine)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import torch

import config
from force_comparison import moment_arm_matrix, static_optimisation

N_QUERIES = 400


def direct_loadsharing(R_s: np.ndarray, tau: np.ndarray, fmax: np.ndarray):
    """
    Closed-form minimiser of sum (f_i/Fmax_i)^2 s.t. R^T f = tau.

    With W = diag(1/Fmax^2) the objective is f^T W f; the equality-constrained
    minimiser is f = W^-1 R (R^T W^-1 R)^-1 tau. Returns (f, box_ok).
    """
    Winv = fmax ** 2                       # diagonal of W^-1
    A = R_s * Winv[:, None]                # W^-1 R   (nu x nv)
    M = R_s.T @ A                          # R^T W^-1 R  (nv x nv), here 4x4
    f = A @ np.linalg.solve(M, tau)
    return f, bool(np.all(f >= -1e-9) and np.all(f <= fmax + 1e-9))


def main():
    from rl.environment import ArmReachEnv
    from rl.evaluate import load_model

    torch.set_num_threads(1)               # a deployed device is not 8-core
    run = os.path.join(config.RUNS_DIR, "SAC_seed0")
    model_sb3, vec = load_model(run, "SAC")
    obs_rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon

    env = ArmReachEnv(domain_rand=False, seed=0)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())

    # Collect a representative set of states by rolling the policy out.
    states = []
    obs, _ = env.reset(seed=0)
    while len(states) < N_QUERIES:
        o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                    -clip, clip).astype(np.float32)
        a, _ = model_sb3.predict(o, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        d = env.data
        R = moment_arm_matrix(env.model, d)
        f_signed = d.actuator_force.copy()
        sgn = np.sign(f_signed); sgn[sgn == 0] = -1.0
        states.append((o.copy(), R * sgn[:, None], d.qfrc_actuator.copy()))
        if term or trunc:
            obs, _ = env.reset(seed=len(states))

    def timeit(fn, n=None):
        n = n or len(states)
        for i in range(20):                # warm-up
            fn(states[i % len(states)])
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(states[i % len(states)])
            ts.append((time.perf_counter() - t0) * 1e6)
        ts = np.array(ts)
        return ts.mean(), np.percentile(ts, 95), np.percentile(ts, 99)

    # --- network -----------------------------------------------------------
    def net(s):
        model_sb3.predict(s[0], deterministic=True)

    # --- classical, general-purpose solver ---------------------------------
    def slsqp(s):
        static_optimisation(s[1], s[2], fmax)

    # --- classical, direct ------------------------------------------------
    box_ok = 0
    def direct(s):
        nonlocal box_ok
        _, ok = direct_loadsharing(s[1], s[2], fmax)
        box_ok += ok

    # --- moment-arm extraction (classical-only prerequisite) ---------------
    def marms(s):
        moment_arm_matrix(env.model, env.data)

    print("=" * 78)
    print(f"LOAD-SHARING LATENCY — {N_QUERIES} queries, 1 torch thread")
    print("=" * 78)
    rows = []
    for label, fn in (("Network forward pass", net),
                      ("Classical: SLSQP (general solver)", slsqp),
                      ("Classical: direct 4x4 solve", direct),
                      ("Moment-arm extraction (classical only)", marms)):
        m, p95, p99 = timeit(fn)
        rows.append((label, m, p95, p99))
        print(f"  {label:<40}{m:>10.1f}{p95:>10.1f}{p99:>10.1f}  us")
    print(f"  {'':<40}{'mean':>10}{'p95':>10}{'p99':>10}")

    net_us = rows[0][1]
    slsqp_us = rows[1][1]
    direct_us = rows[2][1] + rows[3][1]     # direct solve + its prerequisite

    print()
    print(f"  Box constraints inactive in {100*box_ok/len(states):.1f}% of states")
    print(f"  (the closed form is exact only where they are inactive)")
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Network                                : {net_us:8.1f} us")
    print(f"  Classical, as implemented (SLSQP)      : {slsqp_us:8.1f} us  "
          f"-> network {slsqp_us/net_us:.1f}x faster")
    print(f"  Classical, direct + moment arms        : {direct_us:8.1f} us  "
          f"-> network {direct_us/net_us:.2f}x "
          f"{'faster' if direct_us > net_us else 'SLOWER'}")
    print()
    if direct_us < net_us:
        print("  The speed advantage DOES NOT SURVIVE a properly implemented")
        print("  classical solver. The deployment argument based on latency must")
        print("  be withdrawn or restated.")
    else:
        print("  The network retains an advantage even against the direct solver.")
        print("  Report THIS ratio, not the Newton-Raphson one, when the claim")
        print("  concerns load-sharing rather than forward simulation.")

    dest = os.path.join(config.RUNS_DIR, "loadsharing_latency.json")
    with open(dest, "w") as f:
        json.dump({"n_queries": N_QUERIES,
                   "rows": [{"label": l, "mean_us": m, "p95_us": p,
                             "p99_us": q} for l, m, p, q in rows],
                   "box_inactive_pct": 100.0 * box_ok / len(states),
                   "network_us": net_us,
                   "classical_slsqp_us": slsqp_us,
                   "classical_direct_total_us": direct_us}, f, indent=2)
    print(f"\nwrote {dest}")
    print("NOTE: run on an idle machine; concurrent training inflates all rows.")


if __name__ == "__main__":
    main()
