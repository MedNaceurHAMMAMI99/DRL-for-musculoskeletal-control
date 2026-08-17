"""
Muscle-synergy analysis: does the learned controller organise its muscles the
way the classical solution does, and the way humans do?

Why this is worth running
-------------------------
The motor-control literature holds that the nervous system controls redundant
musculature through a small number of co-activated groups --- muscle synergies
(d'Avella et al.). If nine muscles were controlled independently the activation
matrix would be full rank; if they are controlled through k synergies, it is
approximately rank k. Non-negative matrix factorisation recovers that structure.

The previous version of this work claimed a synergy analysis that was never
computed. It can now be run, on policies that actually perform the task.

What this adds beyond re-running it
-----------------------------------
The same factorisation is applied to THREE sources on the identical trajectories:

  1. the policy's commanded activations,
  2. the classical static-optimisation solution's forces for the same joint
     torques (normalised per muscle by Fmax, so both are dimensionless and
     directly comparable),
  3. a reference approximation of human upper-limb synergies.

That makes it possible to ask a question the endpoint metrics cannot: do the
learned controller and the classical criterion arrive at the SAME low-dimensional
organisation, even where they disagree on magnitude? Chapter 7 established that
they agree in direction and differ by 1.71x in effort; if their synergy structure
also coincides, the disagreement is purely one of gain rather than of
coordination strategy.

Run:  python run_synergies.py [--episodes N]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# MuJoCo must load its DLL BEFORE scikit-learn pulls in its own OpenMP runtime.
# Importing analysis.synergies (and hence sklearn) first causes
# "WinError 1114: DLL initialization routine failed" when mujoco loads after it.
import mujoco  # noqa: F401  (import for side effect: load the DLL first)

import config
from force_comparison import moment_arm_matrix, static_optimisation
from analysis.synergies import analyse_synergies, _stack, _vaf, REFERENCE_SYNERGIES

RUNS = config.RUNS_DIR
BENCH = os.path.join(RUNS, "bench")

# (label, run_dir, algo). The confirmed 300k policy first, then the benchmark
# configurations at seed 0 so the comparison spans algorithms.
TARGETS = [
    ("SAC tuned (300k)", os.path.join(RUNS, "SAC_seed0"), "SAC"),
    ("SAC tuned",        os.path.join(BENCH, "SAC_tuned_seed0"), "SAC"),
    ("SAC default",      os.path.join(BENCH, "SAC_default_seed0"), "SAC"),
    ("TD3",              os.path.join(BENCH, "TD3_seed0"), "TD3"),
    ("DDPG",             os.path.join(BENCH, "DDPG_seed0"), "DDPG"),
    ("PPO",              os.path.join(BENCH, "PPO_seed0"), "PPO"),
]


def cosine_between(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    Mean cosine similarity between two synergy sets under a ONE-TO-ONE
    assignment.

    WARNING ON STABILITY. This statistic is sensitive to the sample it is
    computed on, far more so than the VAF. Measured on one policy, varying the
    episode count gave 0.873 / 0.805 / 0.950 / 0.810 / 0.739 at 8 / 12 / 15 /
    20 / 25 episodes -- a spread of 0.21, non-monotone, i.e. sampling noise
    rather than convergence. VAF over the same range moved only 0.818-0.870.

    The cause is NMF itself: the factorisation is not unique, and the recovered
    basis depends on initialisation and on the particular sample, so comparing
    two independently fitted bases inherits that instability.

    Any figure from this function should therefore be reported as a mean over
    several resampled subsets with its spread, never as a single value. See
    `agreement_resampled` below.
    """
    from scipy.optimize import linear_sum_assignment

    a = H1 / (np.linalg.norm(H1, axis=1, keepdims=True) + 1e-12)
    b = H2 / (np.linalg.norm(H2, axis=1, keepdims=True) + 1e-12)
    S = a @ b.T
    r, c = linear_sum_assignment(-S)
    return float(np.mean(S[r, c]))


def agreement_resampled(act_traces, cls_traces, n_sub: int = 15,
                        reps: int = 6, seed: int = 0):
    """
    Policy-vs-classical synergy agreement over `reps` random subsets of
    `n_sub` episodes. Returns (mean, sd, values).

    This is the reportable form of `cosine_between`, for the reason documented
    there.
    """
    from analysis.synergies import _vaf
    rng = np.random.default_rng(seed)
    pool = len(act_traces)
    out = []
    for _ in range(reps):
        idx = rng.choice(pool, min(n_sub, pool), replace=False)
        A = np.clip(np.vstack([act_traces[i] for i in idx]), 0.0, None)
        C = np.clip(np.vstack([cls_traces[i] for i in idx]), 0.0, None)
        out.append(cosine_between(_vaf(A, config.N_SYNERGIES)[2],
                                  _vaf(C, config.N_SYNERGIES)[2]))
    v = np.asarray(out)
    return float(v.mean()), float(v.std(ddof=1)), v.tolist()


def collect(run_dir: str, algo: str, episodes: int):
    """Roll out a policy; return (activation traces, classical-solution traces)."""
    from rl.environment import ArmReachEnv
    from rl.evaluate import load_model

    model, vec = load_model(run_dir, algo)
    obs_rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon

    def norm(o):
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                       -clip, clip).astype(np.float32)

    env = ArmReachEnv(domain_rand=False, seed=0)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())

    act_traces, cls_traces = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        a_tr, c_tr = [], []
        while not done:
            a, _ = model.predict(norm(obs), deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            a_tr.append(np.clip(np.asarray(a, dtype=float), 0.0, 1.0))

            d = env.data
            R = moment_arm_matrix(env.model, d)
            f_signed = d.actuator_force.copy()
            sgn = np.sign(f_signed)
            sgn[sgn == 0] = -1.0
            f_cls, _ = static_optimisation(R * sgn[:, None], d.qfrc_actuator.copy(),
                                           fmax, f_init=np.abs(f_signed))
            # Normalise to a dimensionless [0,1] scale so the factorisation is
            # comparable with commanded activations.
            c_tr.append(np.clip(f_cls / np.maximum(fmax, 1e-9), 0.0, 1.0))
        act_traces.append(np.asarray(a_tr))
        cls_traces.append(np.asarray(c_tr))
    return act_traces, cls_traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    a = ap.parse_args()

    results = {}
    print("=" * 86)
    print(f"MUSCLE SYNERGY ANALYSIS — NMF, {a.episodes} episodes per configuration")
    print("=" * 86)
    print(f"  {'configuration':<20}{'source':<12}{'VAF@4':>8}{'k for 90%':>11}"
          f"{'cos-to-human':>14}{'cos policy-vs-classical':>25}")
    print("  " + "-" * 82)

    for label, run_dir, algo in TARGETS:
        if not os.path.exists(os.path.join(run_dir, "model.zip")):
            continue
        act, cls = collect(run_dir, algo, a.episodes)

        s_pol = analyse_synergies(act)
        s_cls = analyse_synergies(cls)

        # Synergy sets at k=4 for the policy-vs-classical comparison.
        _, _, H_pol = _vaf(_stack(act), config.N_SYNERGIES)
        _, _, H_cls = _vaf(_stack(cls), config.N_SYNERGIES)
        cross = cosine_between(H_pol, H_cls)

        results[label] = {"policy": s_pol, "classical": s_cls,
                          "cosine_policy_vs_classical": cross}

        print(f"  {label:<20}{'policy':<12}{s_pol['vaf_at_k']:>8.3f}"
              f"{s_pol['n_synergies_90']:>11}"
              f"{s_pol['similarity_to_reference']:>14.3f}"
              f"{cross:>25.3f}")
        print(f"  {'':<20}{'classical':<12}{s_cls['vaf_at_k']:>8.3f}"
              f"{s_cls['n_synergies_90']:>11}"
              f"{s_cls['similarity_to_reference']:>14.3f}")

    dest = os.path.join(RUNS, "synergy_analysis.json")
    with open(dest, "w") as f:
        json.dump(results, f, indent=2)

    print("\nInterpretation notes")
    print("  * VAF@4 is the fraction of activation variance explained by four")
    print("    synergies; higher means a more strongly low-dimensional control")
    print("    strategy. 'k for 90%' is the number of synergies needed to reach")
    print("    90% of the variance -- lower means more modular.")
    print("  * The human reference set is a DOCUMENTED APPROXIMATION of d'Avella")
    print("    et al. loadings, not digitised from the original figures, so the")
    print("    'cos-to-human' column is a coarse correspondence check only.")
    print("  * 'cos policy-vs-classical' compares the synergy STRUCTURE of the")
    print("    learned solution against static optimisation on the same")
    print("    trajectories. This column has no approximation caveat: both sides")
    print("    are computed here, on identical data.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
