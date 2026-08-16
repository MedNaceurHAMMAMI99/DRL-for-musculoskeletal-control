"""
THE THESIS QUESTION: does DRL reproduce the per-muscle forces that classical
calculation prescribes?

Everything this project has measured so far is endpoint kinematics — reaching
error, drift, path length, success rate. None of it says whether the MUSCLE
FORCES are right, which is what the thesis actually asks. A policy can reach a
target with a completely non-physiological activation pattern and score
perfectly on every existing metric (indeed, the co-contraction diagnosed in
STEPS.md §11 did exactly that).

The `force_comparison` block already in experiments_results.json does NOT answer
this. It compares MuJoCo's actuator_force against this project's Newton-Raphson
Hill solver — two forward models of the same muscle agreeing to 5.25% NRMSE.
That validates the solver. It says nothing about DRL.

The comparison, and why it is fair
----------------------------------
The classical answer to "which muscle forces produce this movement" is STATIC
OPTIMISATION (the load-sharing / muscle-redundancy problem, Crowninshield &
Brand 1981): nine muscles actuate four joints, so the mapping from joint torque
to muscle force is underdetermined and is resolved by minimising an effort
criterion.

At every timestep of a policy rollout we therefore solve

    minimise    sum_i (f_i / Fmax_i)^2
    subject to  R^T f = tau_required        (produce the SAME joint torques)
                0 <= f_i <= Fmax_i          (muscles pull only, bounded)

and compare f_classical against the f_policy MuJoCo actually produced.

`tau_required` is taken as `data.qfrc_actuator` — the generalised force the
muscles actually produced on this timestep. This is exact and avoids
inverse-dynamics differentiation noise entirely. It also guarantees the problem
is FEASIBLE, because f_policy is itself a witness satisfying the constraint.

Both methods therefore solve the identical redundancy problem on the identical
trajectory. Any disagreement is purely about load-sharing strategy — which is
precisely the scientific question.

The headline metric
-------------------
`effort_ratio` = sum(f_policy/Fmax)^2 / sum(f_classical/Fmax)^2, the factor by
which the policy exceeds the minimum effort needed to produce the very same
joint torques. A value near 1 means the policy discovered the classical
minimum-effort solution. A large value quantifies co-contraction directly, in a
way no endpoint metric can.

Documented approximation
------------------------
Fmax is peak isometric force (`actuator_gainprm[:, 2]`). The force available
from a muscle actually varies with its length and shortening velocity, and that
modulation is NOT included in the bound. This is the standard textbook
simplification for static optimisation and it is stated here rather than hidden;
it makes the classical solution slightly optimistic (it may allocate force a
muscle could not deliver at that length), which if anything biases effort_ratio
UP, i.e. against the policy. Interpretation should allow for that.

Run:  python force_comparison.py [--episodes N] [--run runs/SAC_seed0]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
from scipy.optimize import minimize

import config


def moment_arm_matrix(model, data) -> np.ndarray:
    """R (nu x nv). MuJoCo >= 3.2 stores actuator_moment sparse; densify it."""
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return R


def static_optimisation(R: np.ndarray, tau: np.ndarray, fmax: np.ndarray,
                        f_init: np.ndarray = None):
    """
    min sum (f_i/fmax_i)^2  s.t.  R^T f = tau,  0 <= f <= fmax.

    Returns (f, ok). Magnitudes only: muscle tension is non-negative here and
    the sign convention is reconciled by the caller.
    """
    n = len(fmax)
    w = 1.0 / np.maximum(fmax, 1e-9)

    def obj(f):
        return float(np.sum((f * w) ** 2))

    def jac(f):
        return 2.0 * f * w ** 2

    cons = [{"type": "eq",
             "fun": lambda f: R.T @ f - tau,
             "jac": lambda f: R.T}]
    x0 = np.clip(f_init, 0, fmax) if f_init is not None else 0.1 * fmax

    res = minimize(obj, x0, jac=jac, bounds=list(zip(np.zeros(n), fmax)),
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-9})
    return np.clip(res.x, 0.0, fmax), bool(res.success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--run", default=os.path.join(config.RUNS_DIR, "SAC_seed0"))
    ap.add_argument("--algo", default="SAC")
    args = ap.parse_args()

    from rl.environment import ArmReachEnv
    from rl.evaluate import load_model

    model_sb3, vec = load_model(args.run, args.algo)
    obs_rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon

    def norm(o):
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                       -clip, clip).astype(np.float32)

    env = ArmReachEnv(domain_rand=False, seed=0)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())

    F_pol, F_cls, A_pol = [], [], []
    tau_res_pol, tau_res_cls = [], []
    n_fail = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            a, _ = model_sb3.predict(norm(obs), deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc

            d = env.data
            R    = moment_arm_matrix(env.model, d)
            tau  = d.qfrc_actuator.copy()
            # Muscle tension is signed negative in MuJoCo; work in magnitudes and
            # fold the sign into R so R^T f reproduces the same generalised force.
            f_pol_signed = d.actuator_force.copy()
            sgn   = np.sign(f_pol_signed)
            sgn[sgn == 0] = -1.0                     # muscles pull by convention
            f_pol = np.abs(f_pol_signed)
            R_s   = R * sgn[:, None]                 # so R_s^T @ |f| == R^T @ f

            f_cls, ok = static_optimisation(R_s, tau, fmax, f_init=f_pol)
            n_fail += (not ok)

            F_pol.append(f_pol)
            F_cls.append(f_cls)
            A_pol.append(np.asarray(a, dtype=np.float64))
            tau_res_pol.append(np.linalg.norm(R_s.T @ f_pol - tau))
            tau_res_cls.append(np.linalg.norm(R_s.T @ f_cls - tau))

    F_pol = np.array(F_pol)
    F_cls = np.array(F_cls)
    A_pol = np.array(A_pol)
    n = len(F_pol)

    # ── Sanity: does R^T f_policy actually reproduce qfrc_actuator? ──────────
    # If this residual is not ~0 the moment-arm extraction or sign convention is
    # wrong and every number below is meaningless. Checked, not assumed.
    res_pol = float(np.mean(tau_res_pol))
    res_cls = float(np.mean(tau_res_cls))
    tau_scale = float(np.mean([np.linalg.norm(t) for t in [F_pol[0]]])) or 1.0

    print("=" * 78)
    print(f"FORCE COMPARISON — DRL vs static optimisation, {n:,} timesteps "
          f"({args.episodes} episodes)")
    print("=" * 78)
    print("\n0. VALIDITY CHECKS")
    print(f"   mean || R^T f_policy    - tau ||  = {res_pol:.4e} N*m  "
          f"(must be ~0: validates moment arms + sign convention)")
    print(f"   mean || R^T f_classical - tau ||  = {res_cls:.4e} N*m  "
          f"(must be ~0: the QP met its constraint)")
    print(f"   static-optimisation solver failures: {n_fail} / {n}")
    if res_pol > 1e-3:
        print("   *** WARNING: policy residual is NOT ~0. The torque identity does")
        print("   *** not hold, so the comparison below is not valid. Stop here.")

    # ── Effort: the headline ────────────────────────────────────────────────
    e_pol = np.sum((F_pol / fmax) ** 2, axis=1)
    e_cls = np.sum((F_cls / fmax) ** 2, axis=1)
    ratio = e_pol / np.maximum(e_cls, 1e-12)

    # The per-timestep MEAN of this ratio is not a usable statistic. When the
    # required torque is near zero the minimum-effort solution is near zero, so
    # the ratio diverges on those timesteps and the mean is dominated by them
    # (measured: mean 5.34x against median 1.66x on the same 2,000 samples).
    # The AGGREGATE ratio — total policy effort over total classical effort — is
    # the honest summary: it weights each timestep by how much work was actually
    # being done, and cannot be inflated by near-idle samples.
    agg = float(e_pol.sum() / max(e_cls.sum(), 1e-12))
    # Restrict the ratio DISTRIBUTION to timesteps doing non-trivial work, so
    # the quartiles describe loaded motion rather than idle hovering.
    active = e_cls > np.percentile(e_cls, 25)
    r_act = ratio[active]

    print("\n1. EFFORT — how much more than the minimum needed for the SAME torques")
    print(f"   policy      mean sum (f/Fmax)^2 : {e_pol.mean():.4f}")
    print(f"   classical   mean sum (f/Fmax)^2 : {e_cls.mean():.4f}")
    print(f"   AGGREGATE effort ratio          : {agg:.2f}x   <- headline")
    print(f"   per-step ratio, median          : {np.median(ratio):.2f}x")
    print(f"   per-step ratio, median (loaded) : {np.median(r_act):.2f}x")
    print(f"   per-step ratio, IQR (loaded)    : "
          f"{np.percentile(r_act, 25):.2f}x - {np.percentile(r_act, 75):.2f}x")
    print(f"   per-step ratio, mean            : {ratio.mean():.2f}x  "
          f"(skewed by near-idle steps — do not quote)")
    print("   (1.0 = the policy found the classical minimum-effort solution)")

    # ── Per-muscle agreement ────────────────────────────────────────────────
    print("\n2. PER-MUSCLE FORCE AGREEMENT")
    print(f"   {'muscle':<15}{'policy N':>10}{'classic N':>11}"
          f"{'RMSE N':>9}{'NRMSE%':>9}{'r':>8}")
    per_muscle = {}
    for i, name in enumerate(config.MUSCLE_NAMES):
        p, c = F_pol[:, i], F_cls[:, i]
        rmse = float(np.sqrt(np.mean((p - c) ** 2)))
        nrmse = 100.0 * rmse / max(fmax[i], 1e-9)
        r = float(np.corrcoef(p, c)[0, 1]) if p.std() > 1e-9 and c.std() > 1e-9 else float("nan")
        per_muscle[name] = {"policy_mean_N": float(p.mean()),
                            "classical_mean_N": float(c.mean()),
                            "rmse_N": rmse, "nrmse_pct_of_fmax": nrmse,
                            "pearson_r": r}
        print(f"   {name:<15}{p.mean():>10.1f}{c.mean():>11.1f}"
              f"{rmse:>9.1f}{nrmse:>9.1f}{r:>8.3f}")

    overall_rmse = float(np.sqrt(np.mean((F_pol - F_cls) ** 2)))
    overall_nrmse = 100.0 * overall_rmse / float(fmax.mean())
    r_all = float(np.corrcoef(F_pol.ravel(), F_cls.ravel())[0, 1])

    # Pattern agreement: cosine similarity per timestep. Answers "is the SHAPE
    # of the activation pattern the same", independently of its magnitude —
    # a policy could scale every muscle up uniformly and still be coordinating
    # correctly.
    num = np.sum(F_pol * F_cls, axis=1)
    den = np.linalg.norm(F_pol, axis=1) * np.linalg.norm(F_cls, axis=1)
    cos = num / np.maximum(den, 1e-12)

    # NULL BASELINE — cosine between non-negative vectors has a high floor and
    # must never be reported without it. Muscle forces are all >= 0, so two
    # unrelated force vectors already agree strongly by construction: random
    # non-negative 9-vectors score ~0.76. The null used here is stronger than
    # random: it keeps each timestep's ACTUAL policy magnitudes and only permutes
    # which muscle they belong to, destroying the correspondence while preserving
    # the magnitude distribution exactly. Any excess over this null is the real
    # signal.
    rng = np.random.default_rng(0)
    null = np.empty(n)
    for t in range(n):
        p = rng.permutation(F_pol[t])
        null[t] = float(p @ F_cls[t] /
                        max(np.linalg.norm(p) * np.linalg.norm(F_cls[t]), 1e-12))
    null_mean, null_p95 = float(null.mean()), float(np.percentile(null, 95))
    excess = float(cos.mean() - null_mean)

    print(f"\n   overall RMSE {overall_rmse:.1f} N  "
          f"({overall_nrmse:.1f}% of mean Fmax),  pearson r = {r_all:.3f}")
    print(f"   pattern cosine  : mean {cos.mean():.3f}, median {np.median(cos):.3f}")
    print(f"   NULL (shuffled) : mean {null_mean:.3f}, 95th pct {null_p95:.3f}")
    print(f"   excess over null: {excess:+.3f}   "
          f"{'(mean cosine does NOT clear the null 95th pct)' if cos.mean() < null_p95 else ''}")
    print(f"   -> Pearson r is the mean-centred measure and has no such floor;")
    print(f"      prefer it when quoting a single agreement figure.")

    print("\n3. INTERPRETATION")
    if res_pol > 1e-3:
        verdict = "INVALID — torque identity failed"
        print("   Not interpretable; see the validity check above.")
    elif agg < 1.5 and cos.mean() > 0.9:
        verdict = "AGREES"
        print("   The policy reproduces the classical load-sharing solution.")
        print("   DRL is a viable substitute for static optimisation here, and")
        print("   the measured latency advantage (7.3x) is the practical payoff.")
    elif cos.mean() > 0.8:
        verdict = "PATTERN AGREES, MAGNITUDE DOES NOT"
        print("   The COORDINATION pattern matches but the policy uses")
        print(f"   {agg:.1f}x the necessary effort — i.e. it recruits the")
        print("   right muscles in the right proportions, then co-contracts on top.")
        print("   DRL captures the synergy structure but not the effort optimum.")
    else:
        verdict = "DISAGREES"
        print(f"   The policy's load sharing differs from the classical solution")
        print(f"   (pattern cosine {cos.mean():.2f}, effort {agg:.1f}x).")
        print("   It achieves the movement by a different distribution of force.")
        print("   DRL is NOT a drop-in replacement for static optimisation here —")
        print("   which is a real finding, not a failure: it shows the redundancy")
        print("   problem admits solutions the effort criterion does not select.")

    out = {
        "run": args.run, "episodes": args.episodes, "timesteps": n,
        "verdict": verdict,
        "validity": {"mean_residual_policy_Nm": res_pol,
                     "mean_residual_classical_Nm": res_cls,
                     "solver_failures": int(n_fail)},
        "effort": {"policy_mean": float(e_pol.mean()),
                   "classical_mean": float(e_cls.mean()),
                   "ratio_aggregate": agg,               # the reportable one
                   "ratio_median": float(np.median(ratio)),
                   "ratio_median_loaded": float(np.median(r_act)),
                   "ratio_iqr_loaded": [float(np.percentile(r_act, 25)),
                                        float(np.percentile(r_act, 75))],
                   "ratio_mean_SKEWED_DO_NOT_QUOTE": float(ratio.mean())},
        "overall": {"rmse_N": overall_rmse, "nrmse_pct": overall_nrmse,
                    "pearson_r": r_all,
                    "pattern_cosine_mean": float(cos.mean()),
                    "pattern_cosine_median": float(np.median(cos)),
                    "pattern_cosine_null_mean": null_mean,
                    "pattern_cosine_null_p95": null_p95,
                    "pattern_cosine_excess_over_null": excess},
        "per_muscle": per_muscle,
        "note": ("tau taken as data.qfrc_actuator (exact, no inverse-dynamics "
                 "noise; guarantees feasibility since f_policy is a witness). "
                 "Fmax is peak isometric force; force-length-velocity modulation "
                 "of the available force is NOT in the bound."),
    }
    dest = os.path.join(args.run, "force_comparison_drl_vs_classical.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
