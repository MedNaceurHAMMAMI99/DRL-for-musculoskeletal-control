"""
The missing baseline: a CONVENTIONAL controller solving the same reaching task.

Why this exists
---------------
Chapter 7 compares the learned forces against static optimisation, but static
optimisation is only ever asked the load-sharing sub-question: tau is taken as
the torque the POLICY produced, so the classical method never chooses a
trajectory of its own. The thesis question --- can DRL replace conventional
calculation --- requires a conventional controller that solves the whole task,
and no such row existed in the results.

`rl/baselines.py` already implements one: a Cartesian PD law producing a desired
end-effector wrench, mapped through the Jacobian transpose to joint torques, then
distributed onto the muscles by non-negative least squares on the moment-arm
matrix. That is impedance control plus classical load sharing --- a conventional
pipeline, requiring no training.

This script evaluates it under EXACTLY the protocol used for the learned
policies (same episodes, same metrics, same seeds), then runs the same
force-comparison analysis on it.

What to expect, and why it is informative either way
----------------------------------------------------
The PD controller distributes force by solving a non-negative least-squares
problem, so its effort ratio against static optimisation should be close to 1 by
construction. That makes it the natural UPPER BOUND on the effort axis --- the
reference the learned policy's 1.71x should be judged against, rather than
against an abstract optimum.

The open question is accuracy. If the conventional controller also reaches
accurately, the case for DRL weakens considerably and the thesis must say so. If
it reaches poorly, the result is a clean trade-off: conventional control is
efficient but imprecise on this model, DRL is precise but wasteful.

Caveat on prior numbers
-----------------------
The PD baseline's July results are not trustworthy: it crashed on MuJoCo >= 3.2
because `actuator_moment` became sparse, and the densification fix postdates
them. Everything here is measured fresh.

Run:  python run_conventional.py [--episodes 50]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

import config
from force_comparison import moment_arm_matrix, static_optimisation


def static_optimisation_bounded(R: np.ndarray, tau: np.ndarray,
                                lo: np.ndarray, hi: np.ndarray):
    """
    min sum ((f_i - lo_i)/(hi_i - lo_i))^2  s.t.  R^T f = tau,  lo <= f <= hi.

    The effort criterion is expressed over the ACTIVATION-equivalent range
    rather than over absolute force, because the passive component lo_i is
    produced whether the muscle is driven or not and therefore costs nothing.
    Minimising activation, not total tension, is what the classical criterion
    intends.

    Falls back to the least-squares solution when the equality constraint is
    infeasible within the bounds -- which happens when the requested torque
    simply cannot be produced at the current posture.
    """
    from scipy.optimize import minimize

    span = np.maximum(hi - lo, 1e-9)

    def obj(f):
        return float(np.sum(((f - lo) / span) ** 2))

    def jac(f):
        return 2.0 * (f - lo) / span ** 2

    cons = [{"type": "eq", "fun": lambda f: R.T @ f - tau,
             "jac": lambda f: R.T}]
    x0 = np.clip(lo + 0.1 * span, lo, hi)
    res = minimize(obj, x0, jac=jac, bounds=list(zip(lo, hi)),
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 100, "ftol": 1e-8})
    f = np.clip(res.x, lo, hi)
    return f, bool(res.success)


def conventional_action(env, kp: float, kd: float) -> np.ndarray:
    """
    A properly formulated conventional controller.

    Two corrections to the baseline in rl/baselines.py, both of which are
    standard and both of which materially change the result:

    1. GRAVITY / BIAS COMPENSATION. The original law was
           tau_des = J^T (Kp*e - Kd*xdot)
       with no gravity term. Under gravity a musculoskeletal arm then droops
       until the position error alone generates enough torque to hold it, which
       is a large steady-state offset -- measured at ~0.46 m, worse than a random
       policy. Every textbook impedance controller adds the bias term:
           tau_des = J^T (Kp*e - Kd*xdot) + h(q, qdot)
       where MuJoCo's `qfrc_bias` supplies gravity, Coriolis and centrifugal
       terms together. Omitting it does not produce a weak conventional
       controller; it produces a broken one, and comparing against it would be
       a strawman.

    2. PER-MUSCLE FORCE SCALING. The original mapped activations through a
       single global constant (300 N for every muscle). Muscle strengths in this
       model span roughly an order of magnitude, so that misallocates effort
       systematically. Scaling each column of the moment-arm matrix by that
       muscle's own Fmax makes the NNLS solution a genuine load-sharing
       solution in activation space.
    """
    model, data = env.model, env.data
    site_id = env._site_id
    ee = data.site_xpos[site_id].copy()

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    ee_vel  = jacp @ data.qvel
    f_des   = kp * (env.target - ee) - kd * ee_vel
    tau_des = jacp.T @ f_des + data.qfrc_bias        # <- bias compensation

    # --- state-dependent achievable force range -------------------------
    # A muscle's force is NOT a*Fmax. In MuJoCo it is gain(l,v)*a + bias(l,v):
    # state dependent, with a passive component that is non-zero even at a = 0.
    # Mapping a desired FORCE to an ACTIVATION therefore requires inverting the
    # muscle model at the current length and velocity -- which is exactly the
    # Newton-Raphson equilibrium solve this thesis proposes to replace.
    #
    # Probing at a = 0 and a = 1 recovers that affine relation exactly (two
    # extra mj_forward evaluations), giving the true achievable interval
    # [f0, f1] per muscle. This both makes the controller correct and makes the
    # latency comparison honest: the conventional pipeline genuinely must invert
    # the model at every control step, and that cost is charged to it.
    # Probe `act`, NOT `ctrl`. MuJoCo muscle actuators have first-order
    # activation dynamics: ctrl is the command, `act` is the state, and
    # actuator_force = gain(l,v)*act + bias(l,v). mj_forward does not integrate
    # ctrl into act, so probing ctrl leaves the force unchanged, collapses the
    # achievable range to zero width, and the controller emits zero activation.
    saved_act = data.act.copy()
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    f0 = data.actuator_force.copy()
    data.act[:] = 1.0
    mujoco.mj_forward(model, data)
    f1 = data.actuator_force.copy()
    data.act[:] = saved_act
    mujoco.mj_forward(model, data)

    R = moment_arm_matrix(model, data)
    sgn = np.sign(f1); sgn[sgn == 0] = -1.0
    lo, hi = np.abs(f0), np.abs(f1)                  # magnitudes, lo <= hi
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    R_s = R * sgn[:, None]

    # Minimum-effort load sharing over the ACHIEVABLE range: the same criterion
    # static optimisation uses, but with the correct state-dependent bounds
    # rather than peak isometric force.
    f, ok = static_optimisation_bounded(R_s, tau_des, lo, hi)
    span = np.maximum(hi - lo, 1e-9)
    return np.clip((f - lo) / span, 0.0, 1.0).astype(np.float32)


def tune_gains(episodes: int = 6, seed: int = 0):
    """
    Coarse grid search over the PD gains.

    An untuned baseline is not a fair comparison: the learned policy received a
    sixteen-trial hyperparameter search, so the conventional controller must at
    least have its two free parameters chosen rather than guessed.
    """
    from rl.environment import ArmReachEnv

    best = (None, float("inf"))
    for kp in (60.0, 150.0, 400.0, 1000.0):
        for kd in (8.0, 25.0, 60.0):
            env = ArmReachEnv(domain_rand=False, seed=seed)
            errs = []
            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(
                        conventional_action(env, kp, kd))
                    done = term or trunc
                errs.append(info["err"])
            m = float(np.mean(errs))
            print(f"    kp={kp:>6.0f} kd={kd:>5.0f}  final err {m:.4f}", flush=True)
            if m < best[1]:
                best = ((kp, kd), m)
    print(f"    -> best gains kp={best[0][0]:.0f}, kd={best[0][1]:.0f} "
          f"(final err {best[1]:.4f})")
    return best[0]


def evaluate_controller(kind: str, episodes: int, seed: int = 0,
                        gains: tuple = None):
    """
    Drive the RAW environment with a non-learning controller and compute the
    same metric set rl/evaluate.py produces for the learned policies, plus the
    force-agreement statistics of force_comparison.py.
    """
    from rl.environment import ArmReachEnv
    from rl.baselines import _random_action

    kp, kd = gains if gains else (150.0, 25.0)
    env = ArmReachEnv(domain_rand=False, seed=seed)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())
    rng = np.random.default_rng(seed)

    finals, mins, lens, energies = [], [], [], []
    successes, blowups = [], []
    ever2, ever5 = [], []
    F_pol, F_cls = [], []
    step_us = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        errs, blew = [], False
        info = {}
        ep_energy, steps = 0.0, 0
        while not done:
            t0 = time.perf_counter()
            a = (conventional_action(env, kp, kd) if kind == "PD"
                 else _random_action(env, rng))
            step_us.append((time.perf_counter() - t0) * 1e6)

            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            errs.append(info["err"])
            ep_energy += info.get("energy", 0.0)
            steps += 1
            blew = blew or bool(info.get("blew_up"))

            d = env.data
            R = moment_arm_matrix(env.model, d)
            f_signed = d.actuator_force.copy()
            sgn = np.sign(f_signed); sgn[sgn == 0] = -1.0
            f_pol = np.abs(f_signed)
            R_s = R * sgn[:, None]
            f_cls, _ = static_optimisation(R_s, d.qfrc_actuator.copy(), fmax,
                                           f_init=f_pol)
            F_pol.append(f_pol); F_cls.append(f_cls)

        finals.append(errs[-1]); mins.append(min(errs)); lens.append(steps)
        energies.append(ep_energy / max(steps, 1))
        successes.append(bool(info.get("success", False)))
        blowups.append(blew)
        ever2.append(min(errs) < 0.02); ever5.append(min(errs) < 0.05)

    finals = np.array(finals); mins = np.array(mins)
    F_pol = np.array(F_pol); F_cls = np.array(F_cls)

    e_pol = np.sum((F_pol / fmax) ** 2, axis=1)
    e_cls = np.sum((F_cls / fmax) ** 2, axis=1)
    effort_ratio = float(e_pol.sum() / max(e_cls.sum(), 1e-12))

    num = np.sum(F_pol * F_cls, axis=1)
    den = np.linalg.norm(F_pol, axis=1) * np.linalg.norm(F_cls, axis=1)
    cos = num / np.maximum(den, 1e-12)
    rng2 = np.random.default_rng(0)
    null = np.array([float(rng2.permutation(F_pol[t]) @ F_cls[t] /
                           max(np.linalg.norm(F_pol[t]) *
                               np.linalg.norm(F_cls[t]), 1e-12))
                     for t in range(len(F_pol))])
    r_all = float(np.corrcoef(F_pol.ravel(), F_cls.ravel())[0, 1])

    return {
        "controller": kind, "episodes": episodes,
        "mean_final_error": float(finals.mean()),
        "mean_min_error": float(mins.mean()),
        "success_rate": float(np.mean(successes)),
        "reach_2cm": float(np.mean(finals < 0.02)),
        "reach_5cm": float(np.mean(finals < 0.05)),
        "reach_10cm": float(np.mean(finals < 0.10)),
        "touch_2cm": float(np.mean(ever2)),
        "touch_5cm": float(np.mean(ever5)),
        "mean_energy": float(np.mean(energies)),
        "blow_up_rate": float(np.mean(blowups)),
        "mean_episode_len": float(np.mean(lens)),
        "effort_ratio": effort_ratio,
        "pattern_cosine_mean": float(cos.mean()),
        "pattern_cosine_null_mean": float(null.mean()),
        "pearson_r": r_all,
        "control_step_us_mean": float(np.mean(step_us)),
        "control_step_us_p95": float(np.percentile(step_us, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()

    print("  tuning conventional-controller gains ...", flush=True)
    gains = tune_gains()

    out = {"gains": {"kp": gains[0], "kd": gains[1]}}
    for kind in ("PD", "Random"):
        print(f"  evaluating {kind} ...", flush=True)
        out[kind] = evaluate_controller(kind, a.episodes, gains=gains)

    dest = os.path.join(config.RUNS_DIR, "conventional_baseline.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    # Learned reference: the confirmed 300k SAC policy, same protocol.
    drl = {"mean_final_error": 0.1063, "mean_min_error": 0.0740,
           "success_rate": 0.04, "reach_5cm": 0.34, "reach_10cm": 0.58,
           "touch_2cm": 0.20, "mean_energy": 276804.0, "effort_ratio": 1.71,
           "pattern_cosine_mean": 0.869, "pearson_r": 0.865}

    print("\n" + "=" * 84)
    print(f"CONVENTIONAL CONTROLLER vs DRL — {a.episodes} episodes, identical protocol")
    print("=" * 84)
    keys = [("mean_final_error", "final error (m)"),
            ("mean_min_error", "closest approach (m)"),
            ("success_rate", "success rate"),
            ("reach_5cm", "ends < 5 cm"),
            ("reach_10cm", "ends < 10 cm"),
            ("touch_2cm", "touches 2 cm"),
            ("mean_energy", "energy"),
            ("effort_ratio", "effort ratio vs classical"),
            ("pattern_cosine_mean", "pattern cosine"),
            ("pearson_r", "pearson r")]
    print(f"  {'metric':<26}{'PD (conventional)':>20}{'Random':>14}{'SAC tuned':>14}")
    print("  " + "-" * 74)
    for k, label in keys:
        pd_v = out['PD'].get(k, float('nan'))
        rd_v = out['Random'].get(k, float('nan'))
        dl_v = drl.get(k, float('nan'))
        fmt = "{:>20.0f}" if k == "mean_energy" else "{:>20.4f}"
        fmt2 = "{:>14.0f}" if k == "mean_energy" else "{:>14.4f}"
        print(f"  {label:<26}" + fmt.format(pd_v) + fmt2.format(rd_v)
              + fmt2.format(dl_v))

    print(f"\n  PD control-step cost: {out['PD']['control_step_us_mean']:.0f} us "
          f"(p95 {out['PD']['control_step_us_p95']:.0f} us)  "
          f"— measured under load, treat as an upper bound")
    print(f"  PD null cosine: {out['PD']['pattern_cosine_null_mean']:.3f} "
          f"(the floor its {out['PD']['pattern_cosine_mean']:.3f} must clear)")

    print("\nREADING THIS TABLE")
    er = out['PD']['effort_ratio']
    print(f"  * PD's effort ratio is {er:.2f}x. It distributes force by "
          f"non-negative least\n    squares, so a value near 1 is expected BY "
          f"CONSTRUCTION and is a sanity\n    check on the analysis, not a "
          f"finding about conventional control.")
    if out['PD']['mean_final_error'] < drl['mean_final_error']:
        print("  * The conventional controller REACHES MORE ACCURATELY than the")
        print("    learned policy. The case for DRL on this task is weak and the")
        print("    thesis must say so plainly.")
    else:
        print("  * The learned policy reaches more accurately than the conventional")
        print("    controller, while using more effort. That is the trade-off the")
        print("    thesis should present as its headline.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
