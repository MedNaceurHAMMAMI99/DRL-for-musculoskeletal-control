"""
Non-learning baselines: a model-based PD controller and a random policy.

These give the DRL algorithms something to be compared against and are cheap
(no training). Both are evaluated on the raw `ArmReachEnv` (no observation
normalisation) with the SAME success criterion, reward, and episode budget as
the learned policies, so their metrics are directly comparable.

PD baseline (documented, approximate — this IS the implementation, not a claim)
------------------------------------------------------------------------------
A Cartesian PD law produces a desired end-effector wrench from the reaching
error and end-effector velocity:

    f_des = Kp * (target - ee) - Kd * ee_vel            (in R^3)

mapped to a desired joint torque by the site Jacobian transpose:

    tau_des = J_site^T f_des                             (in R^{nv})

and finally to non-negative muscle activations by non-negative least squares on
the muscle moment-arm matrix R = data.actuator_moment (muscles can only pull):

    a* = argmin_{a>=0} || R^T a - tau_des ||,   a = clip(a*, 0, 1)

This is a legitimate impedance-style baseline; it is deliberately simple and its
gains are hand-set (documented here), not tuned. It is not expected to beat the
learned policies — it exists to bound "how far does learning get you over a
straightforward model-based controller."
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import mujoco
from scipy.optimize import nnls
import config


# ── PD gains (hand-set, documented) ───────────────────────────────────────────
PD_KP = 60.0
PD_KD = 8.0
_FORCE_SCALE = 300.0   # nominal muscle-force scale folded into the NNLS mapping


def _pd_action(env) -> np.ndarray:
    """One PD control step: obs error -> non-negative muscle activations."""
    model, data = env.model, env.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    ee = data.site_xpos[site_id].copy()

    # End-effector Jacobian (3 x nv).
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    ee_vel = jacp @ data.qvel
    f_des  = PD_KP * (env.target - ee) - PD_KD * ee_vel      # R^3 wrench
    tau_des = jacp.T @ f_des                                  # R^{nv}

    # Muscle moment-arm matrix R (nu x nv): generalized force = R^T @ (force).
    # MuJoCo >= 3.2 stores actuator_moment sparse; densify explicitly.
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    A = (R * _FORCE_SCALE).T                                  # nv x nu
    try:
        a, _ = nnls(A, tau_des, maxiter=200)
    except (RuntimeError, np.linalg.LinAlgError):
        # NNLS did not converge / matrix near-singular for this configuration;
        # fall back to the clipped least-squares solution (muscles cannot push).
        a = np.clip(np.linalg.lstsq(A, tau_des, rcond=None)[0], 0.0, None)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _random_action(env, rng) -> np.ndarray:
    return rng.uniform(0.0, 1.0, config.N_MUSCLES).astype(np.float32)


def evaluate_baseline(kind: str = "PD", n_episodes: int = None, seed: int = 0,
                      log_activations: bool = False) -> dict:
    """
    Evaluate a baseline controller. Returns the SAME dict shape as
    rl.evaluate.evaluate so it flows into the identical statistics/analysis.
    """
    from rl.environment import ArmReachEnv

    n_episodes = n_episodes or config.EVAL_EPISODES
    kind = kind.upper()
    env  = ArmReachEnv(domain_rand=False, seed=seed)
    rng  = np.random.default_rng(seed)

    successes, rewards, errors, energies, lengths = [], [], [], [], []
    act_traces = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = trunc = False
        ep_r = ep_e = 0.0
        steps = 0
        traj  = []
        while not (done or trunc):
            a = _pd_action(env) if kind == "PD" else _random_action(env, rng)
            if log_activations:
                traj.append(a.copy())
            obs, r, done, trunc, info = env.step(a)
            ep_r += r
            ep_e += info.get("energy", 0.0)
            steps += 1
        successes.append(bool(info.get("success", False)))
        rewards.append(ep_r)
        errors.append(float(info.get("err", float("nan"))))
        energies.append(ep_e / max(steps, 1))
        lengths.append(steps)
        if log_activations:
            act_traces.append(np.asarray(traj, dtype=np.float32))

    out = {
        "success_rate":     float(np.mean(successes)),
        "mean_reward":      float(np.mean(rewards)),
        "mean_final_error": float(np.nanmean(errors)),
        "mean_energy":      float(np.mean(energies)),
        "n_episodes":       n_episodes,
        "success_per_ep":   [bool(s) for s in successes],
        "reward_per_ep":    [float(x) for x in rewards],
        "error_per_ep":     [float(x) for x in errors],
        "energy_per_ep":    [float(x) for x in energies],
    }
    if log_activations:
        out["activations"] = act_traces
    return out
