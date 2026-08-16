"""
Static feasibility test — CAN the 9 muscles hold the target postures at all?

This is the question the 1e6-step SAC run raised: the policy plateaus ~0.35 m
from the target and never satisfies (err < 2 cm AND |qdot| < 0.1). Either the
learning is inadequate, or the criterion is mechanically unattainable. This
test answers it without any learning, analytically, per target posture.

At a target joint configuration q* with zero velocity, muscle force is affine
in activation:  f_i(a_i) = f_pass_i + a_i * (f_full_i - f_pass_i),
so the achievable actuator torque is  R^T f_pass + A a  with
A = R^T diag(f_full - f_pass) and 0 <= a <= 1.

Static equilibrium requires that to cancel gravity + passive terms:
    A a = qfrc_bias - qfrc_passive - R^T f_pass  =:  b
Bounded least squares gives the best achievable a and the residual torque per
joint. A large residual on a DOF means gravity wins there and the arm sags —
no policy, however well trained, can hold that posture.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
from scipy.optimize import lsq_linear
import config

JOINTS = ["shoulder_flex", "shoulder_abd", "shoulder_rot", "elbow_flex"]


def dense_moment(model, data):
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return R


def hold_residual(model, data, q):
    """Best-case static holding torque residual at configuration q."""
    data.qpos[:] = q
    data.qvel[:] = 0.0

    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    f_pass = data.actuator_force.copy()
    R      = dense_moment(model, data)
    bias   = data.qfrc_bias.copy()
    passv  = data.qfrc_passive.copy()

    f_full = np.zeros(model.nu)
    for i in range(model.nu):
        data.act[:] = 0.0
        data.act[i] = 1.0
        mujoco.mj_forward(model, data)
        f_full[i] = data.actuator_force[i]
    data.act[:] = 0.0

    A = R.T @ np.diag(f_full - f_pass)          # nv x nu
    b = bias - passv - R.T @ f_pass
    # Solve only over the muscle-controlled DOFs. shoulder_rot (index 2) is
    # held by its own stiff constraint, not by muscles; including it in the
    # objective makes least squares sacrifice the real DOFs chasing torque it
    # can never produce.
    rows = [0, 1, 3]
    sol = lsq_linear(A[rows], b[rows], bounds=(0.0, 1.0))
    residual = A @ sol.x - b                    # uncancelled torque, N*m
    return residual, sol.x, bias


def sample_q_star(model, rng, restrict_rot=True):
    lo, hi = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
    if restrict_rot:
        lo[2], hi[2] = -0.02, 0.02              # match env target sampling
    return lo + rng.uniform(0.05, 0.95, model.nq) * (hi - lo)


def main(n=200):
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    data  = mujoco.MjData(model)
    rng   = np.random.default_rng(0)

    res, acts, biases = [], [], []
    for _ in range(n):
        q = sample_q_star(model, rng)
        r, a, bias = hold_residual(model, data, q)
        res.append(np.abs(r)); acts.append(a); biases.append(np.abs(bias))
    res, acts, biases = np.array(res), np.array(acts), np.array(biases)

    print(f"== Static hold feasibility over {n} target postures ==")
    print(f"{'joint':<16} {'|gravity tau|':>14} {'|residual tau|':>15} {'held?':>8}")
    for j, name in enumerate(JOINTS):
        med_g = np.median(biases[:, j])
        med_r = np.median(res[:, j])
        frac_held = np.mean(res[:, j] < 0.5)
        print(f"{name:<16} {med_g:14.2f} {med_r:15.2f} {100*frac_held:7.0f}%")

    all_held = np.mean(np.all(res < 0.5, axis=1))
    print(f"\n   postures where ALL joints hold (residual < 0.5 N*m): "
          f"{100*all_held:.0f}%")
    print(f"   median activation used: {np.median(acts):.3f}   "
          f"fraction of muscles saturated at 1.0: "
          f"{100*np.mean(acts > 0.99):.0f}%")

    # Same test with the rest posture, as a reference point.
    q0 = np.zeros(model.nq)
    r0, a0, bias0 = hold_residual(model, data, q0)
    print(f"\n   rest posture residual: {np.round(r0, 2)} (N*m)")
    print(f"   rest posture gravity : {np.round(bias0, 2)} (N*m)")


if __name__ == "__main__":
    main()
