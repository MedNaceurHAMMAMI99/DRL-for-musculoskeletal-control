"""
Reach oracle — an upper bound on task performance, with privileged information.

A joint-space PD law with gravity compensation, whose desired torque is mapped
to non-negative muscle activations by bounded least squares on the moment-arm
matrix. It is given the target's generating joint configuration q* directly
(privileged: the RL policy only sees the Cartesian error vector), so it bounds
what is dynamically achievable in the episode budget.

If this controller reaches < 2 cm with near-zero velocity, the success
criterion is attainable and any 0% RL result is a learning problem.
If it cannot, the criterion or the episode budget is the problem, not the agent.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
from scipy.optimize import lsq_linear
import config

CTRL_EVERY = 5                       # zero-order hold (control @ 100 Hz)
KP = np.array([120.0, 120.0, 0.0, 90.0])
KD = np.array([18.0, 18.0, 0.0, 12.0])
ROWS = [0, 1, 3]                     # muscle-controlled DOFs


def dense_moment(model, data):
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return R


def solve_activation(model, data, q_star):
    tau = (KP * (q_star - data.qpos) - KD * data.qvel
           + data.qfrc_bias - data.qfrc_passive)
    R = dense_moment(model, data)
    f_pass = data.actuator_force.copy()
    f_full = np.zeros(model.nu)
    act_save = data.act.copy()
    for i in range(model.nu):
        data.act[:] = 0.0
        data.act[i] = 1.0
        mujoco.mj_forward(model, data)
        f_full[i] = data.actuator_force[i]
    data.act[:] = act_save
    mujoco.mj_forward(model, data)

    A = R.T @ np.diag(f_full - f_pass)
    b = tau - R.T @ f_pass
    sol = lsq_linear(A[ROWS], b[ROWS], bounds=(0.0, 1.0))
    return sol.x


def run_episode(model, site_id, q_star, target, max_steps=1000):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    a = np.zeros(model.nu)
    errs = []
    for t in range(max_steps):
        if t % CTRL_EVERY == 0:
            a = solve_activation(model, data, q_star)
        data.ctrl[:] = a
        mujoco.mj_step(model, data)
        errs.append(float(np.linalg.norm(target - data.site_xpos[site_id])))
    v = float(np.linalg.norm(data.qvel[:4]))
    return np.array(errs), v


def main(n=15):
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    lo, hi = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
    lo[2], hi[2] = -0.02, 0.02

    finals, mins, vels = [], [], []
    for ep in range(n):
        q_star = lo + rng.uniform(0.05, 0.95, model.nq) * (hi - lo)
        scratch.qpos[:] = q_star
        mujoco.mj_forward(model, scratch)
        target = scratch.site_xpos[site_id].copy()
        errs, v = run_episode(model, site_id, q_star, target)
        finals.append(errs[-1]); mins.append(errs.min()); vels.append(v)
        print(f"  ep{ep:2d}: start {errs[0]:.3f}  min {errs.min():.4f}  "
              f"final {errs[-1]:.4f}  |qdot| {v:.3f}", flush=True)

    finals, mins, vels = np.array(finals), np.array(mins), np.array(vels)
    print(f"\n== Reach oracle (n={n}, 2 s budget, privileged q*) ==")
    print(f"   final err : median {np.median(finals):.4f}  mean {finals.mean():.4f}"
          f"  max {finals.max():.4f} (m)")
    print(f"   final < 2 cm : {100*np.mean(finals < 0.02):.0f}%"
          f"   < 5 cm : {100*np.mean(finals < 0.05):.0f}%")
    print(f"   SUCCESS (err<2cm AND |qdot|<0.1): "
          f"{100*np.mean((finals < 0.02) & (vels < 0.1)):.0f}%")


if __name__ == "__main__":
    main()
