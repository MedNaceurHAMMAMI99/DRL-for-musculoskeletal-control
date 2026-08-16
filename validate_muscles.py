"""
Muscle-function validation: activate each muscle alone from rest and record
the resulting joint displacements and end-effector motion.

Pass criteria (checked programmatically):
  - every muscle moves at least one joint by >= 0.05 rad;
  - elbow flexors flex (elbow +), extensors extend or hold it at the 0 limit;
  - deltoid_ant produces shoulder flexion (+), deltoid_post extension (-),
    deltoid_med abduction (+);
  - a co-activation hold test: moderate uniform activation must not explode.

Writes runs/muscle_function.json — a real, reportable model-validation artifact.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import config

JOINTS = ["shoulder_flex", "shoulder_abd", "shoulder_rot", "elbow_flex"]


MID_POSE = np.array([0.5, 0.3, 0.0, 1.2])  # mid-workspace start; the hanging
# pose sits ON the elbow-extension limit, where extensors correctly have
# nothing to act against and would trivially "fail" a did-it-move check.


def solo_activation(model, act_level=0.6, n_steps=400):
    out = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        data = mujoco.MjData(model)
        data.qpos[:4] = MID_POSE
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        ee0 = data.site_xpos[site].copy()
        for _ in range(n_steps):
            data.ctrl[:] = 0.0
            data.ctrl[i] = act_level
            mujoco.mj_step(model, data)
        dq = data.qpos[:4] - MID_POSE
        ee1 = data.site_xpos[site].copy()
        out[name] = {
            "dq": {j: round(float(d), 4) for j, d in zip(JOINTS, dq)},
            "ee_displacement_m": round(float(np.linalg.norm(ee1 - ee0)), 4),
            "peak_force_N": round(float(np.abs(data.actuator_force[i])), 1),
        }
    return out


def coactivation_hold(model, act_level=0.3, n_steps=1000):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    max_vel = 0.0
    for _ in range(n_steps):
        data.ctrl[:] = act_level
        mujoco.mj_step(model, data)
        max_vel = max(max_vel, float(np.max(np.abs(data.qvel[:4]))))
    return {"final_q": {j: round(float(v), 4) for j, v in zip(JOINTS, data.qpos[:4])},
            "max_abs_qvel": round(max_vel, 3),
            "stable": bool(max_vel < 50)}


def main():
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    solo = solo_activation(model)
    hold = coactivation_hold(model)

    print(f"{'muscle':<14} {'d_shf':>7} {'d_abd':>7} {'d_rot':>7} {'d_elb':>7} "
          f"{'|d_ee| m':>9} {'F peak N':>9}")
    for name, r in solo.items():
        dq = r["dq"]
        print(f"{name:<14} {dq['shoulder_flex']:7.3f} {dq['shoulder_abd']:7.3f} "
              f"{dq['shoulder_rot']:7.3f} {dq['elbow_flex']:7.3f} "
              f"{r['ee_displacement_m']:9.3f} {r['peak_force_N']:9.1f}")
    print("co-activation hold (a=0.3, 2 s):", hold)

    checks = {
        "all_muscles_move_a_joint":
            all(max(abs(v) for v in r["dq"].values()) >= 0.05 for r in solo.values()),
        "flexors_flex_elbow":
            all(solo[m]["dq"]["elbow_flex"] > 0.05
                for m in ["biceps_long", "biceps_short", "brachialis"]),
        "extensors_do_not_flex_elbow":
            all(solo[m]["dq"]["elbow_flex"] < 0.05
                for m in ["triceps_long", "triceps_lat", "triceps_med"]),
        "delt_ant_flexes_shoulder": solo["deltoid_ant"]["dq"]["shoulder_flex"] > 0.05,
        "delt_post_extends_shoulder": solo["deltoid_post"]["dq"]["shoulder_flex"] < -0.05,
        "delt_med_abducts": solo["deltoid_med"]["dq"]["shoulder_abd"] > 0.05,
        "coactivation_stable": hold["stable"],
    }
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    with open(os.path.join(config.RUNS_DIR, "muscle_function.json"), "w") as f:
        json.dump({"solo_activation": solo, "coactivation_hold": hold,
                   "checks": checks}, f, indent=2)
    print("\nwrote runs/muscle_function.json")
    return all(checks.values())


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
