"""
Measure each tendon's min/max length over the joint workspace and print the
corresponding actuator lengthrange values for arm.xml.

MuJoCo maps a muscle's operating range onto its normalized force-length curve
via `lengthrange`; the standard practice (what MuJoCo's own auto-computation
does) is to use the actual tendon length extremes over the reachable joint
space. Hand-set placeholders distort the F-L curve, so we measure.
"""

import itertools
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

grid = [np.linspace(lo, hi, 7) for lo, hi in model.jnt_range]
tmin = np.full(model.ntendon, np.inf)
tmax = np.full(model.ntendon, -np.inf)

for q in itertools.product(*grid):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    tmin = np.minimum(tmin, data.ten_length)
    tmax = np.maximum(tmax, data.ten_length)

print(f"{'tendon':<14} {'min':>8} {'max':>8}")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
    print(f"{name:<14} {tmin[i]:8.4f} {tmax[i]:8.4f}")

print("\nPatch lines for arm.xml <actuator>:")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
    force = model.actuator_gainprm[i, 2]
    print(f'    <muscle name="{name}"  tendon="{name}"  force="{force:.0f}"  '
          f'lengthrange="{tmin[i]:.4f} {tmax[i]:.4f}"/>')
