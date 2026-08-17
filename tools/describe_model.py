"""Extract the ground-truth model description straight from the compiled MJCF."""
import os, sys
import numpy as np
sys.path.insert(0, r"C:\Users\moham\thesis_run")
os.chdir(r"C:\Users\moham\thesis_run")
import mujoco, config

m = mujoco.MjModel.from_xml_path(config.ARM_XML)
name = lambda t, i: mujoco.mj_id2name(m, t, i)

print("=" * 74)
print("BODIES")
tot = 0.0
for i in range(1, m.nbody):
    tot += m.body_mass[i]
    print(f"  {name(mujoco.mjtObj.mjOBJ_BODY,i):12s} mass={m.body_mass[i]:.3f} kg   "
          f"diaginertia={np.array2string(m.body_inertia[i], precision=5)}")
print(f"  TOTAL moving mass = {tot:.3f} kg")

print("\nJOINTS  (nq=%d nv=%d)" % (m.nq, m.nv))
for i in range(m.njnt):
    lo, hi = m.jnt_range[i]
    lim = "limited" if m.jnt_limited[i] else "free"
    print(f"  {name(mujoco.mjtObj.mjOBJ_JOINT,i):16s} range=[{np.degrees(lo):7.1f}, "
          f"{np.degrees(hi):7.1f}] deg  ({lim})")

print("\nACTUATORS  (nu=%d)" % m.nu)
print(f"  {'name':16s} {'gainprm (Fmax etc.)':38s} {'ctrlrange'}")
for i in range(m.nu):
    gp = np.array2string(m.actuator_gainprm[i][:6], precision=3, suppress_small=True)
    cr = np.array2string(m.actuator_ctrlrange[i], precision=2)
    print(f"  {name(mujoco.mjtObj.mjOBJ_ACTUATOR,i):16s} {gp:38s} {cr}")

print("\nconfig.MUSCLE_NAMES:")
for i, n in enumerate(config.MUSCLE_NAMES):
    print(f"  [{i}] {n}")
print("\nCCI groups from config:")
print("  agonist   idx", config.CCI_AGONIST_IDX,
      "->", [config.MUSCLE_NAMES[i] for i in config.CCI_AGONIST_IDX])
print("  antagonist idx", config.CCI_ANTAGONIST_IDX,
      "->", [config.MUSCLE_NAMES[i] for i in config.CCI_ANTAGONIST_IDX])

# peak isometric force actually used by the load-sharing analysis
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("\nFmax as used by force_comparison (actuator_gainprm[:,2] or scale):")
for i in range(m.nu):
    print(f"  {name(mujoco.mjtObj.mjOBJ_ACTUATOR,i):16s} "
          f"gainprm={m.actuator_gainprm[i][:3]}  lengthrange={m.actuator_lengthrange[i]}")
