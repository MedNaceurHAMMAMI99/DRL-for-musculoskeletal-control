"""
Is the MuJoCo model stable under the activations a policy can actually command?

The action space is [0,1]^9, so full co-activation (every muscle at 1.0) is a
legal command. A physical arm model must not explode when every muscle
contracts — if it does, the RL agent gets handed a "destroy the episode" button,
which is exactly the exploit diagnose_termination.py found (100% of episodes
ended in a numerical blow-up at ~7 steps).

Fixing the reward removes the *incentive* to press that button. This script asks
whether the button should exist at all, and tests the two standard remedies:

  * armature — rotor/limb inertia added to each DOF. Muscle-driven models are
    stiff systems: large forces on short (2-3 cm) moment arms produce enormous
    angular accelerations, and with no armature the effective inertia seen by
    the integrator is tiny. Armature is the standard MuJoCo stabiliser and is
    physically real (limb + tendon + soft-tissue inertia).
  * a smaller timestep — halves the integration error per step, at negligible
    cost here because wall-clock is dominated by gradient updates, not physics.

Reported as: how many of the 100 agent steps complete before the state goes
non-finite or a joint exceeds 50 rad/s (the environment's blow-up guard).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import config

FRAME_SKIP = config.FRAME_SKIP
AGENT_STEPS = 100


def run(ctrl_fn, armature=0.0, timestep=None, frame_skip=FRAME_SKIP, seed=0):
    """Return (steps_survived, peak_qvel, reason)."""
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    data = mujoco.MjData(model)
    if armature > 0:
        model.dof_armature[:] = armature
    if timestep is not None:
        model.opt.timestep = timestep
    rng = np.random.default_rng(seed)
    mujoco.mj_forward(model, data)

    peak_v = 0.0
    for step in range(AGENT_STEPS):
        data.ctrl[:] = ctrl_fn(step, rng)
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                return step, peak_v, "non-finite"
            peak_v = max(peak_v, float(np.max(np.abs(data.qvel[:4]))))
            if np.any(np.abs(data.qvel[:4]) > 50):
                return step, peak_v, "qvel>50"
    return AGENT_STEPS, peak_v, "survived"


PATTERNS = {
    "full co-activation (all 1.0)": lambda s, r: np.ones(9),
    "random uniform [0,1]":         lambda s, r: r.uniform(0, 1, 9),
    "alternating full on/off":      lambda s, r: np.ones(9) if s % 2 else np.zeros(9),
    "policy-like (~0.40 mean)":     lambda s, r: np.clip(r.normal(0.40, 0.15, 9), 0, 1),
}

CONFIGS = [
    ("current (ts=0.002, armature=0)",      dict()),
    ("armature=0.01",                        dict(armature=0.01)),
    ("armature=0.05",                        dict(armature=0.05)),
    ("ts=0.001, frame_skip=20",              dict(timestep=0.001, frame_skip=20)),
    ("armature=0.01 + ts=0.001, fs=20",      dict(armature=0.01, timestep=0.001, frame_skip=20)),
]

print("=" * 78)
print(f"STABILITY TEST — {AGENT_STEPS} agent steps survived (higher is better)")
print("=" * 78)
header = f"{'config':<34}" + "".join(f"{k.split(' (')[0][:16]:>17}" for k in PATTERNS)
print(header)
print("-" * 78)
for cname, kw in CONFIGS:
    cells = []
    for pname, fn in PATTERNS.items():
        steps, peak_v, reason = run(fn, **kw)
        mark = "OK " if steps == AGENT_STEPS else "!! "
        cells.append(f"{mark}{steps:>3}/{AGENT_STEPS} {reason[:6]:>7}")
    print(f"{cname:<34}" + "".join(f"{c:>17}" for c in cells))

print()
print("Interpretation: any '!!' under 'full co-activation' or 'policy-like' means")
print("the agent can reach a numerical blow-up with a legal action sequence.")
