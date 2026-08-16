"""
Does the trained policy actually USE the target, or has it collapsed to a
single mean posture?

Runs episodes, recording each target and the end-effector position the policy
settles at. If the settle points cluster tightly regardless of target (low
correlation, small spread vs target spread), the policy has learned the
workspace-centroid solution and is ignoring the goal — a very different
failure from "reaches toward the goal but imprecisely".

Also times raw physics stepping vs full RL stepping, to size the control
decimation (frame skip) trade-off.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import mujoco
import config
from rl.evaluate import load_model


def target_usage(n=25):
    model, envs = load_model("runs/SAC_seed0", "SAC")
    targets, settles, finals = [], [], []
    base = envs.venv.envs[0].unwrapped
    for _ in range(n):
        obs = envs.reset()
        done = [False]
        # The vec env auto-resets on termination, so the state must be sampled
        # BEFORE each step; the last sample is the terminal one.
        tgt = np.asarray(base.target, dtype=float)
        ee = base.data.site_xpos[base._site_id].copy()
        while not done[0]:
            a, _ = model.predict(obs, deterministic=True)
            tgt = np.asarray(base.target, dtype=float)
            ee = base.data.site_xpos[base._site_id].copy()
            obs, r, done, infos = envs.step(a)
        targets.append(tgt); settles.append(ee)
        finals.append(float(np.linalg.norm(tgt - ee)))
    targets, settles = np.array(targets), np.array(settles)

    print("== Target usage ==")
    print(f"   target spread (std per axis): {np.round(targets.std(axis=0), 3)}")
    print(f"   settle spread (std per axis): {np.round(settles.std(axis=0), 3)}")
    print(f"   settle centroid            : {np.round(settles.mean(axis=0), 3)}")
    print(f"   target centroid            : {np.round(targets.mean(axis=0), 3)}")
    for ax, name in enumerate("xyz"):
        c = np.corrcoef(targets[:, ax], settles[:, ax])[0, 1]
        print(f"   corr(target_{name}, settle_{name}) = {c:+.3f}")
    ratio = settles.std(axis=0).mean() / targets.std(axis=0).mean()
    print(f"   settle-spread / target-spread = {ratio:.3f}"
          f"   (near 0 => ignores target; near 1 => tracks it)")
    print(f"   mean final error = {np.mean(finals):.3f} m")


def timing():
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    data = mujoco.MjData(model)
    n = 50_000
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
        if data.time > 100:
            mujoco.mj_resetData(model, data)
    dt_phys = (time.perf_counter() - t0) / n
    print("\n== Timing ==")
    print(f"   raw physics step : {dt_phys*1e6:8.1f} us")
    print(f"   -> 1e6 physics steps = {dt_phys*1e6/60:.1f} min (single env)")
    print(f"   measured SAC run : 1e6 RL steps = 72.7 min "
          f"(= physics {dt_phys*1e6/60:.1f} min + ~{72.7 - dt_phys*1e6/60:.1f} min "
          f"of gradient updates / overhead)")


if __name__ == "__main__":
    target_usage()
    timing()
