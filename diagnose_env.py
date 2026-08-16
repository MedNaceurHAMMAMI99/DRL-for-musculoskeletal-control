"""
Environment diagnostic — run BEFORE and AFTER env fixes.

Measures, with no training involved:
  1. Initial reaching-error distribution over many resets vs the termination
     threshold (is the task dead on arrival?).
  2. Episode length + termination cause under a random policy.
  3. Magnitude of each reward term (is any term dominating?).
  4. Approximate reachable workspace of the end effector vs the target box.
  5. Domain-randomization drift of model parameters across resets
     (does body_mass random-walk away from nominal?).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import config
from rl.environment import ArmReachEnv


def main(n_resets=1000, n_episodes=50):
    rng = np.random.default_rng(0)

    # ---- 1. initial error distribution + 5. DR drift ----------------------
    env = ArmReachEnv(domain_rand=True, seed=0)
    nominal_mass = env.model.body_mass.copy()
    errs0, mass_ratio = [], []
    for _ in range(n_resets):
        obs, _ = env.reset()
        errs0.append(float(np.linalg.norm(obs[-3:])))
        mass_ratio.append(float(env.model.body_mass.sum() / nominal_mass.sum()))
    errs0 = np.array(errs0)
    print("== 1. Initial reaching error (n=%d resets, domain_rand=True) ==" % n_resets)
    print("   min %.3f  median %.3f  mean %.3f  max %.3f (m)"
          % (errs0.min(), np.median(errs0), errs0.mean(), errs0.max()))
    print("   fraction starting ABOVE 0.60 m failure threshold: %.1f%%"
          % (100.0 * np.mean(errs0 > 0.60)))
    print("   fraction starting above 0.50 m: %.1f%%" % (100.0 * np.mean(errs0 > 0.50)))
    print("== 5. DR drift across resets (same env instance) ==")
    print("   body-mass ratio vs nominal after %d resets: %.3f (1.0 = no drift)"
          % (n_resets, mass_ratio[-1]))
    print("   trajectory of ratio: start %.3f  25%% %.3f  50%% %.3f  75%% %.3f  end %.3f"
          % (mass_ratio[0], mass_ratio[n_resets // 4], mass_ratio[n_resets // 2],
             mass_ratio[3 * n_resets // 4], mass_ratio[-1]))

    # ---- 2 + 3. random-policy episodes: length, termination cause, terms ---
    env = ArmReachEnv(domain_rand=True, seed=1)
    lengths, causes = [], {"success": 0, "err>thresh": 0, "vel": 0, "truncated": 0}
    term_err2, term_energy_w, term_smooth, raw_energy = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        prev_a = np.zeros(9, dtype=np.float32)
        for t in range(env.max_steps):
            a = rng.uniform(0, 1, 9).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            err = info["err"]
            term_err2.append(env.w1 * err ** 2)
            e_pen = getattr(env, "last_energy_penalty", None)
            term_energy_w.append(e_pen if e_pen is not None
                                 else env.w2 * info["energy"])
            term_smooth.append(env.w3 * float(np.sum((a - prev_a) ** 2)))
            raw_energy.append(info["energy"])
            prev_a = a
            if term or trunc:
                if info.get("success"):
                    causes["success"] += 1
                elif trunc:
                    causes["truncated"] += 1
                elif err > 0.60:
                    causes["err>thresh"] += 1
                else:
                    causes["vel"] += 1
                lengths.append(t + 1)
                break
    print("== 2. Random-policy episodes (n=%d) ==" % n_episodes)
    print("   episode length: min %d  median %d  max %d (max_steps=%d)"
          % (min(lengths), int(np.median(lengths)), max(lengths), env.max_steps))
    print("   termination causes:", causes)
    print("== 3. Per-step reward-term magnitudes (weighted, random policy) ==")
    print("   w1*err^2      : mean %10.3f  max %10.3f" % (np.mean(term_err2), np.max(term_err2)))
    print("   energy term   : mean %10.3f  max %10.3f" % (np.mean(term_energy_w), np.max(term_energy_w)))
    print("   w3*|da|^2     : mean %10.3f  max %10.3f" % (np.mean(term_smooth), np.max(term_smooth)))
    print("   raw sum(F^2)  : mean %10.0f  max %10.0f" % (np.mean(raw_energy), np.max(raw_energy)))

    # ---- 4. reachable workspace vs target box ------------------------------
    env = ArmReachEnv(domain_rand=False, seed=2)
    pts = []
    for ep in range(30):
        env.reset()
        a = rng.uniform(0, 1, 9).astype(np.float32)
        for t in range(300):
            if t % 50 == 0:
                a = rng.uniform(0, 1, 9).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            ee = env.target - obs[-3:]  # target - err_vec = ee position
            pts.append(ee.copy())
            if term or trunc:
                break
    pts = np.array(pts)
    lo = np.array([-0.20, -0.30, -0.45])
    hi = np.array([0.45, 0.30, 0.00])
    print("== 4. End-effector excursion under random muscle input (n=%d pts) ==" % len(pts))
    print("   ee min  :", np.round(pts.min(axis=0), 3))
    print("   ee max  :", np.round(pts.max(axis=0), 3))
    print("   target lo:", lo, " hi:", hi)
    in_box = np.mean(np.all((pts >= lo) & (pts <= hi), axis=1))
    print("   fraction of visited ee points inside target box: %.1f%%" % (100 * in_box))


if __name__ == "__main__":
    main()
