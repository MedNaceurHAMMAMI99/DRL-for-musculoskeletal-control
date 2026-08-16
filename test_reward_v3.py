"""
Verification for the v3 reward/termination fixes.

Each check corresponds to a specific defect found on 2026-08-14:

  1. Per-step reward is strictly positive at every state a rollout visits.
     This is the property that removes the suicide exploit — if r > 0 always,
     then ending an episode early always forfeits value. A single negative
     sample means w4 is too small and the exploit is still reachable.
  2. Episodes run to truncation, not to an early termination.
  3. Blow-up is strictly penalised relative to surviving the same state.
  4. On a non-finite blow-up the reported error is the last finite error, not
     the home-pose error the v2 code produced by resetting first.
  5. Domain randomisation varies elements independently rather than applying
     one global scale factor.
  6. Success no longer terminates the episode.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from rl.environment import ArmReachEnv

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 70)
print("REWARD v3 VERIFICATION")
print("=" * 70)

# ---- 1 & 2: reward positivity and episode completion under random policy ----
env = ArmReachEnv(domain_rand=True, seed=0)
rng = np.random.default_rng(0)
# Blow-up steps are excluded from the positivity check: the w6 penalty is
# SUPPOSED to make that one step negative. The property that matters is that
# every ordinary step pays more than nothing, so surviving always beats quitting.
ok_r, blow_r, lengths, blew_flags = [], [], [], []
for ep in range(30):
    obs, _ = env.reset()
    n = 0
    while True:
        a = rng.uniform(0, 1, 9)
        obs, r, term, trunc, info = env.step(a)
        (blow_r if info["blew_up"] else ok_r).append(r)
        n += 1
        if term or trunc:
            break
    lengths.append(n)
    blew_flags.append(info["blew_up"])
ok_r = np.asarray(ok_r)

check("per-step reward strictly positive on every non-failure step",
      ok_r.min() > 0,
      f"min={ok_r.min():.3f}  mean={ok_r.mean():.3f}  max={ok_r.max():.3f}")
# A random policy flails hard enough to trip the 50 rad/s guard sometimes; that
# is a genuine failure the agent must learn to avoid, not a defect. What must
# hold is that nothing ELSE ends an episode early — in particular that reaching
# the target no longer does.
survived = [L for L, b in zip(lengths, blew_flags) if not b]
check("every episode that did not blow up ran to full length",
      all(L == config.MAX_EPISODE_STEPS for L in survived),
      f"{len(survived)}/30 survived, lengths {sorted(set(survived))}; "
      f"{sum(blew_flags)}/30 hit the velocity guard under RANDOM actions")

# A benign, non-flailing policy must complete every episode.
env_b = ArmReachEnv(domain_rand=True, seed=7)
benign_lengths = []
for ep in range(10):
    env_b.reset()
    n = 0
    while True:
        _, _, term, trunc, _ = env_b.step(np.full(9, 0.15))
        n += 1
        if term or trunc:
            break
    benign_lengths.append(n)
check("a benign policy completes full-length episodes",
      all(L == config.MAX_EPISODE_STEPS for L in benign_lengths),
      f"lengths {sorted(set(benign_lengths))} of {config.MAX_EPISODE_STEPS}")

# Worst-case probe: the adversarial patterns from test_stability.py, which are
# the actions an exploiting policy would choose.
worst = []
for pattern in (np.ones(9), np.zeros(9)):
    env.reset()
    for _ in range(config.MAX_EPISODE_STEPS):
        _, r, term, trunc, _ = env.step(pattern)
        worst.append(r)
        if term or trunc:
            break
check("reward positive under full co-activation and zero activation",
      min(worst) > 0, f"min={min(worst):.3f}")

# ---- 3: blow-up is penalised ----
env_pen = ArmReachEnv(domain_rand=False, seed=1)
env_pen.reset()
_, r_normal, _, _, _ = env_pen.step(np.full(9, 0.2))
# Force the velocity guard by writing an impossible joint velocity directly.
env_pen.reset()
env_pen.data.qvel[:4] = 999.0
_, r_blow, term_blow, _, info_blow = env_pen.step(np.full(9, 0.2))
check("blow-up terminates", term_blow is True)
check("blow-up flagged in info", info_blow.get("blew_up") is True)
check("blow-up reward is worse than a normal step",
      r_blow < r_normal - config.REWARD_WEIGHTS["w6"] / 2,
      f"blow-up {r_blow:.3f} vs normal {r_normal:.3f} (w6={config.REWARD_WEIGHTS['w6']})")

# ---- 4: error reported on a non-finite blow-up ----
env_nf = ArmReachEnv(domain_rand=False, seed=2)
env_nf.reset()
# Advance a few steps so a meaningful "last finite error" exists and differs
# from the home pose, then poison the state to force the non-finite branch.
for _ in range(5):
    env_nf.step(np.full(9, 0.3))
last_err_before = env_nf._last_err
env_nf.data.qpos[0] = np.nan
_, _, term_nf, _, info_nf = env_nf.step(np.full(9, 0.3))
home_err = float(np.linalg.norm(
    env_nf.target - env_nf.data.site_xpos[env_nf._site_id]))
check("non-finite blow-up terminates", term_nf is True)
check("reported error is the last finite error, not the post-reset home pose",
      abs(info_nf["err"] - last_err_before) < 1e-9 and abs(info_nf["err"] - home_err) > 1e-6,
      f"reported {info_nf['err']:.4f}, last finite {last_err_before:.4f}, "
      f"home pose {home_err:.4f}")

# ---- 5: domain randomisation is per-element ----
env_dr = ArmReachEnv(domain_rand=True, seed=3)
env_dr.reset()
with np.errstate(invalid="ignore", divide="ignore"):
    # The world body has zero mass; its ratio is meaningless and filtered below.
    ratios_mass = env_dr.model.body_mass / env_dr._nominal_body_mass
ratios_gain = (env_dr.model.actuator_gainprm[:, 2]
               / env_dr._nominal_gainprm[:, 2])
finite_mass = ratios_mass[np.isfinite(ratios_mass) & (ratios_mass > 0)]
check("mass randomisation varies per body",
      float(np.std(finite_mass)) > 1e-6,
      f"std of per-body multiplier = {np.std(finite_mass):.4f}")
check("muscle-strength randomisation varies per muscle",
      float(np.std(ratios_gain)) > 1e-6,
      f"std of per-muscle multiplier = {np.std(ratios_gain):.4f}")

# ---- 6: success does not terminate ----
# Place the target on the end effector so the success predicate fires while the
# arm is still at rest (velocity is ~0 immediately after reset).
env_s = ArmReachEnv(domain_rand=False, seed=4)
env_s.reset()
env_s.target = env_s.data.site_xpos[env_s._site_id].copy()
_, r_s, term_s, _, info_s = env_s.step(np.zeros(9))
check("success predicate fires when on target and settled",
      info_s["success"] is True, f"err={info_s['err']:.4f}")
check("success does NOT terminate the episode", term_s is False)
check("success bonus is paid",
      r_s > config.REWARD_WEIGHTS["w4"],
      f"reward {r_s:.3f} > w4 {config.REWARD_WEIGHTS['w4']}")

print()
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
