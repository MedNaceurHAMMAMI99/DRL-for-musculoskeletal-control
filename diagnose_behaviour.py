"""
What is the v3 policy actually doing?

Pilot 4 removed the termination exploit — episodes now run their full length and
the agent is genuinely attempting the task — but accuracy did not improve
(0.242 m final error, 0% within 5 cm) and effort stayed at ~1.25e6, i.e. ~48% RMS
activation where ~8% holds the arm against gravity.

The leading hypothesis is a STIFFENING strategy: if agonists and antagonists both
fire hard, their torques cancel, the arm becomes rigid, and it can neither be
positioned precisely nor be destabilised. Rigidity would also be *rewarded* under
v3, because blow-ups now cost w6 and a stiff arm cannot blow up.

This script separates that from the alternatives:
  * Is the policy target-aware at all, or has it collapsed to one fixed posture?
    (correlation between target position and final hand position; spread of final
    positions relative to spread of targets)
  * Does it approach and then drift, or never approach? (error over time)
  * Is it co-contracting? (per-muscle activation, and the Falconer & Winter
    co-contraction index over the elbow flexor/extensor groups)
  * Does the hand actually move? (path length, displacement from start)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl import algorithms as algo_registry
from rl.environment import ArmReachEnv

OUT = os.path.join(config.RUNS_DIR, "SAC_seed0")
N_EPISODES = 30

# Drive the RAW environment, not a VecEnv. A DummyVecEnv auto-resets the instant
# an episode ends, so anything read from the underlying env afterwards is the
# NEXT episode's home pose — which silently turns "final hand position" into a
# constant. The policy still needs VecNormalize's observation statistics, so we
# load them and apply normalize_obs() by hand.
_probe = DummyVecEnv([lambda: ArmReachEnv(domain_rand=False, seed=0)])
vecnorm = VecNormalize.load(os.path.join(OUT, "vecnormalize.pkl"), _probe)
vecnorm.training = False
model = algo_registry.REGISTRY["SAC"]["cls"].load(os.path.join(OUT, "model"))
env = ArmReachEnv(domain_rand=False, seed=0)

targets, finals, starts = [], [], []
err_curves, act_traces, path_lengths, ep_lengths, blew = [], [], [], [], []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    target = env.target.copy()
    start_ee = env.data.site_xpos[env._site_id].copy()
    errs, acts, positions = [], [], []
    while True:
        a, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        acts.append(np.clip(a, 0.0, 1.0))
        obs, r, term, trunc, info = env.step(a)
        positions.append(env.data.site_xpos[env._site_id].copy())
        errs.append(info["err"])
        if term or trunc:
            break
    positions = np.asarray(positions)
    targets.append(target)
    starts.append(start_ee)
    finals.append(positions[-1])
    err_curves.append(errs)
    ep_lengths.append(len(errs))
    blew.append(info["blew_up"])
    act_traces.append(np.asarray(acts))
    path_lengths.append(float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))

print(f"episode lengths: mean {np.mean(ep_lengths):.1f}, "
      f"{sum(blew)}/{N_EPISODES} blew up")

targets = np.asarray(targets)
finals = np.asarray(finals)
starts = np.asarray(starts)

print("=" * 70)
print(f"BEHAVIOUR DIAGNOSTIC — {N_EPISODES} episodes")
print("=" * 70)

# ---- 1. Is the policy target-aware, or collapsed to a fixed posture? ----
print("\n1. TARGET AWARENESS")
for i, axis in enumerate("xyz"):
    c = np.corrcoef(targets[:, i], finals[:, i])[0, 1]
    print(f"   corr(target_{axis}, final_hand_{axis}) = {c:+.3f}")
spread_t = np.std(targets, axis=0)
spread_f = np.std(finals, axis=0)
print(f"   target spread  (std xyz): {np.round(spread_t, 3)}")
print(f"   final   spread (std xyz): {np.round(spread_f, 3)}")
ratio = float(np.mean(spread_f / np.maximum(spread_t, 1e-9)))
print(f"   spread ratio = {ratio:.2f}   "
      f"(1.0 = tracks targets fully; ~0 = same posture regardless of target)")

# ---- 2. Approach or drift? ----
# Only full-length episodes. Truncating every episode to the shortest one lets a
# single blown-up episode decide the window for all of them.
full = [e for e, b in zip(err_curves, blew) if not b and len(e) == config.MAX_EPISODE_STEPS]
E = np.asarray(full)
L = E.shape[1]
print(f"\n2. ERROR OVER TIME (mean across {len(full)} full-length episodes)")
marks = [0, L // 10, L // 4, L // 2, 3 * L // 4, L - 1]
for m in marks:
    print(f"   step {m:>3}: {E[:, m].mean():.3f} m")
print(f"   best mid-episode error: {E.min(axis=1).mean():.3f} m at "
      f"mean step {E.argmin(axis=1).mean():.0f}")
print(f"   final error:            {E[:, -1].mean():.3f} m")
drift = E[:, -1].mean() - E.min(axis=1).mean()
print(f"   -> drift after closest approach: {drift:+.3f} m")

# ---- 3. Co-contraction ----
A = np.concatenate(act_traces, axis=0)          # (total_steps, 9)
print("\n3. MUSCLE ACTIVATION (mean over all steps)")
for name, v in zip(config.MUSCLE_NAMES, A.mean(axis=0)):
    bar = "#" * int(round(v * 40))
    print(f"   {name:<14} {v:.3f}  {bar}")
print(f"   overall mean activation: {A.mean():.3f}  "
      f"(posture holding needs ~0.08)")

# Falconer & Winter co-contraction index over the elbow flexor/extensor groups:
# 2 * common area / total area, i.e. how much of the total drive is wasted
# pulling against itself. 0 = pure reciprocal activation, 1 = perfect co-contraction.
ago = A[:, config.CCI_AGONIST_IDX].mean(axis=1)
ant = A[:, config.CCI_ANTAGONIST_IDX].mean(axis=1)
cci = float(2.0 * np.minimum(ago, ant).sum() / max((ago + ant).sum(), 1e-9))
print(f"   elbow co-contraction index (Falconer & Winter): {cci:.3f}")
print("     0 = reciprocal (healthy), 1 = agonist and antagonist fully cancel")

# ---- 4. Does the hand move? ----
disp = np.linalg.norm(finals - starts, axis=1)
need = np.linalg.norm(targets - starts, axis=1)
print("\n4. MOVEMENT")
print(f"   distance the hand needed to travel: {need.mean():.3f} m")
print(f"   net displacement achieved:          {disp.mean():.3f} m")
print(f"   total path length travelled:        {np.mean(path_lengths):.3f} m")
print(f"   -> fraction of required distance covered: {np.mean(disp/need):.2f}")

json.dump(dict(spread_ratio=ratio, cci=cci, mean_activation=float(A.mean()),
               drift=float(drift), mean_final_error=float(E[:, -1].mean()),
               frac_distance_covered=float(np.mean(disp / need))),
          open(os.path.join(OUT, "behaviour_diagnostic.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'behaviour_diagnostic.json')}")
