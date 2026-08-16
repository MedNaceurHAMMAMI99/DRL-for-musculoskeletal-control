"""
Diagnostic: is the policy deliberately ending episodes early?

Hypothesis under test. Every per-step reward in environment.py is <= 0 (the
weights make it a pure cost; w4, the old alive bonus, is 0), and a numerical
blow-up sets `terminated=True` with NO penalty. In an episodic MDP `terminated`
means "bootstrap nothing" — the return from that point is exactly 0. So an agent
that destabilises the simulation collects 0 for the rest of the episode instead
of ~100 steps of negative reward. If that saving exceeds the w5=10 success bonus,
the optimal policy is to blow the arm up as fast as possible, and reaching is
irrelevant.

Circumstantial evidence that this is happening: the MuJoCo NaN-QACC warnings
appear only after ~700k steps and grow more frequent (a learned behaviour, not a
random glitch); the policy sits at ~48% RMS activation where 8% holds posture;
and pilot3's mean episode reward (-4.04) is far less negative than 100 steps of
the observed per-step cost would give (~-18), which only makes sense if episodes
are ending early.

This script measures it directly, per episode: length, why it ended, mean
activation, and the true error at the final step (as opposed to the error
environment.py reports after its blow-up reset, which is a separate bug).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from rl.evaluate import load_model

OUT = os.path.join(config.RUNS_DIR, "SAC_seed0")
N_EPISODES = 50

model, envs = load_model(OUT, "SAC")
raw = envs.venv.envs[0].unwrapped          # the ArmReachEnv underneath

rows = []
for ep in range(N_EPISODES):
    obs = envs.reset()
    done = [False]
    acts, errs = [], []
    steps = 0
    # Track the true error each step, before any blow-up reset can overwrite it.
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        acts.append(np.clip(action[0], 0.0, 1.0))
        true_err = float(np.linalg.norm(
            raw.target - raw.data.site_xpos[raw._site_id]))
        obs, r, done, infos = envs.step(action)
        errs.append(true_err)
        steps += 1
    info = infos[0]
    # An episode that stopped before max_steps without success terminated on the
    # blow-up guard — that is the behaviour under investigation.
    success = bool(info.get("success", False))
    truncated = steps >= config.MAX_EPISODE_STEPS
    blew_up = (not success) and (not truncated)
    rows.append(dict(
        ep=ep, steps=steps, success=success, blew_up=blew_up,
        mean_act=float(np.mean(acts)), max_act=float(np.max(acts)),
        err_reported=float(info.get("err", np.nan)),
        err_true_last=float(errs[-1]),
    ))

steps_arr = np.array([r["steps"] for r in rows])
blew = np.array([r["blew_up"] for r in rows])
succ = np.array([r["success"] for r in rows])
rep = np.array([r["err_reported"] for r in rows])
tru = np.array([r["err_true_last"] for r in rows])
act = np.array([r["mean_act"] for r in rows])

print("=" * 68)
print(f"TERMINATION DIAGNOSTIC — {N_EPISODES} episodes, "
      f"max_steps={config.MAX_EPISODE_STEPS}")
print("=" * 68)
print(f"  episode length: mean {steps_arr.mean():.1f}  median {np.median(steps_arr):.0f}"
      f"  min {steps_arr.min()}  max {steps_arr.max()}")
print(f"  ended by BLOW-UP : {100*blew.mean():5.1f}%  ({blew.sum()}/{N_EPISODES})")
print(f"  ended by SUCCESS : {100*succ.mean():5.1f}%")
print(f"  ran full episode : {100*(~blew & ~succ).mean():5.1f}%")
print()
print(f"  mean activation across episodes: {act.mean():.3f} "
      f"(posture-holding needs ~0.08)")
print()
print("  Final-error metric contamination (blow-up resets the sim first):")
print(f"    reported mean_final_error : {np.nanmean(rep):.4f} m")
print(f"    TRUE   mean_final_error   : {np.nanmean(tru):.4f} m")
if blew.any():
    print(f"    on blown-up episodes only : reported {np.nanmean(rep[blew]):.4f} m"
          f"  vs true {np.nanmean(tru[blew]):.4f} m")

print()
print("  Per-episode detail (first 15):")
print("    ep  steps  blew_up  mean_act  err_reported  err_true")
for r in rows[:15]:
    print(f"    {r['ep']:>2}  {r['steps']:>5}  {str(r['blew_up']):>7}  "
          f"{r['mean_act']:>8.3f}  {r['err_reported']:>12.4f}  {r['err_true_last']:>8.4f}")

with open(os.path.join(OUT, "termination_diagnostic.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 'termination_diagnostic.json')}")
