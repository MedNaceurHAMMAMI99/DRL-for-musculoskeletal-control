"""
Pilot 4 — the first run on the v3 reward, and the first run of this project in
which the agent is actually incentivised to attempt the task.

Pilots 1-3 are void. diagnose_termination.py established that under the v2
reward the optimal policy was to destroy the simulation on step ~7 of 100, and
that the policy had learned exactly that in 100% of evaluation episodes. Their
error numbers describe a flailing arm 0.14 s into a deliberate self-termination,
not reaching performance.

What changed (all verified by test_reward_v3.py, 14 checks):
  * per-step reward is strictly positive, so quitting early forfeits value
  * success no longer terminates — the task is reach AND hold
  * blow-up costs w6 and is detected via MuJoCo's warning counters, which is the
    only reliable signal since MuJoCo silently self-resets on divergence
  * blown-up episodes report their last finite error, not the home pose
  * domain randomisation varies per body/muscle instead of one global scale

This run gates the full 4-algorithm x 10-seed grid (~50 h). The gate is not
"did the number improve" but "is the agent solving the task": blow_up_rate 0,
full-length episodes, and a monotone fall in error.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from rl.train import train
from rl.evaluate import load_model, evaluate

OUT = os.path.join(config.RUNS_DIR, "SAC_seed0")

print(f"PILOT4 — SAC seed 0, {config.TRAIN_STEPS:,} steps, reward v3, "
      f"FRAME_SKIP={config.FRAME_SKIP} ({config.MAX_EPISODE_STEPS} agent steps/episode)",
      flush=True)

train("SAC", out_dir=OUT, seed=0, total_steps=config.TRAIN_STEPS)

results = {}
# Report the final policy AND the best validation checkpoint. Selection happened
# on the seed+10_000 validation stream inside EvalCurveCallback; evaluation here
# runs on the seed-0 stream, so the reported number is not the selected-on one.
for label, run_dir in (("final", OUT), ("best", os.path.join(OUT, "best"))):
    if not os.path.exists(os.path.join(run_dir, "model.zip")):
        continue
    print(f"\nEvaluating [{label}] (deterministic, no domain randomisation)...",
          flush=True)
    model, envs = load_model(run_dir, "SAC")
    results[label] = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=0)

with open(os.path.join(OUT, "pilot4_eval.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 62)
print("PILOT4 RESULT")
print("=" * 62)
keys = ["blow_up_rate", "mean_episode_len", "success_rate", "mean_final_error",
        "mean_min_error", "reach_2cm", "reach_5cm", "reach_10cm",
        "touch_2cm", "touch_5cm", "mean_energy", "mean_reward"]
print(f"  {'metric':>18}" + "".join(f"{lab:>12}" for lab in results))
for k in keys:
    row = "".join(f"{results[lab][k]:>12.4f}" for lab in results)
    print(f"  {k:>18}{row}")

print("\nGATE for the full grid — all three must hold:")
for lab, r in results.items():
    print(f"  [{lab}] blow_up_rate={r['blow_up_rate']:.2f} (need 0.00), "
          f"episode_len={r['mean_episode_len']:.0f} (need {config.MAX_EPISODE_STEPS}), "
          f"final_error={r['mean_final_error']:.3f} m")
print("\nReference: the privileged oracle (reach_oracle.py) reaches <2 cm at some")
print("point in 27% of episodes with a median final error of 0.08 m.")
