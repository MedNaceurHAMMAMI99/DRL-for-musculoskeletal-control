"""
Pilot 3 — the 50 Hz (FRAME_SKIP=10) validation run, relaunched.

This is the rerun of the SAC seed-0 1e6-step run that a forced Windows-Update
reboot destroyed on 2026-08-14 at ~900k steps. Its purpose is unchanged: decide
whether the 10x increase in reaching attempts per step budget (config.FRAME_SKIP)
actually fixes the precision plateau, BEFORE committing ~40-50 h to the full
4-algorithm x 10-seed grid.

Unlike the lost run this one is resumable (rl/train.py checkpoints every
config.CHECKPOINT_FREQ steps), so re-running this script after an interruption
continues where it stopped instead of starting over.

It also runs the real 50-episode evaluation at the end, which the lost run never
reached — the console learning curve alone cannot answer the precision question,
because it uses only 5 episodes per point and no graded-accuracy measures.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from rl.train import train
from rl.evaluate import load_model, evaluate

OUT = os.path.join(config.RUNS_DIR, "SAC_seed0")

print(f"PILOT3 — SAC seed 0, {config.TRAIN_STEPS:,} steps, "
      f"FRAME_SKIP={config.FRAME_SKIP} ({config.MAX_EPISODE_STEPS} agent steps/episode)",
      flush=True)

train("SAC", out_dir=OUT, seed=0, total_steps=config.TRAIN_STEPS)

print("\nEvaluating (deterministic, no domain randomisation)...", flush=True)
model, envs = load_model(OUT, "SAC")
res = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=0)

with open(os.path.join(OUT, "pilot3_eval.json"), "w") as f:
    json.dump(res, f, indent=2)

summary = {k: v for k, v in res.items() if not k.endswith("_per_ep")}
print("\nPILOT3 RESULT:")
for k, v in summary.items():
    print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
print("\nComparison targets — lost 50 Hz run reached err 0.160 @300k (5-ep curve);")
print("the earlier 1-action-per-physics-step 1e6 run plateaued at 0.295 m, 0% success.")
