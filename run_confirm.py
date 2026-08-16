"""
Confirmation run — do the search's best parameters hold at full budget?

The search selected on a 100,000-step budget scored over 20 evaluation
episodes. The pilots are 300,000 steps scored over 50. Those are different
protocols, and the search's headline (mean_final_error 0.1177 m,
mean_min_error 0.0839 m) cannot be compared to pilot 6's (0.204 / 0.132)
until it is re-measured under the pilot protocol. That is all this run does.

It matters because mean_min_error is the number that had never moved:
0.178 -> 0.134 -> 0.132 across pilots 4, 5 and 6, through three reward
redesigns and the 4x replay-ratio correction. The search reports 0.0839, which
is at the privileged oracle's median final error (0.08 m). If that survives 50
episodes at 300k steps, the approach-precision plateau is genuinely broken and
the grid is worth its compute. If it does not, it was a 20-episode artifact and
the plateau stands.

Parameters are READ FROM THE STUDY, never transcribed, so this run cannot
silently disagree with the search it is confirming.

Everything else is held at the pilot-6 protocol: SAC, seed 0, 300k steps,
reward v4, FRAME_SKIP=10, replay ratio 1.87, domain randomisation on for
training, evaluation deterministic with domain randomisation off, 50 episodes.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(8)

config.EVAL_FREQ = 25_000

from rl.train import train
from rl.evaluate import load_model, evaluate
from rl import algorithms as algo_registry

OUT   = os.path.join(config.RUNS_DIR, "SAC_seed0")
STEPS = 300_000

# ── Single-instance lock (see run_pilot6.py for why this exists) ─────────────
_LOCK = os.path.join(config.RUNS_DIR, "confirm.lock")
os.makedirs(config.RUNS_DIR, exist_ok=True)
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    sys.exit(f"REFUSING TO START: {_LOCK} exists. Another run is in progress.")
os.write(_fd, str(os.getpid()).encode())
os.close(_fd)
import atexit
atexit.register(lambda: os.path.exists(_LOCK) and os.remove(_LOCK))

if os.path.exists(os.path.join(OUT, "checkpoint", "progress.json")):
    sys.exit(f"REFUSING TO START: {OUT}/checkpoint/ holds a resumable run. "
             "train() resumes by default — archive it first.")

# ── Best parameters, read from the study ────────────────────────────────────
db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
study = optuna.load_study(study_name="SAC_arm_reach", storage=f"sqlite:///{db}")
best = study.best_trial
params = dict(best.params)
# gradient_steps is fixed rather than searched, so it is not in trial.params;
# it must still be passed or the run silently reverts to the registry default.
params["gradient_steps"] = 2 * config.N_ENVS

print("=" * 74)
print("CONFIRMATION RUN — search best params at the pilot protocol")
print("=" * 74)
print(f"  source: study trial {best.number}, "
      f"mean_final_error {best.value:.4f} m at 100k steps / 20 episodes")
print(f"  budget: {STEPS:,} steps, {config.EVAL_EPISODES} evaluation episodes")
print("  params:")
for k, v in sorted(params.items()):
    print(f"    {k:<16} {v}")
print("\n  search-reported metrics to be confirmed:")
for k in ("mean_min_error", "drift", "reach_10cm", "touch_5cm", "mean_energy"):
    if k in best.user_attrs:
        print(f"    {k:<16} {best.user_attrs[k]:.4f}")
print(flush=True)

train("SAC", out_dir=OUT, seed=0, total_steps=STEPS, hyperparams=params)

results = {}
for label, run_dir in (("final", OUT), ("best", os.path.join(OUT, "best"))):
    if not os.path.exists(os.path.join(run_dir, "model.zip")):
        continue
    print(f"\nEvaluating [{label}] (deterministic, no domain randomisation)...",
          flush=True)
    model, envs = load_model(run_dir, "SAC")
    results[label] = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=0)

with open(os.path.join(OUT, "confirm_eval.json"), "w") as f:
    json.dump({"study_trial": best.number, "params": params,
               "search_value": best.value,
               "search_user_attrs": dict(best.user_attrs),
               "results": results}, f, indent=2)

# Pilot 6: same protocol, registry defaults, replay ratio 0.93.
PILOT6 = {"blow_up_rate": 0.0, "mean_episode_len": 100.0, "success_rate": 0.0,
          "mean_final_error": 0.2040, "mean_min_error": 0.1316, "reach_2cm": 0.0,
          "reach_5cm": 0.0, "reach_10cm": 0.18, "touch_2cm": 0.02,
          "touch_5cm": 0.18, "mean_energy": 774667.0, "mean_reward": 493.53}

keys = ["blow_up_rate", "mean_episode_len", "success_rate", "mean_final_error",
        "mean_min_error", "reach_2cm", "reach_5cm", "reach_10cm",
        "touch_2cm", "touch_5cm", "mean_energy", "mean_reward"]

print("\n" + "=" * 74)
print("CONFIRMATION RESULT — 300k steps, 50 episodes")
print("=" * 74)
labels = list(results)
print(f"  {'metric':>18}{'pilot6':>13}" + "".join(f"{l:>13}" for l in labels))
for k in keys:
    row = "".join(f"{results[l][k]:>13.4f}" for l in labels)
    print(f"  {k:>18}{PILOT6[k]:>13.4f}{row}")

print("\nVERDICT — does the approach-precision improvement survive?")
if results:
    best_min = min(r["mean_min_error"] for r in results.values())
    print(f"  mean_min_error: pilot6 {PILOT6['mean_min_error']:.4f} -> "
          f"this run {best_min:.4f}   (search claimed 0.0839; oracle median 0.08)")
    if best_min < 0.10:
        print("  CONFIRMED at full budget. The plateau that survived three reward")
        print("  redesigns and the replay-ratio fix is broken. Size the grid.")
    elif best_min < PILOT6["mean_min_error"]:
        print("  Partially confirmed — better than pilot 6 but short of the")
        print("  search's claim. The 20-episode figure was optimistic.")
    else:
        print("  NOT CONFIRMED. The search's improvement was a 20-episode/100k")
        print("  artifact and the approach-precision plateau still stands.")

print("\nReference: the privileged oracle reaches <2 cm at some point in 27% of")
print("episodes with a median final error of 0.08 m.")
