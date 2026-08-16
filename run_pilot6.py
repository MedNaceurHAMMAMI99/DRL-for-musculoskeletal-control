"""
Pilot 6 — the replay-ratio experiment.

ONE variable changes from pilot 5: the number of gradient updates per
environment step. Everything else — the v4 reward, FRAME_SKIP=10, the model,
the evaluation protocol, seed 0 — is identical, so whatever this run does is
attributable to the replay ratio and to nothing else. Pilot 5 changed three
coupled reward terms at once and consequently could not identify which of them
mattered; this run does not repeat that.

The bug being tested
--------------------
`gradient_steps=1` with `config.N_ENVS = 4` gave a replay ratio of 0.233
updates per environment step instead of the standard ~1 (measured, not
inferred: verify_replay_ratio.py). Every run before today therefore did about
233,000 gradient updates for a nominal 1,000,000-step budget.

Budget, and why it is 300k and not 1e6
--------------------------------------
The fix quadruples gradient work per environment step, and gradient updates
dominate wall-clock (14.4 ms/update at 8 threads vs ~7 us for a physics step —
bench_threads.py). A 1e6-step run would now take ~3.7 h.

300,000 steps at the corrected ratio is ~280,000 gradient updates in ~70 min:
MORE gradient work than pilot 5's entire 1e6-step, 80-minute run (233,000),
from 3.3x fewer environment samples. That makes the comparison sharp in both
directions:

  * at the SAME env-step count (300k), pilot 5 had done only ~70k updates and
    was at 0.215 m;
  * at the SAME wall-clock (~75 min), pilot 5 finished its whole 1e6-step run
    at 0.254 m final / 0.215 m best.

Beating either from 300k steps means the replay ratio was the binding
constraint. Failing to means it was not, and the precision plateau has a cause
that is still unidentified — which is equally worth knowing before any grid.

The evaluation curve is logged every 25k steps rather than the usual 100k, so a
300k run still yields 12 points and the question "does it converge" is
answerable. Non-convergence of this curve is the symptom the whole diagnosis
rests on.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import config

# Measured fastest on this machine (bench_threads.py): 14.4 ms/update at 8
# threads vs 25.7 at 1. Set explicitly — the pilots never set it, and leaving it
# implicit is what made the four-worker Optuna launch oversubscribe the CPU.
torch.set_num_threads(8)

# Finer curve for a shorter run (see docstring). Mutated before train() reads it.
config.EVAL_FREQ = 25_000

from rl.train import train
from rl.evaluate import load_model, evaluate

OUT   = os.path.join(config.RUNS_DIR, "SAC_seed0")
STEPS = 300_000

if os.path.exists(os.path.join(OUT, "checkpoint", "progress.json")):
    sys.exit(
        f"REFUSING TO START: {OUT}/checkpoint/ already holds a resumable run.\n"
        "train() resumes by default, so this would continue an older policy "
        "instead of testing the fix. Archive that directory first.")

# Single-instance lock. Two copies of this script were once started against the
# same out_dir (a process check matched "python" while the interpreter is named
# "python3.12", so the first launch was wrongly believed dead). They interleaved
# writes to the same checkpoint and together saturated all 16 threads — the exact
# sustained-load condition under which this machine hard-locked earlier the same
# day. The checkpoint guard above cannot catch it because it only trips once
# 50k steps have been written.
_LOCK = os.path.join(config.RUNS_DIR, "pilot6.lock")
os.makedirs(config.RUNS_DIR, exist_ok=True)
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    try:
        with open(_LOCK) as f:
            _owner = f.read().strip()
    except OSError:
        _owner = "unknown"
    sys.exit(f"REFUSING TO START: {_LOCK} exists (held by PID {_owner}).\n"
             "Another run is in progress. If it is definitely dead, delete the "
             "lock file and retry.")
os.write(_fd, str(os.getpid()).encode())
os.close(_fd)

import atexit
atexit.register(lambda: os.path.exists(_LOCK) and os.remove(_LOCK))

print(f"PILOT6 — SAC seed 0, {STEPS:,} steps, replay ratio FIXED "
      f"(gradient_steps=-1, ~0.93 updates/step)", flush=True)
print(f"  reward v4 unchanged, FRAME_SKIP={config.FRAME_SKIP}, "
      f"{config.MAX_EPISODE_STEPS} agent steps/episode", flush=True)
print(f"  expected ~{0.933*STEPS:,.0f} gradient updates "
      f"(pilot 5 did ~233,000 in 1,000,000 steps)", flush=True)

train("SAC", out_dir=OUT, seed=0, total_steps=STEPS)

results = {}
for label, run_dir in (("final", OUT), ("best", os.path.join(OUT, "best"))):
    if not os.path.exists(os.path.join(run_dir, "model.zip")):
        continue
    print(f"\nEvaluating [{label}] (deterministic, no domain randomisation)...",
          flush=True)
    model, envs = load_model(run_dir, "SAC")
    results[label] = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=0)

with open(os.path.join(OUT, "pilot6_eval.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 74)
print("PILOT6 RESULT — replay ratio 0.93 (was 0.23)")
print("=" * 74)
keys = ["blow_up_rate", "mean_episode_len", "success_rate", "mean_final_error",
        "mean_min_error", "reach_2cm", "reach_5cm", "reach_10cm",
        "touch_2cm", "touch_5cm", "mean_energy", "mean_reward"]

# Pilot 5, for a like-for-like column: 1e6 env steps, ~233k updates, 80.6 min.
PILOT5 = {"blow_up_rate": 0.02, "mean_episode_len": 98.54, "success_rate": 0.0,
          "mean_final_error": 0.2540, "mean_min_error": 0.1344, "reach_2cm": 0.0,
          "reach_5cm": 0.02, "reach_10cm": 0.14, "touch_2cm": 0.0,
          "touch_5cm": 0.14, "mean_energy": 1250576.0, "mean_reward": 473.87}

labels = list(results)
print(f"  {'metric':>18}{'pilot5(1e6)':>14}" + "".join(f"{l:>12}" for l in labels))
for k in keys:
    row = "".join(f"{results[l][k]:>12.4f}" for l in labels)
    print(f"  {k:>18}{PILOT5[k]:>14.4f}{row}")

print("\nVERDICT")
if results:
    best_err = min(r["mean_final_error"] for r in results.values())
    if best_err < PILOT5["mean_final_error"] * 0.85:
        print(f"  Final error {best_err:.3f} m beats pilot 5's {PILOT5['mean_final_error']:.3f} m")
        print("  from 3.3x FEWER environment samples. The replay ratio was a")
        print("  binding constraint. Scale the budget and size the grid.")
    elif best_err < PILOT5["mean_final_error"]:
        print(f"  Final error {best_err:.3f} m is better than pilot 5's "
              f"{PILOT5['mean_final_error']:.3f} m but not decisively.")
        print("  The ratio mattered; something else still binds. Do NOT launch the grid.")
    else:
        print(f"  Final error {best_err:.3f} m does NOT beat pilot 5's "
              f"{PILOT5['mean_final_error']:.3f} m.")
        print("  The replay ratio was not the binding constraint. The precision")
        print("  plateau has a cause that is still unidentified.")

print("\nReference: the privileged oracle (reach_oracle.py) reaches <2 cm at some")
print("point in 27% of episodes with a median final error of 0.08 m.")
