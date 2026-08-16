"""
Effort-weight sweep: can the 1.71x load-sharing discrepancy be removed by asking
for it, and what does it cost in accuracy?

The finding this tests
----------------------
Chapter 7 established that the learned controller expends 1.71x the minimum
muscular effort required to produce the same joint torques. The natural
objection is that the reward never asked for efficiency, and inspection confirms
it: measured on the trained policy, the per-step reward terms are

    effort term      w2 * mean_i (f_i/Fmax_i)^2 = 0.051
    error cost       w1 * (err^2 + 0.5*err)     = 0.067   (at err = 0.11 m)
    precision bonus  w7 * exp(-err/0.15)        = 1.441   (at err = 0.11 m)

The effort term is roughly 28 times smaller than the precision bonus. It is not
that the reward uses a different criterion from static optimisation --- its
effort term is already sum-of-squared-normalised-force, the Crowninshield--Brand
form, differing only by the mean-versus-sum factor of nine. It is that the term
is weighted so low it cannot influence the solution.

The experiment
--------------
Retrain at increasing effort weight, holding everything else fixed, and measure
both the effort ratio against static optimisation and the reaching accuracy.
This traces a trade-off curve rather than asserting a single outcome, which is
the honest form of the experiment: reducing wasted effort must eventually cost
accuracy, and the question is where the knee lies.

  w2 = 1   (current, already measured: effort 1.71x, final error 0.106 m)
  w2 = 5
  w2 = 20

Three outcomes are distinguishable and all are reportable:

  * effort ratio falls toward 1 with little accuracy loss -> the discrepancy was
    purely a reward-weighting artefact, and DRL does recover minimum-effort load
    sharing when asked. This would materially strengthen the thesis.
  * effort falls but accuracy degrades sharply -> a genuine trade-off exists,
    and the Pareto curve is itself the contribution.
  * effort does not fall -> the discrepancy is structural rather than a matter
    of objective weighting, which is a stronger negative result than the current
    one because it has been isolated.

Run:  python run_effort_sweep.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(8)
config.EVAL_FREQ = 50_000

from rl.train import train
from rl.evaluate import load_model, evaluate

STEPS   = 300_000
SEED    = 0
W2_LIST = [5.0, 20.0]          # w2 = 1.0 already measured
OUT_DIR = os.path.join(config.RUNS_DIR, "effort_sweep")
RESULTS = os.path.join(OUT_DIR, "effort_sweep_results.json")

_LOCK = os.path.join(config.RUNS_DIR, "sweep.lock")
os.makedirs(OUT_DIR, exist_ok=True)
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    sys.exit(f"REFUSING TO START: {_LOCK} exists — another run is in progress.")
os.write(_fd, str(os.getpid()).encode())
os.close(_fd)
import atexit
atexit.register(lambda: os.path.exists(_LOCK) and os.remove(_LOCK))


def tuned_params():
    """The hyperparameters selected by the Optuna study — read, not transcribed."""
    db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
    study = optuna.load_study(study_name="SAC_arm_reach",
                              storage=f"sqlite:///{db}")
    p = dict(study.best_trial.params)
    p["gradient_steps"] = 2 * config.N_ENVS
    return p


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def main():
    params = tuned_params()
    results = load_results()

    print("=" * 78)
    print("EFFORT-WEIGHT SWEEP — does asking for efficiency remove the 1.71x gap?")
    print("=" * 78)
    print(f"  {STEPS:,} steps, seed {SEED}, tuned hyperparameters, "
          f"all reward weights fixed except w2")
    print(f"  baseline already measured: w2=1.0 -> effort 1.71x, "
          f"final error 0.106 m\n", flush=True)

    for w2 in W2_LIST:
        key = f"w2_{w2:g}"
        if key in results:
            print(f"  {key}: already complete, skipping")
            continue

        weights = dict(config.REWARD_WEIGHTS)
        weights["w2"] = w2
        out = os.path.join(OUT_DIR, key)

        print(f"\n=== {key} (effort weight {w2:g}) ===", flush=True)
        t0 = time.perf_counter()
        train("SAC", out_dir=out, seed=SEED, total_steps=STEPS,
              hyperparams=params, reward_weights=weights)
        wall = (time.perf_counter() - t0) / 60

        # Evaluate under the REPORTED protocol: default reward weights, so the
        # metrics stay comparable with every other run in this project. Only the
        # TRAINING objective differed.
        model, envs = load_model(out, "SAC")
        m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=SEED)
        envs.close()

        results[key] = {"w2": w2, "steps": STEPS, "wall_min": wall,
                        "metrics": {k: float(v) for k, v in m.items()
                                    if isinstance(v, (int, float))}}
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  -> final_err={m['mean_final_error']:.4f}  "
              f"min_err={m['mean_min_error']:.4f}  "
              f"energy={m['mean_energy']:.0f}  ({wall:.0f} min)", flush=True)
        print(f"  run force_comparison.py --run {out} for the effort ratio",
              flush=True)

    print("\n" + "=" * 78)
    print("SWEEP COMPLETE — now measure the effort ratio for each:")
    for w2 in W2_LIST:
        print(f"  python force_comparison.py --run "
              f"{os.path.join(OUT_DIR, f'w2_{w2:g}')} --episodes 20")
    print("=" * 78)


if __name__ == "__main__":
    main()
