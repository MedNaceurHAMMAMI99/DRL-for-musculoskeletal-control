"""
Small four-algorithm benchmark — exploratory, not the thesis grid.

Question
--------
Which of SAC / TD3 / DDPG / PPO actually does best on this task? Nothing in this
project has ever answered that: the only four-way run was the July 12k-step
smoke test, in which all four scored 0% and every Wilcoxon comparison tied at
p=1.0, under a reward later proven exploitable and a model with its joints
locked in degrees. That data is void (see ALGORITHM_COMPARISON_NOTES.md §1).

Design
------
FIVE configurations, not four. `SAC_default` is included alongside `SAC_tuned`
because the decisive finding so far is that SAC's DEFAULT `target_entropy`
(= -dim(A) = -9) is what prevented the policy from settling, and the search
selected ~-19. Without the default arm, a SAC win would be unattributable
between "SAC is the better algorithm" and "SAC is the only algorithm we tuned".
With it, the tuning effect and the algorithm effect are separately visible.

  SAC_default   registry defaults            (target_entropy auto = -9)
  SAC_tuned     Optuna trial 12 parameters   (target_entropy = -19.6)
  TD3           registry defaults
  DDPG          registry defaults
  PPO           registry defaults

Replay ratio is held at 1.87 (`gradient_steps = 2*N_ENVS`) for every off-policy
configuration, so it is NOT a confound between them — it is the setting the
confirmation run used, and §17 established that a lower ratio is simply a bug.
PPO is on-policy and has no such parameter.

What this can and cannot support
--------------------------------
* CAN: a preliminary ranking, an estimate of seed variance, and a measurement
  of how much of SAC's performance is tuning rather than algorithm.
* CANNOT: statistical significance. Three seeds is not enough for a Wilcoxon
  test to be meaningful, and none is computed here. Do not put p-values from
  this run in the thesis.
* CONFOUND to report, not hide: at 100k steps PPO performs only ~12 policy
  updates (n_steps=2048, N_ENVS=4) against ~187k gradient steps for the
  off-policy algorithms. That IS the sample-efficiency difference under an
  equal-STEP budget, and must be reported as such rather than as "PPO is worse".
* TD3/DDPG remain untuned, so a SAC_tuned win over them is partly a tuning
  result. ALGORITHM_COMPARISON_NOTES.md §4 lists the options for fixing that.

Resumability
------------
Results are written after EVERY run, and any run whose model.zip already exists
is skipped. The job can be killed at any point and restarted without losing
completed work, and partial results are readable throughout.

Run:  python run_benchmark.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(8)          # measured fastest; one process only

config.EVAL_FREQ = 50_000         # coarse curve; this run is about final metrics

from rl.train import train
from rl.evaluate import load_model, evaluate

SEEDS      = [0, 1, 2]
STEPS      = 100_000
BENCH_DIR  = os.path.join(config.RUNS_DIR, "bench")
RESULTS    = os.path.join(BENCH_DIR, "benchmark_results.json")
RATIO_GS   = 2 * config.N_ENVS    # replay ratio 1.87, as in the confirmation run

_LOCK = os.path.join(config.RUNS_DIR, "bench.lock")
os.makedirs(BENCH_DIR, exist_ok=True)
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    sys.exit(f"REFUSING TO START: {_LOCK} exists. Another run is in progress.")
os.write(_fd, str(os.getpid()).encode())
os.close(_fd)
import atexit
atexit.register(lambda: os.path.exists(_LOCK) and os.remove(_LOCK))


def tuned_sac_params() -> dict:
    """Optuna's selected parameters, read from the study — never transcribed."""
    db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
    study = optuna.load_study(study_name="SAC_arm_reach",
                              storage=f"sqlite:///{db}")
    p = dict(study.best_trial.params)
    p["gradient_steps"] = RATIO_GS
    return p


CONFIGS = {
    "SAC_default": ("SAC",  {"gradient_steps": RATIO_GS}),
    "SAC_tuned":   ("SAC",  tuned_sac_params()),
    "TD3":         ("TD3",  {"gradient_steps": RATIO_GS}),
    "DDPG":        ("DDPG", {"gradient_steps": RATIO_GS}),
    "PPO":         ("PPO",  {}),          # on-policy: no replay ratio
}

METRICS = ["mean_final_error", "mean_min_error", "success_rate",
           "reach_5cm", "reach_10cm", "touch_2cm", "touch_5cm",
           "mean_energy", "blow_up_rate", "mean_episode_len"]


def load_results() -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_results(r: dict):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=2)
    os.replace(tmp, RESULTS)


def summarise(results: dict):
    """Mean +- std across seeds, ranked by mean_final_error."""
    rows = []
    for name in CONFIGS:
        vals = {m: [] for m in METRICS}
        walls = []
        for s in SEEDS:
            k = f"{name}_seed{s}"
            if k in results:
                for m in METRICS:
                    vals[m].append(results[k]["metrics"][m])
                walls.append(results[k]["wall_min"])
        if vals["mean_final_error"]:
            rows.append((name, vals, len(walls), float(np.mean(walls))))
    rows.sort(key=lambda r: np.mean(r[1]["mean_final_error"]))

    print("\n" + "=" * 92)
    print("BENCHMARK — mean +- std across seeds, ranked by final error "
          f"({STEPS:,} steps/run)")
    print("=" * 92)
    hdr = f"  {'config':<13}{'n':>2} {'final err':>16}{'min err':>16}{'reach5cm':>11}{'succ':>8}{'min':>7}"
    print(hdr)
    print("  " + "-" * 88)
    for name, v, n, wall in rows:
        fe, me = np.array(v["mean_final_error"]), np.array(v["mean_min_error"])
        r5, sr = np.array(v["reach_5cm"]), np.array(v["success_rate"])
        print(f"  {name:<13}{n:>2} "
              f"{fe.mean():>8.4f}+-{fe.std():<6.4f}"
              f"{me.mean():>8.4f}+-{me.std():<6.4f}"
              f"{r5.mean():>10.2f} {sr.mean():>7.2f} {wall:>6.0f}")

    if rows:
        print(f"\n  Best by final error: {rows[0][0]}")
        names = [r[0] for r in rows]
        if "SAC_tuned" in names and "SAC_default" in names:
            t = np.mean(dict((r[0], r[1]) for r in rows)["SAC_tuned"]["mean_final_error"])
            d = np.mean(dict((r[0], r[1]) for r in rows)["SAC_default"]["mean_final_error"])
            print(f"  Tuning effect within SAC: {d:.4f} -> {t:.4f} m "
                  f"({100*(d-t)/d:+.1f}%) — this is the target_entropy result, "
                  f"not an algorithm difference.")
    print("\n  Reminders: no significance testing (3 seeds is too few); TD3/DDPG/PPO")
    print("  are UNTUNED; PPO gets ~12 policy updates at this budget vs ~187k")
    print("  gradient steps off-policy. See ALGORITHM_COMPARISON_NOTES.md.")
    print("  Privileged oracle: median final error 0.08 m, touch_2cm 0.27.")


def main():
    results = load_results()
    todo = [(n, s) for n in CONFIGS for s in SEEDS
            if f"{n}_seed{s}" not in results]
    print(f"Benchmark: {len(CONFIGS)} configs x {len(SEEDS)} seeds x "
          f"{STEPS:,} steps — {len(todo)} run(s) remaining, "
          f"{len(results)} already done.")
    print(f"Replay ratio {RATIO_GS/config.N_ENVS:.2f} for all off-policy configs; "
          f"8 torch threads, single process.\n", flush=True)

    for name, seed in todo:
        algo, params = CONFIGS[name]
        out = os.path.join(BENCH_DIR, f"{name}_seed{seed}")
        print(f"\n=== {name} seed {seed} ({algo}) ===", flush=True)
        t0 = time.perf_counter()
        train(algo, out_dir=out, seed=seed, total_steps=STEPS,
              hyperparams=params or None)
        wall = (time.perf_counter() - t0) / 60

        model, envs = load_model(out, algo)
        m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
        envs.close()

        results[f"{name}_seed{seed}"] = {
            "config": name, "algo": algo, "seed": seed, "steps": STEPS,
            "hyperparams": params, "wall_min": wall,
            "metrics": {k: float(m[k]) for k in METRICS},
        }
        save_results(results)          # after EVERY run, so a kill loses nothing
        print(f"  -> final_err={m['mean_final_error']:.4f}  "
              f"min_err={m['mean_min_error']:.4f}  "
              f"succ={m['success_rate']:.2f}  ({wall:.0f} min)", flush=True)
        summarise(results)

    summarise(results)


if __name__ == "__main__":
    main()
