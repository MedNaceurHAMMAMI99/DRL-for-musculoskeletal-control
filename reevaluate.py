"""
Re-evaluate every trained policy with the corrected evaluation code, across
several target seeds.

Why this is necessary
---------------------
Four defects were found in the evaluation path:

  * `load_model` hardcoded the environment seed, so `evaluate(seed=...)` was
    dead code and every configuration was scored on one fixed set of targets.
  * `evaluate` reset the vectorised environment at the top of each loop
    iteration in addition to DummyVecEnv's own auto-reset, consuming two targets
    per episode and scoring on every other one -- a different test set from the
    one force_comparison.py and run_conventional.py used.
  * `load_model` ignored `config.DEVICE`, so SB3's device="auto" selected CUDA
    whenever it was available.
  * A diverged episode could satisfy the success criterion, because `v` is
    forced to 0.0 on divergence.

The first two change which episodes the reported numbers describe, so every
evaluation is repeated here.

What varying the target seed buys
---------------------------------
Measurement on a single policy showed the target draw contributes 3-4x more
variance than the training seed: final error 0.1130 +- 0.0146 across five target
seeds against 0.1033 +- 0.0034 across five training seeds. Every uncertainty
previously reported in this work was training-seed spread on one fixed test set,
and therefore understated.

Reporting both components separately is the fix. Note that between-configuration
comparisons were, and remain, PAIRED on identical targets -- that part was sound.

Run:  python reevaluate.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(8)

import numpy as np

import config
from rl.evaluate import load_model, evaluate

TARGET_SEEDS = [0, 1, 2]
OUT = os.path.join(config.RUNS_DIR, "reevaluation.json")

R = config.RUNS_DIR
GROUPS = {
    "w2_1": [("SAC", os.path.join(R, "SAC_seed0"))] +
            [("SAC", os.path.join(R, "replication", f"SAC_seed{s}")) for s in (1, 2, 3, 4)],
    "w2_5": [("SAC", os.path.join(R, "effort_sweep", "w2_5"))] +
            [("SAC", os.path.join(R, "w2_replication", f"w2_5_seed{s}")) for s in (1, 2, 3, 4)],
    "w2_20": [("SAC", os.path.join(R, "effort_sweep", "w2_20"))] +
             [("SAC", os.path.join(R, "w2_replication", f"w2_20_seed{s}")) for s in (1, 2, 3, 4)],
}
for algo in ("SAC", "TD3", "DDPG", "PPO"):
    GROUPS[f"fair_{algo}"] = [(algo, os.path.join(R, "fair_bench", f"{algo}_seed{s}"))
                              for s in (0, 1, 2)]

METRICS = ("mean_final_error", "mean_min_error", "success_rate",
           "reach_5cm", "reach_10cm", "touch_2cm", "mean_energy",
           "blow_up_rate", "mean_episode_len")


def main():
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except (OSError, json.JSONDecodeError): pass

    t0 = time.perf_counter()
    for group, runs in GROUPS.items():
        for algo, run in runs:
            if not os.path.exists(os.path.join(run, "model.zip")):
                print(f"  [skip missing] {run}"); continue
            for ts in TARGET_SEEDS:
                key = f"{group}|{os.path.basename(run)}|t{ts}"
                if key in res:
                    continue
                model, envs = load_model(run, algo, target_seed=ts, device="cpu")
                m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES)
                envs.close()
                res[key] = {"group": group, "run": os.path.basename(run),
                            "algo": algo, "target_seed": ts,
                            **{k: float(m[k]) for k in METRICS}}
                tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=2)
                os.replace(tmp, OUT)
            print(f"  {group:<12} {os.path.basename(run):<18} done "
                  f"({(time.perf_counter()-t0)/60:.0f} min elapsed)", flush=True)

    # ── report: separate training-seed and target-seed variance ────────────
    print("\n" + "=" * 88)
    print("RE-EVALUATION with corrected code — variance decomposed")
    print("=" * 88)
    print(f"  {'group':<12}{'metric':<20}{'mean':>9}{'sd(train)':>12}"
          f"{'sd(target)':>12}{'sd(total)':>12}")
    print("  " + "-" * 82)
    summary = {}
    for group in GROUPS:
        rows = [v for v in res.values() if v["group"] == group]
        if not rows:
            continue
        runs = sorted({v["run"] for v in rows})
        summary[group] = {}
        for met in ("mean_final_error", "mean_min_error", "reach_5cm", "reach_10cm"):
            M = np.array([[next(v[met] for v in rows
                                if v["run"] == r and v["target_seed"] == t)
                           for t in TARGET_SEEDS] for r in runs])   # runs x targets
            # training-seed sd: spread of per-run means (averaging out targets)
            sd_train = float(M.mean(axis=1).std(ddof=1)) if len(runs) > 1 else float("nan")
            # target-seed sd: spread of per-target means (averaging out runs)
            sd_targ = float(M.mean(axis=0).std(ddof=1))
            sd_tot = float(np.sqrt(np.nan_to_num(sd_train)**2 + sd_targ**2))
            summary[group][met] = {"mean": float(M.mean()), "sd_train": sd_train,
                                   "sd_target": sd_targ, "sd_total": sd_tot,
                                   "n_train": len(runs), "n_target": len(TARGET_SEEDS)}
            print(f"  {group:<12}{met:<20}{M.mean():>9.4f}{sd_train:>12.4f}"
                  f"{sd_targ:>12.4f}{sd_tot:>12.4f}")
        print()

    json.dump({"per_run": res, "summary": summary},
              open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
