"""
Unattended execution of every outstanding experiment, in priority order.

Runs without supervision. Each stage is independently resumable, failures are
caught so one broken stage cannot abort the rest, and results are written to
disk as soon as they exist. Safe to kill and restart at any point: completed
stages are skipped.

Order (highest reviewer-severity first, not shortest first):

  1. CONVENTIONAL   ~40 min  A classical controller solving the same task, at the
                             full 50-episode protocol. Supplies the achievable
                             effort floor, which the 1.71x figure is currently
                             measured against the wrong reference for.
  2. REPLICATION    ~6.5 h   Four further seeds of the confirmed configuration,
                             plus the force comparison on each. Converts the
                             thesis's central result from n=1 to n=5. This is the
                             single most serious reviewer objection.
  3. ABLATION       ~4 h     Two-sided one-at-a-time test isolating whether the
                             target entropy or the target-network rate is
                             responsible for the precision improvement.
  4. EFFORT SWEEP   ~2.5 h   Resumes from its 150k checkpoint. Scientifically
                             interesting but not a reviewer objection, so last.

Thermal safety: strictly sequential, 8 of 16 logical processors, never parallel.
The machine hard-locked under sustained all-core load on 2026-08-14.

Run:  python run_pending.py
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(8)

HERE   = os.path.dirname(os.path.abspath(__file__))
STATE  = os.path.join(config.RUNS_DIR, "pending_state.json")
REPORT = os.path.join(config.RUNS_DIR, "pending_summary.json")

TUNED_SEEDS = [1, 2, 3, 4]      # seed 0 is the existing confirmed run
STEPS_FULL  = 300_000
STEPS_ABL   = 100_000
ABL_SEEDS   = [0, 1, 2]


def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"done": [], "results": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE)


def tuned_params():
    db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
    p = dict(optuna.load_study(study_name="SAC_arm_reach",
                               storage=f"sqlite:///{db}").best_trial.params)
    p["gradient_steps"] = 2 * config.N_ENVS
    return p


def force_ratio(run_dir: str, algo: str = "SAC", episodes: int = 20):
    """Effort ratio + agreement for one policy, via the single implementation."""
    import subprocess
    cmd = [sys.executable, os.path.join(HERE, "force_comparison.py"),
           "--run", run_dir, "--algo", algo, "--episodes", str(episodes)]
    subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    p = os.path.join(run_dir, "force_comparison_drl_vs_classical.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {"effort_ratio": d["effort"]["ratio_aggregate"],
            "pearson_r": d["overall"]["pearson_r"],
            "cosine": d["overall"]["pattern_cosine_mean"],
            "cosine_null": d["overall"]["pattern_cosine_null_mean"]}


# ── Stage 1 ────────────────────────────────────────────────────────────────
def stage_conventional(st):
    import subprocess
    log = os.path.join(HERE, "conventional.log")
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "run_conventional.py"),
                        "--episodes", "50"], cwd=HERE, stdout=f,
                       stderr=subprocess.STDOUT)
    p = os.path.join(config.RUNS_DIR, "conventional_baseline.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        st["results"]["conventional"] = {
            "PD": {k: d["PD"][k] for k in
                   ("mean_final_error", "mean_min_error", "effort_ratio",
                    "mean_energy", "reach_5cm", "reach_10cm", "pearson_r")},
            "gains": d.get("gains")}
        return True
    return False


# ── Stage 2 ────────────────────────────────────────────────────────────────
def stage_replication(st):
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    params = tuned_params()
    out_root = os.path.join(config.RUNS_DIR, "replication")
    os.makedirs(out_root, exist_ok=True)
    res = st["results"].setdefault("replication", {})

    for seed in TUNED_SEEDS:
        key = f"seed{seed}"
        if key in res:
            continue
        out = os.path.join(out_root, f"SAC_seed{seed}")
        print(f"\n--- replication {key} ---", flush=True)
        train("SAC", out_dir=out, seed=seed, total_steps=STEPS_FULL,
              hyperparams=params)
        model, envs = load_model(out, "SAC")
        m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
        envs.close()
        entry = {k: float(m[k]) for k in
                 ("mean_final_error", "mean_min_error", "success_rate",
                  "reach_5cm", "reach_10cm", "touch_2cm", "mean_energy",
                  "blow_up_rate")}
        fr = force_ratio(out)
        if fr:
            entry.update(fr)
        res[key] = entry
        save_state(st)
        print(f"  {key}: err={entry['mean_final_error']:.4f} "
              f"effort={entry.get('effort_ratio', float('nan')):.2f}", flush=True)
    return True


# ── Stage 3 ────────────────────────────────────────────────────────────────
def stage_ablation(st):
    """
    Two-sided one-at-a-time test of the target-entropy attribution.

    arm A: library defaults, ONLY target_entropy set to the tuned value.
           If this recovers most of the improvement, entropy is the mechanism.
    arm B: tuned configuration, ONLY target_entropy reverted to default.
           If this ALSO performs well, entropy is NOT the mechanism and the
           credit belongs to tau or the learning rate.
    """
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    tuned = tuned_params()
    te = tuned["target_entropy"]
    arms = {
        "A_entropy_only": {"gradient_steps": 2 * config.N_ENVS,
                           "target_entropy": te},
        "B_entropy_reverted": {**tuned, "target_entropy": "auto"},
    }
    out_root = os.path.join(config.RUNS_DIR, "ablation")
    os.makedirs(out_root, exist_ok=True)
    res = st["results"].setdefault("ablation", {})

    for arm, hp in arms.items():
        for seed in ABL_SEEDS:
            key = f"{arm}_seed{seed}"
            if key in res:
                continue
            out = os.path.join(out_root, key)
            print(f"\n--- ablation {key} ---", flush=True)
            train("SAC", out_dir=out, seed=seed, total_steps=STEPS_ABL,
                  hyperparams=hp)
            model, envs = load_model(out, "SAC")
            m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
            envs.close()
            res[key] = {"arm": arm, "seed": seed,
                        "mean_final_error": float(m["mean_final_error"]),
                        "mean_min_error": float(m["mean_min_error"])}
            save_state(st)
            print(f"  {key}: err={m['mean_final_error']:.4f}", flush=True)
    return True


# ── Stage 4 ────────────────────────────────────────────────────────────────
def stage_sweep(st):
    import subprocess
    lock = os.path.join(config.RUNS_DIR, "sweep.lock")
    if os.path.exists(lock):
        os.remove(lock)
    log = os.path.join(HERE, "effort_sweep.log")
    with open(log, "a") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "run_effort_sweep.py")],
                       cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    p = os.path.join(config.RUNS_DIR, "effort_sweep", "effort_sweep_results.json")
    if os.path.exists(p):
        with open(p) as f:
            st["results"]["effort_sweep"] = json.load(f)
        return True
    return False


STAGES = [("conventional", stage_conventional),
          ("replication",  stage_replication),
          ("ablation",     stage_ablation),
          ("effort_sweep", stage_sweep)]


def main():
    st = load_state()
    print("=" * 74)
    print("UNATTENDED RUN — pending experiments in reviewer-severity order")
    print("=" * 74)
    print(f"  already complete: {st['done'] or 'none'}\n", flush=True)

    for name, fn in STAGES:
        if name in st["done"]:
            print(f"[skip] {name} (already complete)", flush=True)
            continue
        print(f"\n{'='*74}\n[START] {name}  ({time.strftime('%H:%M:%S')})\n{'='*74}",
              flush=True)
        t0 = time.perf_counter()
        try:
            ok = fn(st)
            mins = (time.perf_counter() - t0) / 60
            if ok:
                st["done"].append(name)
                print(f"[DONE] {name} in {mins:.0f} min", flush=True)
            else:
                print(f"[INCOMPLETE] {name} after {mins:.0f} min "
                      f"— left for a later run", flush=True)
        except Exception:
            print(f"[FAILED] {name}:\n{traceback.format_exc()}", flush=True)
        save_state(st)

    with open(REPORT, "w") as f:
        json.dump(st["results"], f, indent=2)
    print("\n" + "=" * 74)
    print(f"ALL STAGES ATTEMPTED. Completed: {st['done']}")
    print(f"Summary written to {REPORT}")
    print("=" * 74)


if __name__ == "__main__":
    main()
