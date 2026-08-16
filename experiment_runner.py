"""
Experiment runner — the single entry point that produces every REAL number the
thesis and paper report. Nothing here is simulated or hand-authored.

Pipeline
--------
  1. TRAIN    each (algorithm, seed) in the grid to an equal step budget.
  2. EVALUATE each trained policy: per-seed success rate, energy, error, reward,
     plus activation traces for the biological-coordination analyses.
  3. BASELINES evaluate PD and Random (no training).
  4. ANALYSE  aggregate per-seed metrics -> Wilcoxon + BCa statistics; run the
     muscle-synergy (NMF/VAF) and co-contraction-index analyses.
  5. WRITE    everything to runs/results.json, which the LaTeX tables/figures and
     the paper read from. The already-measured, training-independent artifacts
     (N-R vs policy latency, force fidelity, footprint) are copied in from
     appendix_run/experiments_results.json so results.json is the single source.

Usage
-----
  python experiment_runner.py --smoke              # tiny end-to-end proof (minutes)
  python experiment_runner.py --stage all          # full grid (hours-days)
  python experiment_runner.py --algos SAC TD3 --seeds 0 1 2 --steps 1000000
  python experiment_runner.py --stage analyse      # re-aggregate existing runs

Resumable: a run whose model.zip already exists is not retrained.
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import config


def _run_dir(algo, seed):
    return os.path.join(config.RUNS_DIR, f"{algo}_seed{seed}")


def stage_train(algos, seeds, steps):
    from rl.train import train
    for algo in algos:
        for seed in seeds:
            rd = _run_dir(algo, seed)
            if os.path.exists(os.path.join(rd, "model.zip")):
                print(f"  [skip] {algo} seed {seed} already trained")
                continue
            train(algo=algo, out_dir=rd, seed=seed, total_steps=steps)


def stage_evaluate(algos, seeds, eval_episodes, synergy_episodes):
    """Return per-seed metrics and per-algorithm activation traces (best seed)."""
    from rl.evaluate import load_model, evaluate

    per_seed = {a: {} for a in algos}
    activations = {}
    for algo in algos:
        best_sr, best_traces = -1.0, None
        for seed in seeds:
            rd = _run_dir(algo, seed)
            if not os.path.exists(os.path.join(rd, "model.zip")):
                print(f"  [miss] {algo} seed {seed} not trained — skipping")
                continue
            model, envs = load_model(rd, algo=algo)
            m = evaluate(model, envs, n_episodes=eval_episodes, seed=seed)
            per_seed[algo][seed] = {k: m[k] for k in
                                    ("success_rate", "mean_energy",
                                     "mean_final_error", "mean_reward")}
            # Log activation traces from the best seed for synergy/CCI.
            if m["success_rate"] > best_sr:
                ma = evaluate(model, envs, n_episodes=synergy_episodes,
                              seed=seed, log_activations=True)
                best_sr, best_traces = m["success_rate"], ma.get("activations")
        if best_traces:
            activations[algo] = best_traces
    return per_seed, activations


def stage_baselines(eval_episodes):
    from rl.baselines import evaluate_baseline
    out = {}
    for kind in config.BASELINES:
        m = evaluate_baseline(kind, n_episodes=eval_episodes, seed=0)
        out[kind] = {"success_rate": m["success_rate"],
                     "mean_reward": m["mean_reward"],
                     "mean_energy": m["mean_energy"],
                     "mean_final_error": m["mean_final_error"],
                     "success_per_ep": m["success_per_ep"]}
        print(f"  [baseline {kind}] success={m['success_rate']*100:.1f}%")
    return out


def stage_analyse(per_seed, activations, baselines):
    from analysis.stats import compare_algorithms
    from analysis.synergies import analyse_synergies
    from analysis.cci import analyse_cci

    # Per-algorithm per-seed success-rate arrays -> statistics.
    score_arrays = {}
    for algo, seeds_d in per_seed.items():
        arr = [v["success_rate"] for v in seeds_d.values()]
        if len(arr) >= 2:
            score_arrays[algo] = np.array(arr)

    stats_block = {"task_performance": {}, "comparisons": []}
    if len(score_arrays) >= 1:
        summaries, comparisons = compare_algorithms(score_arrays)
        for name, s in summaries.items():
            n = len(score_arrays[name])
            stats_block["task_performance"][name] = {
                "success_mean": s["mean"], "ci_lo": s["ci_lo"],
                "ci_hi": s["ci_hi"], "n_seeds": n,
                "success_std": float(np.std(score_arrays[name])),
            }
        stats_block["comparisons"] = comparisons

    synergy_block, cci_block = {}, {}
    for algo, traces in activations.items():
        synergy_block[algo] = analyse_synergies(traces)
        cci_block[algo]     = analyse_cci(traces)

    return {"statistics": stats_block, "per_seed": per_seed,
            "synergies": synergy_block, "cci": cci_block,
            "baselines": baselines}


def _load_measured_artifacts():
    """Copy in the already-measured, training-independent real results."""
    here = os.path.dirname(__file__)
    local = os.path.join(here, "experiments_results.json")   # portable package copy
    path = local if os.path.exists(local) else os.path.normpath(
        os.path.join(here, "..", "appendix_run", "experiments_results.json"))
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {"latency": d.get("timing"), "force_fidelity": d.get("force_comparison"),
            "footprint": d.get("policy_characterization")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=config.ALGORITHMS)
    p.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    p.add_argument("--steps", type=int, default=config.TRAIN_STEPS)
    p.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES)
    p.add_argument("--synergy-episodes", type=int, default=config.SYNERGY_EPISODES)
    p.add_argument("--stage",
                   choices=["train", "evaluate", "analyse", "report", "all"],
                   default="all")
    p.add_argument("--robustness", action="store_true",
                   help="Also run the force-impulse robustness sweep per algorithm.")
    p.add_argument("--ablation", action="store_true",
                   help="Also run the reward-term ablation (SAC). Retrains per term.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end proof: 2 algos, 2 seeds, few steps.")
    args = p.parse_args()

    if args.stage == "report":
        from analysis import report
        report.main()
        return

    if args.smoke:
        args.algos = ["SAC", "TD3"]
        args.seeds = [0, 1]
        args.steps = 2000
        args.eval_episodes = 3
        args.synergy_episodes = 2

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    print(f"Grid: algos={args.algos} seeds={args.seeds} steps={args.steps:,}")

    if args.stage in ("train", "all"):
        print("\n=== TRAIN ===")
        stage_train(args.algos, args.seeds, args.steps)

    per_seed, activations, baselines = {}, {}, {}
    if args.stage in ("evaluate", "analyse", "all"):
        print("\n=== EVALUATE ===")
        per_seed, activations = stage_evaluate(
            args.algos, args.seeds, args.eval_episodes, args.synergy_episodes)
        print("\n=== BASELINES ===")
        baselines = stage_baselines(args.eval_episodes)

    print("\n=== ANALYSE ===")
    results = stage_analyse(per_seed, activations, baselines)
    results["measured_artifacts"] = _load_measured_artifacts()

    if args.robustness:
        print("\n=== ROBUSTNESS ===")
        from analysis.generalization import evaluate_robustness
        rob = {}
        for algo in args.algos:
            # best seed = first available trained seed
            seed = next((s for s in args.seeds if os.path.exists(
                os.path.join(_run_dir(algo, s), "model.zip"))), None)
            if seed is not None:
                rob[algo] = evaluate_robustness(_run_dir(algo, seed), algo,
                                                n_episodes=args.eval_episodes, seed=seed)
        results["robustness"] = rob

    if args.ablation:
        print("\n=== ABLATION (SAC) ===")
        from analysis.ablation import run_ablation
        results["ablation"] = {"SAC": run_ablation(
            "SAC", seed=args.seeds[0], total_steps=args.steps)}
    results["meta"] = {
        "algos": args.algos, "seeds": args.seeds, "train_steps": args.steps,
        "eval_episodes": args.eval_episodes, "smoke": args.smoke,
        "note": "All numbers here are produced by real training/evaluation. "
                "Do not hand-edit.",
    }

    with open(config.RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {config.RESULTS_JSON}")

    # Render LaTeX tables/figures from the real results.
    if results.get("per_seed"):
        print("\n=== REPORT ===")
        from analysis import report
        report.main()


if __name__ == "__main__":
    main()
