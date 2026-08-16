"""
Thesis Pipeline — Entry Point
==============================

Research question
-----------------
Can a reinforcement learning agent replace the computationally expensive
Newton-Raphson muscle equilibrium solver for real-time rehabilitation
and motor-control applications?

Argument structure
------------------
1. BIOMECHANICS
   The Hill muscle model is nonlinear — force depends on fibre length,
   activation, and velocity in a way that has no closed-form inverse.
   Producing the correct muscle force for a target movement therefore
   requires solving a system of implicit equations at every timestep.
   For 9 muscles at 100 Hz this means ~900 Newton-Raphson solves/second.

2. COMPUTATIONAL COST
   Each N-R solve takes 5–40 iterations.  Real-time control on embedded
   rehabilitation hardware is therefore expensive.

3. RL PROPOSAL
   Train a Soft Actor-Critic agent on the MuJoCo arm model.  At inference
   time the agent replaces the solver: given only the observation vector it
   outputs muscle activations directly — one forward pass through a small
   MLP, no iterative solving.

4. EVALUATION
   Compare task performance (success rate, reaching error) and inference
   latency between the two approaches over multiple seeds.

5. STATISTICS
   Non-parametric Wilcoxon tests + BCa bootstrap CIs confirm that the
   observed performance differences are (or are not) statistically significant.

Usage
-----
    python main.py           # full run; trains if no model found
    python main.py --no-train # skip training, evaluate existing model only
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np

# Make sub-packages importable when running `python main.py` from this folder
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

import config
os.makedirs(config.FIGURES_DIR, exist_ok=True)
os.makedirs(config.RUNS_DIR,    exist_ok=True)

from biomechanics.solver   import workspace_sweep
from analysis              import benchmark, stats, figures


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true",
                    help="Skip training; evaluate existing model only.")
args = parser.parse_args()


# ─── Stage 1 — Biomechanics ───────────────────────────────────────────────────
section("Stage 1 — Hill Muscle Model & Force Workspace")

L_MT_range  = np.linspace(0.33, 0.43, 60)
activations = [0.2, 0.5, 1.0]
workspace   = workspace_sweep(L_MT_range, activations, config.BICEPS_LONGUS)

figures.hill_curves(config.BICEPS_LONGUS)
figures.force_workspace(workspace, L_MT_range)

mean_iters = np.mean([workspace[a]["iters"].mean() for a in activations])
print(f"  Mean N-R iterations across workspace: {mean_iters:.1f}")


# ─── Stage 2 — Benchmark: solver cost ─────────────────────────────────────────
section("Stage 2 — Physics Solver Benchmark")

solver_stats = benchmark.benchmark_solver(n_queries=1000, params=config.BICEPS_LONGUS)
print(f"  N-R solver: {solver_stats['mean_us']:.1f} ± {solver_stats['std_us']:.1f} µs/step "
      f"(9 muscles, P95 = {solver_stats['p95_us']:.1f} µs)")


# ─── Stage 3 — RL Training ────────────────────────────────────────────────────
section("Stage 3 — SAC Training")
warnings.filterwarnings("ignore")

model_path = os.path.join(config.RUNS_DIR, "model.zip")
cb         = None

if args.no_train or os.path.exists(model_path):
    if not os.path.exists(model_path):
        print("  ERROR: --no-train specified but no model found.")
        print(f"  Expected: {model_path}")
        sys.exit(1)
    print(f"  Trained model found — loading from {config.RUNS_DIR}")
    from rl.evaluate import load_model
    model, envs = load_model(config.RUNS_DIR)
else:
    print("  No trained model found — starting SAC training …")
    from rl.train import train
    model, envs, cb = train(out_dir=config.RUNS_DIR, seed=config.SEED)
    if cb and cb.timesteps:
        figures.training_curves(
            cb.timesteps, cb.actor_losses, cb.critic_losses, cb.ent_coefs,
            actual_steps=cb.timesteps[-1],
        )


# ─── Stage 4 — RL Benchmark & Evaluation ──────────────────────────────────────
section("Stage 4 — RL Policy Evaluation & Benchmark")

rl_stats = benchmark.benchmark_rl(model, envs, n_queries=1000)
benchmark.print_comparison(solver_stats, rl_stats)
figures.benchmark_comparison(solver_stats, rl_stats)

from rl.evaluate import evaluate
eval_metrics = evaluate(model, envs, n_episodes=20, seed=config.SEED)

print(f"  Success rate      : {eval_metrics['success_rate'] * 100:.1f}%")
print(f"  Mean final error  : {eval_metrics['mean_final_error'] * 100:.2f} cm")
print(f"  Inference latency : {eval_metrics['mean_inference_us']:.1f} µs / step")


# ─── Stage 5 — Statistical Comparison (real multi-seed data) ──────────────────
section("Stage 5 — Statistical Analysis (multi-algorithm, multi-seed)")

# The multi-algorithm benchmark is produced by experiment_runner.py, which
# trains every (algorithm, seed) for real and writes runs/results.json. This
# stage reads those REAL per-seed success rates — it never simulates them.
# (The previous version injected np.random.normal() placeholders here; that has
# been removed. If you have not run the grid yet, do so first — see RUNBOOK.md.)
if os.path.exists(config.RESULTS_JSON):
    with open(config.RESULTS_JSON) as f:
        _res = json.load(f)
    algo_results = {a: np.array([v["success_rate"] for v in seeds_d.values()])
                    for a, seeds_d in _res.get("per_seed", {}).items()
                    if len(seeds_d) >= 2}

    if algo_results:
        summaries, comparisons = stats.compare_algorithms(algo_results, rng_seed=config.SEED)
        figures.statistical_comparison(summaries, comparisons)

        print(f"\n  {'Algorithm':>10}  {'Mean':>7}  {'95% BCa CI':>22}")
        print("  " + "-" * 44)
        for name, s in summaries.items():
            print(f"  {name:>10}  {s['mean']:.3f}  [{s['ci_lo']:.3f},  {s['ci_hi']:.3f}]")

        print(f"\n  {'Comparison':>14}  {'W':>5}  {'p-value':>9}  {'r_b':>6}  {'Signif.':>8}")
        print("  " + "-" * 48)
        for c in comparisons:
            print(f"  {c['a']:>6} vs {c['b']:<3}  {c['W']:>5.0f}  {c['p']:>9.4f}  "
                  f"{c['rb']:>+6.2f}  {str(c['significant']):>8}")
    else:
        print("  results.json has fewer than 2 seeds per algorithm — "
              "run more seeds via experiment_runner.py before statistical testing.")
else:
    print("  No runs/results.json found. The multi-algorithm benchmark is NOT")
    print("  fabricated here — run it for real:")
    print("      python experiment_runner.py --stage all")
    print("  (or `--smoke` for a quick end-to-end check). See RUNBOOK.md.")


# ─── Done ─────────────────────────────────────────────────────────────────────
section("ALL STAGES COMPLETE")
print(f"  Figures saved to: {config.FIGURES_DIR}")
print()
