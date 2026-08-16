"""
Human-readable digest + integrity checks for a returned runs/results.json.

Use this AFTER copying back the `runs/` folder produced on the compute machine:

    python -m analysis.summarize            # reads runs/results.json
    python -m analysis.summarize path/to/results.json

It prints the run configuration, per-algorithm task performance, the pairwise
statistics, the synergy/CCI/baseline status, and the trustworthy measured
artifacts — and runs a set of INTEGRITY CHECKS that flag the failure modes seen in
this project's history (all-tied ties, a suspiciously-perfect identical ranking
across metrics, undertrained runs, the actor-vs-full-model footprint pitfall).
It does not modify anything; render the LaTeX with `experiment_runner.py --stage
report`.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config


def _load(path=None):
    path = path or config.RESULTS_JSON
    if not os.path.exists(path):
        sys.exit(f"No results file at {path}")
    with open(path) as f:
        return json.load(f), path


def summarize(path=None):
    res, path = _load(path)
    meta = res.get("meta", {})
    per_seed = res.get("per_seed", {})
    stats = res.get("statistics", {})
    print("=" * 70)
    print(f"RESULTS DIGEST  ({path})")
    print("=" * 70)
    print(f"algos={meta.get('algos')}  seeds={meta.get('seeds')}  "
          f"train_steps={meta.get('train_steps'):,}  eval_ep={meta.get('eval_episodes')}")

    # ── Task performance ──
    print("\n-- Task performance (success %, mean +/- std over seeds) --")
    succ = {}
    for a, s in stats.get("task_performance", {}).items():
        succ[a] = s["success_mean"]
        print(f"  {a:6s}  {s['success_mean']*100:6.1f}%  "
              f"(CI [{s['ci_lo']*100:.1f}, {s['ci_hi']*100:.1f}], n={s['n_seeds']})")

    # ── Statistics ──
    print("\n-- Pairwise Wilcoxon (success rate) --")
    n_sig = 0
    for c in stats.get("comparisons", []):
        sig = "SIG" if c["significant"] else "ns"
        n_sig += int(c["significant"])
        print(f"  {c['a']:5s} vs {c['b']:5s}  W={c['W']:.0f}  p={c['p']:.4f}  "
              f"rb={c['rb']:+.2f}  {sig}")

    # ── Synergy / CCI / baselines ──
    syn = res.get("synergies", {})
    if syn:
        print("\n-- Muscle synergies --")
        for a, s in syn.items():
            if "error" in s:
                print(f"  {a:6s}  {s['error']}")
            else:
                print(f"  {a:6s}  VAF={s['vaf_at_k']*100:.1f}%  "
                      f"sim={s['similarity_to_reference']:.2f}  "
                      f"n_samples={s['n_samples']}")
    bl = res.get("baselines", {})
    if bl:
        print("\n-- Baselines --")
        for k, v in bl.items():
            print(f"  {k:8s}  success={v['success_rate']*100:.1f}%")
    art = res.get("measured_artifacts", {})
    if art.get("latency"):
        t = art["latency"]
        print("\n-- Measured artifacts (trustworthy) --")
        print(f"  latency: N-R {t['nr_mean_us']:.0f}us vs policy "
              f"{t['policy_mean_us']:.0f}us  (speedup {t['speedup_mean']:.1f}x)")

    _integrity_checks(res, succ)


def _integrity_checks(res, succ):
    print("\n" + "=" * 70)
    print("INTEGRITY CHECKS")
    print("=" * 70)
    flags = []

    # 1) undertrained / all-zero
    if succ and all(v < 1e-9 for v in succ.values()):
        flags.append("[UNDERTRAINED] every algorithm at 0% success -> increase "
                     "TRAIN_STEPS/SEEDS; this run is not defensible as a benchmark.")
    # 2) all tied
    if succ and len(set(round(v, 6) for v in succ.values())) == 1:
        flags.append("[ALL-TIED] all algorithms share one success rate -> Wilcoxon "
                     "tests are degenerate (p=1.0 by construction).")
    # 3) suspiciously-perfect identical ranking across metrics
    metrics = {}
    per_seed = res.get("per_seed", {})
    for key in ("success_rate", "mean_reward", "mean_energy"):
        order = sorted(per_seed, key=lambda a: np.mean(
            [v[key] for v in per_seed[a].values()]) if per_seed[a] else 0)
        metrics[key] = tuple(order)
    if len(per_seed) >= 3 and len(set(metrics.values())) == 1:
        flags.append("[TOO-CLEAN-ORDERING] algorithms rank identically on EVERY "
                     "metric -> historically the loudest sign of authored (not "
                     "measured) numbers. Verify against raw logs; a real run "
                     "usually ties or reorders somewhere.")
    # 4) footprint actor-vs-full-model
    fp = res.get("measured_artifacts", {}).get("footprint", {})
    if fp.get("n_parameters", 0) > 100_000:
        flags.append("[FOOTPRINT] n_parameters counts the FULL SAC object (actor+2 "
                     "critics+2 targets). Only the actor (~78k) is deployed -- use "
                     "the actor-only figure in the paper.")
    # 5) synergy reportability
    syn = res.get("synergies", {})
    small = [a for a, s in syn.items()
             if isinstance(s, dict) and s.get("n_samples", 0) < 100]
    if small:
        flags.append(f"[SYNERGY] {len(small)} algo(s) have <100 activation samples "
                     "-> synergy/CCI not reportable (report.py suppresses these).")

    if flags:
        for f in flags:
            print("  ! " + f)
    else:
        print("  No integrity flags. Numbers look like a real, adequately-resourced run.")
    print()


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else None)
