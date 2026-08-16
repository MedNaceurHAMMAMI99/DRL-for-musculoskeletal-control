"""
Run the DRL-vs-static-optimisation force comparison across EVERY benchmark
policy, to answer: is the effort gap a SAC artifact or general to DRL here?

`force_comparison.py` measured the confirmed SAC policy against classical static
optimisation and found it uses 1.71x the minimum effort needed to produce the
same joint torques, with the excess concentrated in elbow antagonists. That was
one policy. This asks whether every algorithm does it.

Three outcomes are meaningful and distinguishable:

  * All configurations sit near the same ratio -> the effort gap is a property
    of the REWARD (which never asked for minimum effort), not of any algorithm.
    This is the hypothesis the reward design predicts.
  * Ratios differ systematically by algorithm -> exploration style determines
    load sharing, which would be a genuinely novel result.
  * SAC_tuned differs from SAC_default -> the entropy target governs not just
    precision but muscle coordination, tying this analysis to the main finding.

Each policy is measured by invoking force_comparison.py as a subprocess and
reading the JSON it writes into the run directory, so there is exactly ONE
implementation of the comparison and no chance of the two drifting apart.

Run:  python force_comparison_all.py [--episodes N]
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config

HERE      = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(config.RUNS_DIR, "bench")
OUT_JSON  = os.path.join(BENCH_DIR, "force_comparison_all.json")

# config name -> SB3 algorithm class name
ALGO_OF = {"SAC_default": "SAC", "SAC_tuned": "SAC",
           "TD3": "TD3", "DDPG": "DDPG", "PPO": "PPO"}
SEEDS = [0, 1, 2]


def run_one(run_dir: str, algo: str, episodes: int):
    """Invoke force_comparison.py on one policy; return its JSON, or None."""
    if not os.path.exists(os.path.join(run_dir, "model.zip")):
        return None
    cmd = [sys.executable, os.path.join(HERE, "force_comparison.py"),
           "--run", run_dir, "--algo", algo, "--episodes", str(episodes)]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    dest = os.path.join(run_dir, "force_comparison_drl_vs_classical.json")
    if r.returncode != 0 or not os.path.exists(dest):
        print(f"    FAILED ({r.returncode}): {r.stderr.strip().splitlines()[-1:]}")
        return None
    with open(dest) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    results = {}
    for name, algo in ALGO_OF.items():
        for s in SEEDS:
            run_dir = os.path.join(BENCH_DIR, f"{name}_seed{s}")
            key = f"{name}_seed{s}"
            print(f"  {key} ...", flush=True)
            d = run_one(run_dir, algo, args.episodes)
            if d:
                results[key] = {"config": name, "seed": s,
                                "effort_ratio": d["effort"]["ratio_aggregate"],
                                "pattern_cosine": d["overall"]["pattern_cosine_mean"],
                                "pearson_r": d["overall"]["pearson_r"],
                                "nrmse_pct": d["overall"]["nrmse_pct"],
                                "residual": d["validity"]["mean_residual_policy_Nm"],
                                "per_muscle": d["per_muscle"]}
                print(f"    effort {d['effort']['ratio_aggregate']:.2f}x  "
                      f"cosine {d['overall']['pattern_cosine_mean']:.3f}  "
                      f"r {d['overall']['pearson_r']:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"FORCE AGREEMENT BY ALGORITHM — {args.episodes} episodes each, "
          f"mean +- std over seeds")
    print("=" * 78)
    print(f"  {'config':<14}{'n':>2}{'effort ratio':>18}{'pattern cosine':>18}"
          f"{'pearson r':>14}")
    print("  " + "-" * 74)
    rows = []
    for name in ALGO_OF:
        vals = [v for v in results.values() if v["config"] == name]
        if not vals:
            continue
        er = np.array([v["effort_ratio"] for v in vals])
        pc = np.array([v["pattern_cosine"] for v in vals])
        pr = np.array([v["pearson_r"] for v in vals])
        rows.append((name, len(vals), er, pc, pr))
        print(f"  {name:<14}{len(vals):>2}"
              f"{er.mean():>11.2f}x+-{er.std():<5.2f}"
              f"{pc.mean():>12.3f}+-{pc.std():<5.3f}"
              f"{pr.mean():>9.3f}+-{pr.std():<5.3f}")

    if len(rows) >= 2:
        ers = {n: e.mean() for n, _, e, _, _ in rows}
        lo, hi = min(ers, key=ers.get), max(ers, key=ers.get)
        spread = ers[hi] / max(ers[lo], 1e-9)
        print(f"\n  Lowest effort ratio : {lo} ({ers[lo]:.2f}x)")
        print(f"  Highest effort ratio: {hi} ({ers[hi]:.2f}x)  — {spread:.2f}x spread")
        if spread < 1.4:
            print("\n  All configurations cluster: the effort gap is a property of the")
            print("  REWARD (which never asked for minimum effort), not of any one")
            print("  algorithm. No DRL variant here recovers classical load sharing.")
        else:
            print("\n  Configurations differ materially — exploration style appears to")
            print("  affect load sharing, not just endpoint accuracy.")
        if "SAC_tuned" in ers and "SAC_default" in ers:
            print(f"\n  Entropy-target effect within SAC: "
                  f"{ers['SAC_default']:.2f}x (default) -> "
                  f"{ers['SAC_tuned']:.2f}x (tuned)")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
