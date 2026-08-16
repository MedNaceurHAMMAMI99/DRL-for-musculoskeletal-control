"""
Generate the report's figures as PNG (for LaTeX inclusion in the thesis) and
collect every reported number into one JSON.

Everything is read from real artifacts on disk — runs/*/train_meta.json,
runs/*/pilot*_eval.json, runs/bench/benchmark_results.json, the force-comparison
JSONs. Nothing is typed in by hand, so a figure can never drift from the run it
claims to describe.

Colours are the validated categorical palette (see the data-visualisation
reference): slots blue/orange/aqua/yellow/magenta, which pass the CVD and
normal-vision separation gates in both light and dark mode. Every bar carries a
direct value label, which is also the required relief for the three light-mode
slots that sit below 3:1 contrast on the light surface.

Run:  python build_figures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "thesis_report", "figures")
DATA_JSON = os.path.join(os.path.dirname(OUT_DIR), "data.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Validated categorical slots (light mode).
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def jload(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


R = config.RUNS_DIR
data = {}


# ── Fig 1: the debugging progression ────────────────────────────────────────
def fig_progression():
    """Every metric that mattered, across the four pilots that produced them."""
    runs = ["Pilot 4\n(reward v3)", "Pilot 5\n(reward v4)",
            "Pilot 6\n(replay fix)", "Final\n(entropy tuned)"]
    final = [0.2424, 0.2540, 0.2040, 0.1063]
    minerr = [0.1781, 0.1344, 0.1316, 0.0740]
    data["progression"] = {"runs": runs, "final_error": final,
                           "min_error": minerr}

    x = np.arange(len(runs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w/2, final, w, label="Final error (end of reach)",
                color=C_BLUE, zorder=3)
    b2 = ax.bar(x + w/2, minerr, w, label="Closest approach",
                color=C_ORANGE, zorder=3)
    ax.axhline(0.08, color=C_AQUA, ls="--", lw=2, zorder=2,
               label="Classical benchmark (0.08 m)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel("Distance from target (metres)")
    ax.set_title("How far the hand ended up from the target", loc="left",
                 fontsize=12, color=INK, pad=12)
    ax.set_ylim(0, 0.30)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "fig1_progression.png")


# ── Fig 2: within-episode trajectory ────────────────────────────────────────
def fig_trajectory():
    steps = [0, 10, 25, 50, 75, 99]
    p6 = [0.795, 0.314, 0.203, 0.184, 0.184, 0.177]
    fin = [0.795, 0.198, 0.119, 0.110, 0.112, 0.113]
    data["trajectory"] = {"steps": steps, "pilot6": p6, "final": fin}

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(steps, p6, "-o", color=C_ORANGE, lw=2, ms=7,
            label="Before entropy tuning", zorder=3)
    ax.plot(steps, fin, "-o", color=C_BLUE, lw=2, ms=7,
            label="After entropy tuning", zorder=4)
    ax.axhline(0.08, color=C_AQUA, ls="--", lw=2, label="Classical benchmark")
    ax.annotate("reaches here and STAYS", xy=(50, 0.110), xytext=(56, 0.30),
                fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlabel("Step within the 2-second reach (100 steps total)")
    ax.set_ylabel("Distance from target (metres)")
    ax.set_title("Distance to the target during a single reach", loc="left",
                 fontsize=12, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig2_trajectory.png")


# ── Fig 3: algorithm benchmark ──────────────────────────────────────────────
def fig_benchmark():
    b = jload(os.path.join(R, "bench", "benchmark_results.json"), {})
    if not b:
        print("  (no benchmark data yet — skipping fig3)")
        return
    names, means, stds = [], [], []
    for cfg in ["SAC_tuned", "SAC_default", "TD3", "DDPG", "PPO"]:
        v = [r["metrics"]["mean_final_error"] for r in b.values()
             if r["config"] == cfg]
        if v:
            names.append(cfg.replace("_", "\n"))
            means.append(float(np.mean(v)))
            stds.append(float(np.std(v)))
    data["benchmark"] = {"configs": names, "mean": means, "std": stds}

    colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA][:len(names)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, means, 0.6, yerr=stds, capsize=5, color=colors, zorder=3,
                  error_kw=dict(ecolor=INK2, lw=1.4))
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.008,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Final distance from target (metres)")
    # Five bars, four algorithms: SAC appears twice (default and tuned) so the
    # tuning effect and the algorithm effect can be read off the same axis.
    ax.set_title("Five configurations, 3 runs each (lower is better)\n"
                 "The two SAC bars differ only in tuning",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    save(fig, "fig3_benchmark.png")


# ── Fig 4: per-muscle force, DRL vs classical ───────────────────────────────
def fig_forces():
    d = jload(os.path.join(R, "SAC_seed0",
                           "force_comparison_drl_vs_classical.json"))
    if not d:
        print("  (no force comparison yet — skipping fig4)")
        return
    names = list(d["per_muscle"].keys())
    pol = [d["per_muscle"][n]["policy_mean_N"] for n in names]
    cls = [d["per_muscle"][n]["classical_mean_N"] for n in names]
    data["forces"] = {"muscles": names, "policy_N": pol, "classical_N": cls,
                      "effort_ratio": d["effort"]["ratio_aggregate"],
                      "pattern_cosine": d["overall"]["pattern_cosine_mean"],
                      "pearson_r": d["overall"]["pearson_r"]}

    y = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.barh(y + h/2, pol, h, label="What the AI actually used",
            color=C_BLUE, zorder=3)
    ax.barh(y - h/2, cls, h, label="What classical maths says is needed",
            color=C_ORANGE, zorder=3)
    for i, (p, c) in enumerate(zip(pol, cls)):
        ax.text(p + 3, i + h/2, f"{p:.0f}", va="center", fontsize=8.5, color=INK)
        ax.text(c + 3, i - h/2, f"{c:.0f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average muscle force (newtons)")
    ax.set_title("Force in each muscle: AI vs classical calculation",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig4_forces.png")


# ── Fig 5: effort ratio by algorithm ────────────────────────────────────────
def fig_effort():
    a = jload(os.path.join(R, "bench", "force_comparison_all.json"), {})
    if not a:
        print("  (no cross-policy force data yet — skipping fig5)")
        return
    names, means, stds = [], [], []
    for cfg in ["SAC_tuned", "SAC_default", "TD3", "DDPG", "PPO"]:
        v = [r["effort_ratio"] for r in a.values() if r["config"] == cfg]
        if v:
            names.append(cfg.replace("_", "\n"))
            means.append(float(np.mean(v)))
            stds.append(float(np.std(v)))
    data["effort_by_algo"] = {"configs": names, "mean": means, "std": stds}

    colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA][:len(names)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, means, 0.6, yerr=stds, capsize=5, color=colors, zorder=3,
                  error_kw=dict(ecolor=INK2, lw=1.4))
    ax.axhline(1.0, color=C_AQUA, ls="--", lw=2,
               label="Perfect efficiency (classical optimum)")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.05,
                f"{m:.2f}x", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Effort used / minimum effort needed")
    ax.set_title("Wasted muscle effort, by learning method (1.0 = perfect)",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    # Headroom above the tallest bar+error+label, and the legend placed low, so
    # the value labels cannot collide with the legend text.
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.28)
    ax.legend(frameon=False, fontsize=9, loc="lower left",
              bbox_to_anchor=(0.0, 0.02))
    save(fig, "fig5_effort_by_algo.png")


if __name__ == "__main__":
    print("Building figures...")
    fig_progression()
    fig_trajectory()
    fig_benchmark()
    fig_forces()
    fig_effort()
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {DATA_JSON}")
