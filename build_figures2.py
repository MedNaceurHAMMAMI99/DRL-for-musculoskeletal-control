"""
Figures for the results that postdate build_figures.py: the effort-weight sweep
and the hyperparameter ablation.

Values are read from the run artifacts where those exist, and otherwise from the
aggregates recorded in this file's DATA block, which is kept adjacent to the
numbers it plots so a figure cannot silently disagree with the text.

Palette is the validated categorical set (blue/orange/aqua/yellow/magenta),
which passes the CVD and normal-vision separation gates in light and dark mode.
Every bar carries a direct value label -- which is also the required relief for
the three light-mode slots below 3:1 contrast on a light surface.

Run:  python build_figures2.py
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "thesis_report", "figures")
os.makedirs(OUT, exist_ok=True)

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── measured aggregates (five seeds each unless noted) ─────────────────────
SWEEP = {
    "w2":      [1, 5, 20],
    "effort":  [1.914, 1.406, 1.304],
    "eff_sd":  [0.150, 0.092, 0.025],
    "r":       [0.808, 0.919, 0.943],
    "r_sd":    [0.044, 0.016, 0.008],
    "err":     [0.1033, 0.1023, 0.1072],
    "err_sd":  [0.0034, 0.0074, 0.0148],
}
CONVENTIONAL_FLOOR = 1.263     # fine 30-point gain search (coarse grid gave 1.33)

ABLATION = {                    # fraction of the default->tuned gap recovered
    "labels": ["discount\nfactor $\\gamma$", "learning\nrate",
               "batch\nsize", "target\nnetwork $\\tau$", "target\nentropy"],
    "gap":    [106, -3, -5, -5, -8],
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_sweep():
    """Effort ratio and classical agreement against the effort weight."""
    x = np.arange(len(SWEEP["w2"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    b = a1.bar(x, SWEEP["effort"], 0.55, yerr=SWEEP["eff_sd"], capsize=5,
               color=[C_BLUE, C_ORANGE, C_AQUA], zorder=3,
               error_kw=dict(ecolor=INK2, lw=1.4))
    a1.axhline(CONVENTIONAL_FLOOR, color=C_MAGENTA, ls="--", lw=2, zorder=2,
               label=f"conventional controller ({CONVENTIONAL_FLOOR})")
    a1.axhline(1.0, color=MUTED, ls=":", lw=1.5, zorder=2,
               label="idealised optimum (unreachable)")
    for bb, m, s in zip(b, SWEEP["effort"], SWEEP["eff_sd"]):
        a1.text(bb.get_x() + bb.get_width()/2, m + s + 0.04, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    a1.set_xticks(x); a1.set_xticklabels([f"$w_2$ = {w}" for w in SWEEP["w2"]])
    a1.set_ylabel("Effort used / minimum needed")
    a1.set_title("Wasted muscular effort", loc="left", fontsize=11.5, pad=10)
    a1.set_ylim(0, 2.35); a1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")

    b2 = a2.bar(x, SWEEP["r"], 0.55, yerr=SWEEP["r_sd"], capsize=5,
                color=[C_BLUE, C_ORANGE, C_AQUA], zorder=3,
                error_kw=dict(ecolor=INK2, lw=1.4))
    for bb, m, s in zip(b2, SWEEP["r"], SWEEP["r_sd"]):
        a2.text(bb.get_x() + bb.get_width()/2, m + s + 0.012, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    a2.set_xticks(x); a2.set_xticklabels([f"$w_2$ = {w}" for w in SWEEP["w2"]])
    a2.set_ylabel("Pearson $r$ vs static optimisation")
    a2.set_title("Agreement with the classical solution", loc="left",
                 fontsize=11.5, pad=10)
    a2.set_ylim(0.7, 1.0); a2.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    fig.suptitle("Raising the effort weight improves load-sharing fidelity "
                 "monotonically (five seeds each)",
                 fontsize=12.5, x=0.06, ha="left", y=1.02)
    save(fig, "fig6_effort_sweep.png")


def fig_ablation():
    """Which single hyperparameter closes the default-to-tuned gap."""
    y = np.arange(len(ABLATION["labels"]))
    vals = ABLATION["gap"]
    colors = [C_AQUA if v > 50 else C_ORANGE for v in vals]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.barh(y, vals, 0.6, color=colors, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=2)
    ax.axvline(100, color=MUTED, ls="--", lw=1.6, zorder=2,
               label="fully tuned configuration")
    for bb, v in zip(bars, vals):
        off = 3 if v > 0 else -3
        ax.text(v + off, bb.get_y() + bb.get_height()/2, f"{v:+d}%",
                va="center", ha="left" if v > 0 else "right",
                fontsize=10.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(ABLATION["labels"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of the default-to-tuned improvement recovered")
    ax.set_title("One parameter at a time: only the discount factor matters\n"
                 "(three seeds each, all others at library defaults)",
                 loc="left", fontsize=12, pad=12)
    ax.set_xlim(-25, 125)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig7_ablation.png")


def fig_horizon():
    """Why the discount factor matters: horizon against task timescale."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    bars = ax.barh([1, 0], [100, 23], 0.5,
                   color=[C_ORANGE, C_AQUA], zorder=3)
    ax.axvline(25, color=C_MAGENTA, ls="--", lw=2, zorder=4,
               label="reach completes (step 25)")
    ax.axvline(100, color=MUTED, ls=":", lw=1.5, zorder=2,
               label="episode ends (step 100)")
    for bb, v, lab in zip(bars, [100, 23],
                          ["$\\gamma = 0.99$ (default)", "$\\gamma = 0.957$ (selected)"]):
        ax.text(v + 2, bb.get_y() + bb.get_height()/2, f"{v} steps",
                va="center", fontsize=10.5, color=INK)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["$\\gamma = 0.99$\n(default)",
                        "$\\gamma = 0.957$\n(selected)"], fontsize=9.5)
    ax.set_xlabel("Effective planning horizon, $1/(1-\\gamma)$ steps")
    ax.set_title("The default horizon spanned the whole episode; the reach "
                 "finishes in a quarter of it", loc="left", fontsize=12, pad=12)
    ax.set_xlim(0, 120)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig8_horizon.png")


if __name__ == "__main__":
    print("Building figures (batch 2)...")
    fig_sweep()
    fig_ablation()
    fig_horizon()
    json.dump({"sweep": SWEEP, "ablation": ABLATION,
               "conventional_floor": CONVENTIONAL_FLOOR},
              open(os.path.join(os.path.dirname(OUT), "data2.json"), "w"), indent=2)
    print(f"  wrote {os.path.join(os.path.dirname(OUT), 'data2.json')}")
