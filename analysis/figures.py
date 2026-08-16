"""
All figure generation for the thesis pipeline.

Five figures, in narrative order:
  01_hill_curves.png         — the nonlinear Hill model (WHY the problem is hard)
  02_force_workspace.png     — force field + solver cost (WHAT RL must learn)
  03_benchmark.png           — computation time: solver vs RL (THE TRADE-OFF)
  04_training_curves.png     — SAC convergence (HOW the RL agent learns)
  05_statistical_comparison  — significance testing (IS THE DIFFERENCE REAL)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
    "axes.titlesize":    13,
})


def _save(fig, name: str) -> None:
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def hill_curves(params: dict = None) -> None:
    """
    Show the three Hill force components vs normalised fibre length.
    This figure motivates WHY we need a numerical solver — the force is
    a nonlinear function of L_M with no closed-form inverse.
    """
    from biomechanics.hill_model import f_CE, f_PEE, f_SEE

    params  = params or config.BICEPS_LONGUS
    F_max   = params["F_max"]
    L_S0    = params["L_S0"]
    l_norms = np.linspace(0.5, 1.6, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Hill Muscle Model — Force Components\n"
                 "(Biceps Longus, Holzbaur 2005)", fontweight="bold")

    ax.plot(l_norms, [f_CE(l, 1.0, F_max) / F_max for l in l_norms],
            color=COLORS[0], lw=2, label="f_CE  active (a = 1.0)")
    ax.plot(l_norms, [f_CE(l, 0.5, F_max) / F_max for l in l_norms],
            color=COLORS[0], lw=2, ls="--", alpha=0.6, label="f_CE  active (a = 0.5)")
    ax.plot(l_norms, [f_PEE(l, F_max) / F_max for l in l_norms],
            color=COLORS[1], lw=2, label="f_PEE  passive")

    # f_SEE mapped onto the same x-axis via equivalent tendon strain
    strains  = np.linspace(0, 0.05, 300)
    L_S_vals = L_S0 * (1.0 + strains)
    x_see    = 1.0 + strains / 0.05 * 0.3   # rescale strain to [1.0, 1.3]
    ax.plot(x_see, [f_SEE(ls, L_S0, F_max) / F_max for ls in L_S_vals],
            color=COLORS[2], lw=2, label="f_SEE  tendon (strain rescaled)")

    ax.axvline(1.0, color="grey", ls=":", lw=1.2, label="Optimal length L_M0")
    ax.set_xlabel("Normalised fibre length  l / L_M0")
    ax.set_ylabel("Normalised force  F / F_max")
    ax.set_title("Nonlinear relationships — no closed-form inverse\n"
                 "⟹ Newton-Raphson required at every timestep")
    ax.legend(fontsize=9)
    _save(fig, "01_hill_curves.png")


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def force_workspace(workspace_results: dict, L_MT_range: np.ndarray) -> None:
    """
    Force field F(L_MT, activation) and solver iteration count.
    Shows WHAT the RL agent implicitly learns to replicate, and
    HOW expensive exact computation is at each configuration.
    """
    activations = list(workspace_results.keys())
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Biomechanical Force Field — What the RL Agent Must Learn",
                 fontweight="bold")

    ax = axes[0]
    for a, col in zip(activations, COLORS):
        ax.plot(L_MT_range * 100, workspace_results[a]["F"],
                color=col, lw=2, label=f"a = {a:.1f}")
    ax.set_xlabel("Musculo-tendinous length L_MT (cm)")
    ax.set_ylabel("Muscle force F (N)")
    ax.set_title("Force–Length Relationship\n(nonlinear, activation-dependent)")
    ax.legend()

    ax = axes[1]
    for a, col in zip(activations, COLORS):
        ax.plot(L_MT_range * 100, workspace_results[a]["iters"],
                color=col, lw=2, label=f"a = {a:.1f}")
    ax.axhline(50, color="red", ls=":", lw=1.2, label="Max iterations (50)")
    ax.set_xlabel("Musculo-tendinous length L_MT (cm)")
    ax.set_ylabel("Newton-Raphson iterations")
    ax.set_title("Computational Cost of Exact Solver\n(5–40 iterations per muscle per step)")
    ax.legend()
    _save(fig, "02_force_workspace.png")


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def benchmark_comparison(solver_stats: dict, rl_stats: dict) -> None:
    """Bar chart comparing wall-clock time: physics solver vs RL forward pass."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Computation Time per Control Step\n"
                 "(Physics Solver vs RL Policy)", fontweight="bold")

    means = [solver_stats["mean_us"], rl_stats["mean_us"]]
    stds  = [solver_stats["std_us"],  rl_stats["std_us"]]
    bars  = ax.bar([0, 1], means, yerr=stds, color=COLORS[:2], alpha=0.85,
                   capsize=8, error_kw={"elinewidth": 2.0}, width=0.5)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + max(stds) * 0.2,
                f"{m:.1f} µs", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Physics Solver\n(Newton-Raphson × 9)", "RL Policy\n(SAC forward pass)"],
        fontsize=10)
    ax.set_ylabel("Wall-clock time (µs)")

    speedup = solver_stats["mean_us"] / max(rl_stats["mean_us"], 1e-9)
    direction = "faster" if speedup > 1 else "slower"
    ax.set_title(f"RL is {abs(speedup):.1f}× {direction} per step\n"
                 f"(advantage scales with batch size and GPU deployment)")
    _save(fig, "03_benchmark.png")


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def training_curves(timesteps, actor_losses, critic_losses, ent_coefs,
                    actual_steps: int) -> None:
    """SAC training history — actor loss, critic loss, entropy coefficient."""
    ts  = np.array(timesteps)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"SAC Training Curves — Converged at {actual_steps:,} steps",
        fontweight="bold")

    pairs = [
        (actor_losses,  "Actor loss",           COLORS[0]),
        (critic_losses, "Critic loss",           COLORS[1]),
        (ent_coefs,     "Entropy coefficient α", COLORS[2]),
    ]
    for ax, (vals, label, col) in zip(axes, pairs):
        ax.plot(ts, vals, color=col, lw=1.2)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(label)
        ax.set_title(label)

    axes[2].axhline(0, color="grey", ls="--", lw=1)
    axes[2].set_title("Entropy coef α\n(decreasing ⟹ policy specialising)")
    _save(fig, "04_training_curves.png")


# ── Figure 5 ─────────────────────────────────────────────────────────────────

def statistical_comparison(summaries: dict, comparisons: list) -> None:
    """Mean ± 95% BCa CI bar chart + Wilcoxon p-value heatmap."""
    algos = list(summaries.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Statistical Comparison of RL vs Baseline Algorithms",
                 fontweight="bold")

    means   = [summaries[a]["mean"] * 100 for a in algos]
    errs_lo = [(summaries[a]["mean"] - summaries[a]["ci_lo"]) * 100 for a in algos]
    errs_hi = [(summaries[a]["ci_hi"] - summaries[a]["mean"]) * 100 for a in algos]
    bars = ax1.bar(algos, means, color=COLORS[:len(algos)], alpha=0.85,
                   yerr=[errs_lo, errs_hi], capsize=7,
                   error_kw={"elinewidth": 1.5})
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 1.5,
                 f"{m:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Mean success rate (%)")
    ax1.set_title("Mean ± 95% BCa Bootstrap CI")
    ax1.set_ylim(0, 112)

    n   = len(algos)
    mat = np.full((n, n), np.nan)
    for c in comparisons:
        i, j = algos.index(c["a"]), algos.index(c["b"])
        mat[i, j] = mat[j, i] = c["p"]

    im = ax2.imshow(mat, vmin=0, vmax=0.05, cmap="RdYlGn_r", aspect="auto")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(algos)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(algos)
    alpha_c = comparisons[0]["alpha_corrected"]
    ax2.set_title(f"Wilcoxon p-values\n(Bonferroni α = {alpha_c:.4f})")
    for c in comparisons:
        i, j = algos.index(c["a"]), algos.index(c["b"])
        txt  = f"{c['p']:.4f}\n{'*' if c['significant'] else 'ns'}"
        ax2.text(j, i, txt, ha="center", va="center", fontsize=9)
        ax2.text(i, j, txt, ha="center", va="center", fontsize=9)
    ax2.grid(False)
    plt.colorbar(im, ax=ax2, fraction=0.046, label="p-value")
    _save(fig, "05_statistical_comparison.png")
