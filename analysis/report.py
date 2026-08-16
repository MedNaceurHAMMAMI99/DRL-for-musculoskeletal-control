"""
Render LaTeX tables and figures directly from runs/results.json.

This is the bridge between the real pipeline output and the thesis/paper. It
writes self-contained LaTeX table fragments into ../Report/generated/*.tex, each
matching the label/caption of the corresponding hand-written table in
chapter6.tex, so the document `\\input`s real numbers the moment the grid has run.
It also regenerates the statistical-comparison figure from real per-seed data.

Every table is emitted ONLY if results.json actually contains the data for it, so
partial runs produce partial (but real) reporting rather than fabricated rows.

Usage
-----
    python -m analysis.report                 # from thesis_pipeline/
    python experiment_runner.py --stage report
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config

# Prefer the thesis Report/generated dir (in-repo); on a standalone compute machine
# where no Report/ exists, fall back to a local generated/ folder.
_REPORT_GEN = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Report", "generated"))
GEN_DIR = _REPORT_GEN if os.path.isdir(os.path.dirname(_REPORT_GEN)) else \
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "generated"))

# Display order used throughout the thesis tables.
DRL_ORDER = ["DDPG", "TD3", "SAC", "PPO"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load():
    if not os.path.exists(config.RESULTS_JSON):
        sys.exit(f"No {config.RESULTS_JSON}. Run experiment_runner.py first.")
    with open(config.RESULTS_JSON) as f:
        return json.load(f)


def _agg(per_seed_algo, key, scale=1.0):
    """Mean, std across seeds of a per-seed scalar metric."""
    vals = np.array([v[key] for v in per_seed_algo.values()]) * scale
    return float(np.mean(vals)), float(np.std(vals))


def _ms(mean, std, fmt="{:.1f}", bold=False):
    s = f"${fmt.format(mean)} \\pm {fmt.format(std)}$"
    return f"$\\mathbf{{{fmt.format(mean)} \\pm {fmt.format(std)}}}$" if bold else s


def _sci(x):
    """LaTeX scientific notation, e.g. 4.16e5 -> '4.16\\times10^{5}'."""
    if x == 0 or not np.isfinite(x):
        return f"{x:.2f}"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.2f}{{\\times}}10^{{{exp}}}"


def _sci_ms(mean, std, bold=False):
    inner = f"{_sci(mean)} \\pm {_sci(std)}"
    return f"$\\mathbf{{{inner}}}$" if bold else f"${inner}$"


def _write(name, body):
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  wrote {os.path.relpath(path)}")


def _best_algo(per_seed, key, maximize=True):
    means = {a: np.mean([v[key] for v in per_seed[a].values()])
             for a in per_seed if per_seed[a]}
    if not means:
        return None
    return (max if maximize else min)(means, key=means.get)


# ── tables ────────────────────────────────────────────────────────────────────

def table_results(res):
    per_seed = res.get("per_seed", {})
    baselines = res.get("baselines", {})
    algos = [a for a in DRL_ORDER if per_seed.get(a)]
    if not algos:
        return
    # Only mark a "best" if the algorithms actually differ on success rate;
    # otherwise (e.g. all tied at 0% in an undertrained run) bold nothing.
    succ_means = {a: np.mean([v["success_rate"] for v in per_seed[a].values()])
                  for a in algos}
    best = _best_algo(per_seed, "success_rate", maximize=True) \
        if len(set(round(x, 6) for x in succ_means.values())) > 1 else None

    rows = []
    # Baselines first (single evaluation, no seed std).
    for bl, tag in [("Random", "Random (BL)"), ("PD", "Impedance PD (BL)")]:
        if bl in baselines:
            b = baselines[bl]
            rew = f"${b['mean_reward']:.0f}$" if "mean_reward" in b else "--"
            rows.append(f"    {tag:20s} & {rew} "
                        f"& ${b['success_rate']*100:.1f}$ & ${b['mean_final_error']*100:.1f}$ "
                        f"& ${_sci(b['mean_energy'])}$ \\\\")
    rows.append("    \\midrule")
    for a in algos:
        rmean, rstd = _agg(per_seed[a], "mean_reward")
        smean, sstd = _agg(per_seed[a], "success_rate", 100.0)
        dmean, dstd = _agg(per_seed[a], "mean_final_error", 100.0)
        emean, estd = _agg(per_seed[a], "mean_energy")
        bold = (a == best)
        name = f"\\textbf{{{a}}}" if bold else a
        row = (f"    {name:20s} & {_ms(rmean, rstd, '{:.0f}', bold)} "
               f"& {_ms(smean, sstd, '{:.1f}', bold)} "
               f"& {_ms(dmean, dstd, '{:.1f}', bold)} "
               f"& {_sci_ms(emean, estd, bold)} \\\\")
        if bold:
            row = "    \\rowcolor{lightblue!60}\n" + row
        rows.append(row)

    meta = res.get("meta", {})
    n = meta.get("seeds", [])
    body = _TABLE_RESULTS.format(n=len(n) or "n",
                                 eval_ep=meta.get("eval_episodes", ""),
                                 steps=f"{meta.get('train_steps', 0):,}",
                                 rows="\n".join(rows))
    _write("tab_results.tex", body)


def table_stats(res):
    comps = res.get("statistics", {}).get("comparisons", [])
    if not comps:
        return
    rows = []
    for c in comps:
        sig = "large" if abs(c["rb"]) > 0.5 else ("medium" if abs(c["rb"]) > 0.3 else "small")
        rows.append(f"    {c['a']} vs.\\ {c['b']} & ${c['W']:.0f}$ & ${c['p']:.4f}$ "
                    f"& ${c['rb']:+.2f}$ ({sig}) & {'yes' if c['significant'] else 'no'} \\\\")
    alpha = comps[0].get("alpha_corrected", 0.05)
    body = _TABLE_STATS.format(alpha=f"{alpha:.3f}", rows="\n".join(rows))
    _write("tab_stats.tex", body)


MIN_SYNERGY_SAMPLES = 100   # below this, a VAF is a trivial fit, not a result


def table_synergies(res):
    syn = res.get("synergies", {})
    algos = [a for a in DRL_ORDER if a in syn and "vaf_at_k" in syn[a]]
    if not algos:
        return

    # Guard: undertrained policies terminate almost immediately, so the activation
    # matrix has too few rows for a meaningful NMF (VAF -> 100% trivially on a
    # handful of points). Do NOT report that as a synergy result.
    max_samples = max(syn[a].get("n_samples", 0) for a in algos)
    if max_samples < MIN_SYNERGY_SAMPLES:
        _write("tab_synergies.tex", _SYNERGY_INSUFFICIENT.format(n=max_samples))
        return

    best = max(algos, key=lambda a: syn[a]["similarity_to_reference"])
    rows = []
    for a in algos:
        s = syn[a]
        bold = (a == best)
        name = f"\\textbf{{{a}}}" if bold else a
        vaf = f"{s['vaf_at_k']*100:.1f}"
        sim = f"{s['similarity_to_reference']:.2f}"
        nsy = f"{s['n_synergies_90']}"
        if bold:
            rows.append("    \\rowcolor{lightblue!60}")
            rows.append(f"    {name} & $\\mathbf{{{vaf}}}$ & $\\mathbf{{{sim}}}$ & $\\mathbf{{{nsy}}}$ \\\\")
        else:
            rows.append(f"    {name} & ${vaf}$ & ${sim}$ & ${nsy}$ \\\\")
    body = _TABLE_SYNERGIES.format(k=syn[algos[0]].get("k", config.N_SYNERGIES),
                                   rows="\n".join(rows))
    _write("tab_synergies.tex", body)


def table_robustness(res):
    rob = res.get("robustness", {})
    algos = [a for a in DRL_ORDER if a in rob]
    if not algos:
        return
    rows = []
    for a in algos:
        r = rob[a]
        clean = r.get("Clean", {}).get("success_rate", float("nan")) * 100
        pert  = r.get("Force 10N", {}).get("success_rate", float("nan")) * 100
        deg   = pert - clean
        rows.append(f"    {a} & ${clean:.1f}$ & ${pert:.1f}$ & ${deg:+.1f}$ \\\\")
    body = _TABLE_ROBUSTNESS.format(rows="\n".join(rows))
    _write("tab_robustness.tex", body)


def table_ablation(res):
    abl = res.get("ablation", {})
    algos = [a for a in DRL_ORDER if a in abl]
    if not algos:
        return
    # Rows are conditions; columns are algorithms.
    conds = ["full", "no_w2", "no_w3", "no_w4", "no_w1"]
    labels = {"full": "Full reward", "no_w2": "Without $r_\\text{energy}$",
              "no_w3": "Without $r_\\text{smooth}$", "no_w4": "Without $r_\\text{alive}$",
              "no_w1": "Without $r_\\text{task}$"}
    rows = []
    for cond in conds:
        if not any(cond in abl[a] for a in algos):
            continue
        cells = []
        for a in algos:
            c = abl[a].get(cond)
            cells.append(f"${c['success_rate']*100:.1f}$ / ${_sci(c['mean_energy'])}$"
                         if c else "--")
        rows.append(f"    {labels[cond]} & " + " & ".join(cells) + " \\\\")
    body = _TABLE_ABLATION.format(cols="".join(f"\\textbf{{{a}}} & " for a in algos).rstrip("& "),
                                  rows="\n".join(rows))
    _write("tab_ablation.tex", body)


def figures_from_results(res):
    """Regenerate the statistical-comparison figure from real per-seed data."""
    per_seed = res.get("per_seed", {})
    stats_block = res.get("statistics", {}).get("task_performance", {})
    comps = res.get("statistics", {}).get("comparisons", [])
    if not stats_block or not comps:
        return
    from analysis import figures
    summaries = {a: {"mean": s["success_mean"], "ci_lo": s["ci_lo"],
                     "ci_hi": s["ci_hi"]} for a, s in stats_block.items()}
    figures.statistical_comparison(summaries, comps)
    print("  regenerated 05_statistical_comparison.png")


# ── table skeletons ───────────────────────────────────────────────────────────

_TABLE_RESULTS = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{Task performance (mean $\pm$ standard deviation,
           $n={n}$~seeds, {eval_ep} evaluation episodes) under a
           compute-limited training budget ({steps} steps/seed);
           policies are \emph{{undertrained}} at this budget and success rates are
           reported as measured. The first rows are non-learning baselines.
           Energy cost is the mean per-step $\sum_m F_m^2$ (N$^2$).}}
  \label{{tab:results}}
  \rowcolors{{2}}{{lightblue!30}}{{white}}
  \begin{{tabular}}{{@{{}} L{{2.8cm}} C{{2.8cm}} C{{2.4cm}} C{{2.2cm}} C{{2.4cm}} @{{}}}}
    \toprule
    \rowcolor{{lightblue}}
    \textbf{{Algorithm}} & \textbf{{Reward}} & \textbf{{Success (\%)}} &
    \textbf{{Dist. (cm)}} & \textbf{{Energy cost}} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

_TABLE_STATS = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{Pairwise statistical comparison (paired Wilcoxon signed-rank test on
           per-seed success rate, Bonferroni-corrected
           $\alpha^\star={alpha}$; rank-biserial effect size).}}
  \label{{tab:stats}}
  \rowcolors{{2}}{{lightblue!30}}{{white}}
  \begin{{tabular}}{{@{{}} L{{3.0cm}} C{{2.0cm}} C{{2.2cm}} C{{2.8cm}} C{{2.0cm}} @{{}}}}
    \toprule
    \rowcolor{{lightblue}}
    \textbf{{Comparison}} & \textbf{{Stat.~$W$}} & \textbf{{$p$-value}} &
    \textbf{{$r_\text{{rb}}$}} & \textbf{{Signif.}} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

_TABLE_SYNERGIES = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{NMF goodness of fit ($k={k}$) for muscle activations, and cosine
           similarity of learned synergies to an approximate d'Avella (2006)
           upper-limb reference set (see \texttt{{analysis/synergies.py}}).
           No.\ synergies = smallest $k$ reaching 90\% VAF.}}
  \label{{tab:synergies}}
  \rowcolors{{2}}{{lightblue!30}}{{white}}
  \begin{{tabular}}{{@{{}} L{{2.8cm}} C{{2.4cm}} C{{3.2cm}} C{{2.8cm}} @{{}}}}
    \toprule
    \rowcolor{{lightblue}}
    \textbf{{Algorithm}} & \textbf{{VAF (\%)}} & \textbf{{Sim.\ to d'Avella 2006}} &
    \textbf{{No.\ synergies (VAF~$>$90~\%)}} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

_SYNERGY_INSUFFICIENT = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{Muscle-synergy (NMF/VAF) and co-contraction analyses are
           \textbf{{not reportable at the current compute budget}}. The
           undertrained policies terminate within the first control step(s), so
           the concatenated activation matrix contains only $n={n}$ samples---far
           too few for a meaningful $k{{=}}4$ factorisation (VAF is trivially
           $\approx$100\% on so few points). These analyses require policies that
           produce sustained multi-step reaching trajectories; the analysis code
           (\texttt{{analysis/synergies.py}}, \texttt{{cci.py}}) is implemented and
           will populate this table once such policies are trained.}}
  \label{{tab:synergies}}
\end{{table}}
"""

_TABLE_ROBUSTNESS = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{Robustness to a \SI{{10}}{{\newton}} force-impulse perturbation
           (success rate; evaluated on the trained checkpoints).}}
  \label{{tab:robustness}}
  \rowcolors{{2}}{{lightblue!30}}{{white}}
  \begin{{tabular}}{{@{{}} L{{3cm}} C{{3cm}} C{{3cm}} C{{3cm}} @{{}}}}
    \toprule
    \rowcolor{{lightblue}}
    \textbf{{Algo.}} & \textbf{{Without perturb. (\%)}} &
    \textbf{{With perturb. (\%)}} & \textbf{{Degradation (pts)}} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

_TABLE_ABLATION = r"""% AUTO-GENERATED by analysis/report.py from runs/results.json — do not hand-edit.
\begin{{table}}[H]
  \centering
  \caption{{Impact of reward-term ablation. Values: success rate (\%) / mean
           per-step energy (N$^2$).}}
  \label{{tab:ablation}}
  \rowcolors{{2}}{{lightblue!30}}{{white}}
  \small
  \begin{{tabularx}}{{\textwidth}}{{@{{}} L{{3.2cm}} *{{4}}{{>{{\centering\arraybackslash}}X}} @{{}}}}
    \toprule
    \rowcolor{{lightblue}}
    \textbf{{Configuration}} & {cols} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabularx}}
  \normalsize
\end{{table}}
"""


def main():
    res = _load()
    print(f"Rendering LaTeX from {config.RESULTS_JSON} -> {GEN_DIR}")
    table_results(res)
    table_stats(res)
    table_synergies(res)
    table_robustness(res)
    table_ablation(res)
    figures_from_results(res)
    print("Done. Add \\input{generated/<file>} in chapter6.tex (already wired "
          "with \\IfFileExists fallback).")


if __name__ == "__main__":
    main()
