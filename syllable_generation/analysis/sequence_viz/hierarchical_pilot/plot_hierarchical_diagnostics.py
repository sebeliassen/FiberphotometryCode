#!/usr/bin/env python3
"""
plot_hierarchical_diagnostics.py
=============================================================================
Generates comprehensive 6-panel diagnostic figures for the Repaired Hierarchical
Shared-Subroutine Behavioral Model:
- Panel A: Learned Routine Stage Emissions (Ordered syllable progressions)
- Panel B: Progression Experiment (Free vs. Tied Stage Emissions Held-out LL/bout)
- Panel C: Control A (Real Data vs. Unbiased 1st-Order Markov Surrogates)
- Panel D: Boundary-Clean Routine Dwell (Routine 1 Long vs. Routines 2-6 Short)
- Panel E: Macro-State Routing A^(macro) and Routine Selection Preferences B^(routine)
- Panel F: Control B (Initialization Sensitivity: Initial vs. Learned Durations)
=============================================================================
"""

import sys
from pathlib import Path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
BUNDLE_PKL = OUTPUT_DIR / "hierarchical_model_results_bundle_repaired.pkl"


def main():
    if not BUNDLE_PKL.exists():
        print(f"Results bundle not found at {BUNDLE_PKL}")
        return

    with open(BUNDLE_PKL, "rb") as f:
        data = pickle.load(f)

    model = data["model_free"]
    df_cv = data["progression_cv"]
    df_surr = data["surrogate_summary"]
    df_init = data["init_summary"]
    df_routines = data["df_routines"]
    df_rep = data["df_rep"]
    df_macro = data["df_macro"]
    df_rout_sum = data["df_rout_sum"]

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # ---------------------------------------------------------
    # Panel A: Routine Stage Dominant Emissions
    # ---------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    K = model.K
    S = model.S
    top_sylls = np.zeros((K, S), dtype=int)
    top_probs = np.zeros((K, S), dtype=float)
    for k in range(K):
        for s in range(S):
            best_s = int(np.argmax(model.theta[k, s]))
            top_sylls[k, s] = best_s
            top_probs[k, s] = float(model.theta[k, s, best_s])

    sns.heatmap(top_probs, annot=top_sylls, fmt="d", cmap="YlGnBu", cbar=True,
                yticklabels=[f"Rout {k+1} (E[D]={3.0/model.p_advance[k]:.1f}b)" for k in range(K)],
                xticklabels=[f"Stage {s+1}" for s in range(S)], ax=ax_a)
    ax_a.set_title("A. Routine Stage Syllable Progressions\n(Numbers: Top Syllable ID; Color: Probability)", fontsize=11, fontweight="bold")

    # ---------------------------------------------------------
    # Panel B: The Progression Experiment (Free vs. Tied)
    # ---------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    folds = [f"Fold {f}" for f in df_cv["fold"]]
    x = np.arange(len(folds))
    w = 0.35
    ax_b.bar(x - w/2, df_cv["free_stage_heldout_ll_per_bout"], w, label="Free Stage Progression", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax_b.bar(x + w/2, df_cv["tied_stage_heldout_ll_per_bout"], w, label="Tied Stage Mixture", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(folds, fontsize=10)
    ax_b.set_ylabel("Held-Out Log-Likelihood / Bout", fontsize=10, fontweight="bold")
    mean_delta = df_cv["delta_ll_per_bout"].mean()
    ax_b.set_title(f"B. Progression Experiment (Free vs. Tied)\n(Mean Held-Out Advantage: {mean_delta:+.4f} / bout)", fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=9, loc="lower right")
    ax_b.grid(True, alpha=0.3, axis="y")

    # ---------------------------------------------------------
    # Panel C: Control A (Real vs. Unbiased Surrogates)
    # ---------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    real_clean_routs = df_routines[~df_routines["truncated"]]
    real_bouts = real_clean_routs["bouts"].values
    surr_bouts = df_surr["mean_routine_bouts"].values

    sns.histplot(real_bouts, bins=np.arange(3, 22), color="#1f77b4", stat="density", label="Real Cohort (Boundary-Clean)", kde=True, ax=ax_c, alpha=0.6)
    ax_c.axvline(np.mean(real_bouts), color="#1f77b4", linestyle="--", linewidth=2, label=f"Real Mean ({np.mean(real_bouts):.2f}b)")
    ax_c.axvline(np.mean(surr_bouts), color="#d62728", linestyle="-.", linewidth=2, label=f"Surrogate Mean ({np.mean(surr_bouts):.2f}b)")
    ax_c.set_xlabel("Routine Dwell (Bouts)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax_c.set_title("C. Routine Dwell: Real vs. Unbiased Surrogates\n(Control A: Exact Sparsity Null)", fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Panel D: Dwell Breakdown (Routine 1 vs. Routines 2-6)
    # ---------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    rout_names = [f"R{r}" for r in df_rout_sum["routine_id"]]
    bars = ax_d.bar(rout_names, df_rout_sum["mean_bouts"], color=["#d62728" if r==1 else "#2ca02c" for r in df_rout_sum["routine_id"]], alpha=0.85, edgecolor="black")
    ax_d.set_xlabel("Routine ID", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Mean Dwell (Bouts)", fontsize=10, fontweight="bold")
    ax_d.set_title("D. Mean Dwell per Routine\n(Routine 1 Long vs. Routines 2-6 Short)", fontsize=11, fontweight="bold")
    for bar, pct in zip(bars, df_rout_sum["bout_share_pct"]):
        h = bar.get_height()
        ax_d.annotate(f"{h:.1f}b\n({pct:.0f}% bts)", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax_d.grid(True, alpha=0.3, axis="y")

    # ---------------------------------------------------------
    # Panel E: Upper-State Macro Routing & Preferences
    # ---------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    sns.heatmap(model.B_routine, annot=True, fmt=".2f", cmap="Purples", cbar=True,
                xticklabels=[f"Rout {k+1}" for k in range(model.K)],
                yticklabels=[f"Macro U{u+1}" for u in range(model.M)], ax=ax_e)
    ax_e.set_title("E. Routine Selection Preferences B^(routine)\n(Conditional on Upper Macro State)", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("Routine Choice", fontsize=10)
    ax_e.set_ylabel("Active Macro State", fontsize=10)

    # ---------------------------------------------------------
    # Panel F: Initialization / Prior Sensitivity (Control B)
    # ---------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    regimes = df_init["regime"].unique()
    x_positions = np.arange(len(regimes))
    init_durs = [df_init[df_init["regime"] == r]["initial_expected_duration"].iloc[0] for r in regimes]
    learned_means = [df_init[df_init["regime"] == r]["learned_expected_duration_mean"].mean() for r in regimes]
    learned_stds = [df_init[df_init["regime"] == r]["learned_expected_duration_mean"].std() for r in regimes]

    width = 0.35
    ax_f.bar(x_positions - width/2, init_durs, width, label="Initial Assumption", color="#7f7f7f", alpha=0.7, edgecolor="black")
    ax_f.bar(x_positions + width/2, learned_means, width, yerr=learned_stds, label="Learned Posterior (Mean ± Std)",
             color="#ff7f0e", alpha=0.85, edgecolor="black", capsize=5)
    ax_f.set_xticks(x_positions)
    ax_f.set_xticklabels(regimes, fontsize=9, fontweight="bold")
    ax_f.set_ylabel("Expected Routine Duration (Bouts)", fontsize=10, fontweight="bold")
    ax_f.set_title("F. Prior Sensitivity Sweep (Control B)\n(Multi-Sequence Independent Fitting)", fontsize=11, fontweight="bold")
    ax_f.legend(fontsize=9)
    ax_f.grid(True, alpha=0.3, axis="y")

    plot_path = OUTPUT_DIR / "fig_hierarchical_pilot_diagnostics_repaired.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved repaired diagnostic figure to: {plot_path.name}")


if __name__ == "__main__":
    main()
