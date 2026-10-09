#!/usr/bin/env python3
"""
render_publication_figures.py
=============================================================================
Renders 6 high-density publication-grade figures consolidating all modules:
  Figure 1: Repertoire Dynamics & Differential Syllables
  Figure 2: 4-Tier Hierarchical Clustering Grammar (Levels 1 to 4)
  Figure 3: Higher-Order Inter-Chain Organization Across Waves
  Figure 4: Higher-Order Intra-Chain Execution Fidelity & Stage Hazards
  Figure 5: Intra-Syllable & Intra-Frame Micro-Kinematics (Phase tau in [0, 1])
  Figure 6: Behavioral Transition Networks & 40-Min Session Fatigue
=============================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
BASE_DIR = REPO_ROOT / "syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study"
FIG_DIR = BASE_DIR / "07_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}
COND_ORDER = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]
COND_LABELS = ["3w PI\n(Base)", "8w PI\n(Mid)", "16w PI\n(Peak)", "24w PI\n(Late)", "L-Dopa\n(+Bens)"]

def add_rescue_shading(ax):
    ax.axvspan(3.5, 4.5, color="#f1f5f9", alpha=0.9, zorder=0)
    ax.axvline(x=3.5, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)


# -----------------------------------------------------------------------------
# FIGURE 1: REPERTOIRE DYNAMICS ACROSS WAVES
# -----------------------------------------------------------------------------
def render_figure_1():
    print("Rendering Figure 1: Repertoire Dynamics...")
    df_sess = pd.read_csv(BASE_DIR / "01_repertoire_dynamics/session_repertoire_metrics.csv")
    df_wave = pd.read_csv(BASE_DIR / "01_repertoire_dynamics/longitudinal_syllable_statistics_per_wave.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 1A: Heatmap of log2FC across all active syllables
    ax1 = axes[0, 0]
    # Pivot log2fc
    piv = df_wave.pivot(index="syllable_id", columns="condition", values="log2fc")[COND_ORDER]
    # Filter to syllables that reach |log2fc| >= 0.5 in at least one wave
    keep_sylls = piv.index[piv.abs().max(axis=1) >= 0.5].values
    if len(keep_sylls) > 30:
        # Top 30 by peak deviation
        keep_sylls = piv.abs().max(axis=1).sort_values(ascending=False).head(30).index.values
    piv_sub = piv.loc[sorted(keep_sylls)]
    
    sns.heatmap(piv_sub, cmap="vlag", center=0.0, vmin=-1.5, vmax=1.5, ax=ax1,
                cbar_kws={"label": "log2(PFF / Monomer)"}, linewidths=0.5)
    ax1.set_title("A. Differential Syllable Occupancy log2(PFF/Mon)", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(COND_LABELS, rotation=0, fontsize=9)
    ax1.set_ylabel("Syllable ID", fontsize=10)
    
    # 1B: Repertoire Entropy & Participation Ratio
    ax2 = axes[0, 1]
    add_rescue_shading(ax2)
    for g, color in PALETTE.items():
        sub = df_sess[df_sess["genotype"] == g]
        means = sub.groupby("condition")["shannon_entropy"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["shannon_entropy"].sem().reindex(COND_ORDER).values
        ax2.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} H(S)",
                     marker="o", linewidth=2.2, capsize=4, zorder=3)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(COND_LABELS, fontsize=9)
    ax2.set_ylabel("Shannon Entropy H(S) (bits)", fontsize=10)
    ax2.set_title("B. Repertoire Diversity Over Disease Progression", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    
    # 1C: Locomotor Bout Duration & Switch Rate
    ax3 = axes[1, 0]
    add_rescue_shading(ax3)
    for g, color in PALETTE.items():
        sub = df_sess[df_sess["genotype"] == g]
        means = sub.groupby("condition")["mean_bout_dur_s"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["mean_bout_dur_s"].sem().reindex(COND_ORDER).values
        ax3.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Bout Duration",
                     marker="s", linewidth=2.2, capsize=4, zorder=3)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(COND_LABELS, fontsize=9)
    ax3.set_ylabel("Mean Syllable Bout Duration (s)", fontsize=10)
    ax3.set_title("C. Behavioral Persistence (Dwell Time)", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left", frameon=True)
    
    # 1D: Key Phenotype Syllables Longitudinal Trajectories
    ax4 = axes[1, 1]
    add_rescue_shading(ax4)
    # Highlight Syllable 61 (slow explore, elevated in PFF) and Syllable 42 (dash, suppressed in PFF)
    for sid, style, lbl in [(61, "-", "Syll 61 (Slow Explore)"), (42, "--", "Syll 42 (Dash)")]:
        for g, color in PALETTE.items():
            sub = df_wave[df_wave["syllable_id"] == sid]
            means = sub.set_index("condition").loc[COND_ORDER, f"{g.lower()}_mean_pct"].values
            sems = sub.set_index("condition").loc[COND_ORDER, f"{g.lower()}_sem_pct"].values
            ax4.errorbar(range(5), means, yerr=sems, color=color, linestyle=style,
                         label=f"{g} - {lbl}", marker="o" if style=="-" else "^",
                         linewidth=2.0, capsize=3, zorder=3)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(COND_LABELS, fontsize=9)
    ax4.set_ylabel("Occupancy (% recording time)", fontsize=10)
    ax4.set_title("D. Top Divergent Syllable Trajectories", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=8.5, frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_1_repertoire_longitudinal_master.png")
    plt.close()
    print("Saved Figure 1.")


# -----------------------------------------------------------------------------
# FIGURE 2: HIERARCHICAL CLUSTERING GRAMMAR (LEVELS 1 TO 4)
# -----------------------------------------------------------------------------
def render_figure_2():
    print("Rendering Figure 2: Hierarchical Clustering (Levels 1 to 4)...")
    df_cov = pd.read_csv(BASE_DIR / "02_clustering_levels/hierarchical_clustering_cohort_coverage.csv")
    df_sess = pd.read_csv(BASE_DIR / "02_clustering_levels/hierarchical_clustering_per_session.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 2A: Population Coverage by Resolution Level (5-Step)
    ax1 = axes[0, 0]
    sub5 = df_cov[df_cov["horizon"] == 5]
    levels = ["L1", "L2", "L3", "L4"]
    level_labels = ["Level 1\n(100 states)", "Level 2\n(93 syn)", "Level 3\n(22 meso)", "Level 4\n(14 macro)"]
    
    for g, color in PALETTE.items():
        g_means = []
        g_sems = []
        for lvl in levels:
            sub_gl = sub5[(sub5["genotype"] == g) & (sub5["level"] == lvl)]
            g_means.append(sub_gl["mean_population_coverage_pct"].mean())
            g_sems.append(sub_gl["mean_population_coverage_pct"].sem())
        ax1.plot(range(4), g_means, color=color, marker="o", linewidth=2.5, label=f"{g} (Mean across waves)")
        ax1.fill_between(range(4), np.array(g_means) - np.array(g_sems), np.array(g_means) + np.array(g_sems),
                         color=color, alpha=0.15)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(level_labels, fontsize=9)
    ax1.set_ylabel("5-Step Population Time Coverage (%)", fontsize=10)
    ax1.set_title("A. Multi-Animal Structured Time by Vocabulary Resolution", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    
    # 2B: Recurrent Sequence Families Discovered
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(4)
    # Average across waves for 5-step
    m_fam = [sub5[(sub5["genotype"] == "Monomer") & (sub5["level"] == lvl)]["recurrent_families"].mean() for lvl in levels]
    p_fam = [sub5[(sub5["genotype"] == "PFF") & (sub5["level"] == lvl)]["recurrent_families"].mean() for lvl in levels]
    ax2.bar(x - width/2, m_fam, width, label="Monomer", color=PALETTE["Monomer"])
    ax2.bar(x + width/2, p_fam, width, label="PFF", color=PALETTE["PFF"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(level_labels, fontsize=9)
    ax2.set_ylabel("Recurrent Families (>=15 ep, >=5 mice)", fontsize=10)
    ax2.set_title("B. Recurring Behavioral Routines Discovery", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    
    # 2C: Physical Elapsed Duration of 5-Step Sequences (Consecutive bout merging)
    ax3 = axes[1, 0]
    dur_df = df_sess[df_sess["horizon"] == 5]
    sns.boxplot(data=dur_df, x="level", y="mean_duration_s", hue="genotype", palette=PALETTE, ax=ax3, showfliers=False)
    ax3.set_xticklabels(level_labels, fontsize=9)
    ax3.set_ylabel("Physical Elapsed Duration (s)", fontsize=10)
    ax3.set_title("C. Extension of Continuous Behavioral Arcs", fontsize=12, fontweight="bold")
    
    # 2D: Level 4 Longitudinal Coverage Trajectory
    ax4 = axes[1, 1]
    add_rescue_shading(ax4)
    sub_l4 = sub5[sub5["level"] == "L4"]
    for g, color in PALETTE.items():
        g_sub = sub_l4[sub_l4["genotype"] == g].set_index("condition").reindex(COND_ORDER)
        means = g_sub["mean_population_coverage_pct"].values
        sems = g_sub["sem_population_coverage_pct"].values
        ax4.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Level 4 Coverage",
                     marker="o", linewidth=2.5, capsize=4, zorder=3)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(COND_LABELS, fontsize=9)
    ax4.set_ylabel("Population Time Coverage (%)", fontsize=10)
    ax4.set_title("D. Macro-Syntax Population Coverage Across Waves", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_2_clustering_hierarchy_levels_1_to_4.png")
    plt.close()
    print("Saved Figure 2.")


# -----------------------------------------------------------------------------
# FIGURE 3: INTER-CHAIN ORGANIZATION ACROSS WAVES
# -----------------------------------------------------------------------------
def render_figure_3():
    print("Rendering Figure 3: Inter-Chain Organization...")
    df_inter = pd.read_csv(BASE_DIR / "03_inter_chain/inter_chain_organization_per_session.csv")
    df_wave = pd.read_csv(BASE_DIR / "03_inter_chain/inter_chain_wave_statistics.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 3A: Chain Perseveration Index P_persev
    ax1 = axes[0, 0]
    add_rescue_shading(ax1)
    for g, color in PALETTE.items():
        sub = df_inter[df_inter["genotype"] == g]
        means = sub.groupby("condition")["p_perseveration"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["p_perseveration"].sem().reindex(COND_ORDER).values
        ax1.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} P(persev)",
                     marker="o", linewidth=2.5, capsize=4, zorder=3)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(COND_LABELS, fontsize=9)
    ax1.set_ylabel("Perseveration Probability P(persev)", fontsize=10)
    ax1.set_title("A. Chain Perseveration (Routine Self-Looping)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True)
    
    # 3B: Continuation Hazard P(1 -> 2)
    ax2 = axes[0, 1]
    add_rescue_shading(ax2)
    for g, color in PALETTE.items():
        sub = df_inter[df_inter["genotype"] == g]
        means = sub.groupby("condition")["p_continue_1_to_2"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["p_continue_1_to_2"].sem().reindex(COND_ORDER).values
        ax2.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} P(continue 1->2)",
                     marker="s", linewidth=2.5, capsize=4, zorder=3)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(COND_LABELS, fontsize=9)
    ax2.set_ylabel("Continuation Hazard Rate", fontsize=10)
    ax2.set_title("B. Continuation Beyond Single Cycle", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    
    # 3C: Total Chains Executed per Session
    ax3 = axes[1, 0]
    add_rescue_shading(ax3)
    for g, color in PALETTE.items():
        sub = df_inter[df_inter["genotype"] == g]
        means = sub.groupby("condition")["total_chains_detected"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["total_chains_detected"].sem().reindex(COND_ORDER).values
        ax3.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Total Chains",
                     marker="^", linewidth=2.5, capsize=4, zorder=3)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(COND_LABELS, fontsize=9)
    ax3.set_ylabel("Chains Detected per Session", fontsize=10)
    ax3.set_title("C. Higher-Order Action Chunk Frequency", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    
    # 3D: Maximum Sustained Bout Length
    ax4 = axes[1, 1]
    add_rescue_shading(ax4)
    for g, color in PALETTE.items():
        sub = df_inter[df_inter["genotype"] == g]
        means = sub.groupby("condition")["max_repeat_bout"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["max_repeat_bout"].sem().reindex(COND_ORDER).values
        ax4.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Max Repeat Bout",
                     marker="D", linewidth=2.5, capsize=4, zorder=3)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(COND_LABELS, fontsize=9)
    ax4.set_ylabel("Max Consecutive Cycles in Bout", fontsize=10)
    ax4.set_title("D. Bout Span of Sustained Motor Routines", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_3_inter_chain_organization_master.png")
    plt.close()
    print("Saved Figure 3.")


# -----------------------------------------------------------------------------
# FIGURE 4: INTRA-CHAIN EXECUTION FIDELITY & HAZARDS
# -----------------------------------------------------------------------------
def render_figure_4():
    print("Rendering Figure 4: Intra-Chain Execution Fidelity...")
    df_intra = pd.read_csv(BASE_DIR / "04_intra_chain/intra_chain_execution_per_session.csv")
    df_wave = pd.read_csv(BASE_DIR / "04_intra_chain/intra_chain_wave_statistics.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 4A: Progression Rate by Candidate Routine (16w Peak Deficit)
    ax1 = axes[0, 0]
    sub16 = df_intra[df_intra["condition"] == "16w_PI"]
    chains = ["K1_Rear_Settle", "K3_Vigilance_Pause", "K4_Look_Scrunch", "K5_Sniff_Step_Sniff", "K6_Dash_Steer_Dash"]
    chain_labels = ["K1: Rear", "K3: Vigilance", "K4: Scrunch", "K5: Sniff-Step", "K6: Dash-Steer"]
    
    width = 0.35
    x = np.arange(len(chains))
    m_prog = [sub16[(sub16["genotype"] == "Monomer") & (sub16["chain_name"] == c)]["conditional_progression_rate"].mean() for c in chains]
    p_prog = [sub16[(sub16["genotype"] == "PFF") & (sub16["chain_name"] == c)]["conditional_progression_rate"].mean() for c in chains]
    ax1.bar(x - width/2, m_prog, width, label="Monomer", color=PALETTE["Monomer"])
    ax1.bar(x + width/2, p_prog, width, label="PFF", color=PALETTE["PFF"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(chain_labels, fontsize=9)
    ax1.set_ylabel("Conditional Progression Rate P(full | prefix)", fontsize=10)
    ax1.set_title("A. Routine Completion at Peak Deficit (16w PI)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    
    # 4B: K6 Dash-Steer Progression Timecourse Across Waves
    ax2 = axes[0, 1]
    add_rescue_shading(ax2)
    k6_sub = df_intra[df_intra["chain_name"] == "K6_Dash_Steer_Dash"]
    for g, color in PALETTE.items():
        sub = k6_sub[k6_sub["genotype"] == g]
        means = sub.groupby("condition")["conditional_progression_rate"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["conditional_progression_rate"].sem().reindex(COND_ORDER).values
        ax2.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} K6 Progression",
                     marker="o", linewidth=2.5, capsize=4, zorder=3)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(COND_LABELS, fontsize=9)
    ax2.set_ylabel("Progression Rate P(K6 | Syll 42)", fontsize=10)
    ax2.set_title("B. K6 Steering Routine Commitment Trajectory", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    
    # 4C: Stage 1 Exit Hazard (Premature Abort) Timecourse
    ax3 = axes[1, 0]
    add_rescue_shading(ax3)
    for g, color in PALETTE.items():
        sub = k6_sub[k6_sub["genotype"] == g]
        means = sub.groupby("condition")["stage1_exit_hazard"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["stage1_exit_hazard"].sem().reindex(COND_ORDER).values
        ax3.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Stage 1 Exit Hazard",
                     marker="s", linewidth=2.5, capsize=4, zorder=3)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(COND_LABELS, fontsize=9)
    ax3.set_ylabel("Stage 1 Abort Hazard Rate", fontsize=10)
    ax3.set_title("C. Early Truncation Hazard at Steering Transition", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)
    
    # 4D: Context-Dependent Speed Modulation Ratio inside vs outside
    ax4 = axes[1, 1]
    add_rescue_shading(ax4)
    for g, color in PALETTE.items():
        sub = k6_sub[k6_sub["genotype"] == g]
        means = sub.groupby("condition")["speed_modulation_ratio"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["speed_modulation_ratio"].sem().reindex(COND_ORDER).values
        ax4.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Speed Modulation Ratio",
                     marker="^", linewidth=2.5, capsize=4, zorder=3)
    ax4.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(COND_LABELS, fontsize=9)
    ax4.set_ylabel("Speed Modulation Ratio (v_chain / v_base)", fontsize=10)
    ax4.set_title("D. Context Modulation (Acceleration Inside Routine)", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_4_intra_chain_execution_and_hazards.png")
    plt.close()
    print("Saved Figure 4.")


# -----------------------------------------------------------------------------
# FIGURE 5: INTRA-SYLLABLE & INTRA-FRAME MICRO-KINEMATICS
# -----------------------------------------------------------------------------
def render_figure_5():
    print("Rendering Figure 5: Intra-Syllable & Intra-Frame Micro-Kinematics...")
    df_prof = pd.read_csv(BASE_DIR / "05_intra_frame/micro_velocity_tau_profiles.csv")
    df_micro = pd.read_csv(BASE_DIR / "05_intra_frame/intra_frame_micro_bouts.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 5A: Phase tau Velocity Profile for Syllable 42 (Dash) at 16w Peak Deficit
    ax1 = axes[0, 0]
    p16 = df_prof[(df_prof["syllable_id"] == 42) & (df_prof["condition"] == "16w_PI")]
    for g, color in PALETTE.items():
        sub = p16[p16["genotype"] == g]
        ax1.plot(sub["tau"], sub["mean_velocity"], color=color, linewidth=2.5, label=f"{g} (16w PI)")
        ax1.fill_between(sub["tau"], sub["mean_velocity"] - sub["sem_velocity"], sub["mean_velocity"] + sub["sem_velocity"],
                         color=color, alpha=0.18)
    ax1.set_xlabel("Normalized Execution Phase tau (0 = onset, 1 = offset)", fontsize=10)
    ax1.set_ylabel("Centroid Velocity (px/s)", fontsize=10)
    ax1.set_title("A. Syllable 42 (Dash) Micro-Velocity Profile (16w PI)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower center", frameon=True)
    
    # 5B: Phase tau Velocity Profile for Syllable 42 under L-DOPA Rescue
    ax2 = axes[0, 1]
    p_ld = df_prof[(df_prof["syllable_id"] == 42) & (df_prof["condition"] == "L_dopa")]
    for g, color in PALETTE.items():
        sub = p_ld[p_ld["genotype"] == g]
        ax2.plot(sub["tau"], sub["mean_velocity"], color=color, linewidth=2.5, label=f"{g} (L-Dopa)")
        ax2.fill_between(sub["tau"], sub["mean_velocity"] - sub["sem_velocity"], sub["mean_velocity"] + sub["sem_velocity"],
                         color=color, alpha=0.18)
    ax2.set_xlabel("Normalized Execution Phase tau (0 = onset, 1 = offset)", fontsize=10)
    ax2.set_ylabel("Centroid Velocity (px/s)", fontsize=10)
    ax2.set_title("B. Pharmacological Normalization of Syllable 42 Profile", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower center", frameon=True)
    
    # 5C: Initiation Lag (Peak Phase Timing tau_peak) Across Waves
    ax3 = axes[1, 0]
    add_rescue_shading(ax3)
    s42_bouts = df_micro[df_micro["syllable_id"] == 42]
    for g, color in PALETTE.items():
        sub = s42_bouts[s42_bouts["genotype"] == g]
        means = sub.groupby("condition")["peak_tau"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["peak_tau"].sem().reindex(COND_ORDER).values
        ax3.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} tau_peak",
                     marker="o", linewidth=2.5, capsize=4, zorder=3)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(COND_LABELS, fontsize=9)
    ax3.set_ylabel("Peak Velocity Phase tau_peak", fontsize=10)
    ax3.set_title("C. Initiation Lag in Dash Syllable Across Disease Waves", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)
    
    # 5D: Launch Speed (tau = 0.0) Across Waves
    ax4 = axes[1, 1]
    add_rescue_shading(ax4)
    for g, color in PALETTE.items():
        sub = s42_bouts[s42_bouts["genotype"] == g]
        means = sub.groupby("condition")["launch_speed"].mean().reindex(COND_ORDER).values
        sems = sub.groupby("condition")["launch_speed"].sem().reindex(COND_ORDER).values
        ax4.errorbar(range(5), means, yerr=sems, color=color, label=f"{g} Launch Speed",
                     marker="s", linewidth=2.5, capsize=4, zorder=3)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(COND_LABELS, fontsize=9)
    ax4.set_ylabel("Launch Velocity v(tau=0) (px/s)", fontsize=10)
    ax4.set_title("D. Transition Launch Velocity Exiting Prior Syllable", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_5_intra_frame_micro_kinematics.png")
    plt.close()
    print("Saved Figure 5.")


# -----------------------------------------------------------------------------
# FIGURE 6: NETWORKS & SESSION-TIME MOTOR FATIGUE
# -----------------------------------------------------------------------------
def render_figure_6():
    print("Rendering Figure 6: Networks & Session Fatigue...")
    df_fatigue = pd.read_csv(BASE_DIR / "06_networks_fatigue/session_time_fatigue_curves.csv")
    df_dod = pd.read_csv(BASE_DIR / "06_networks_fatigue/dod_progression_16w_vs_3w.csv", index_col=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # 6A: Difference-of-Differences Progression Heatmap (16w vs 3w)
    ax1 = axes[0, 0]
    sns.heatmap(df_dod.iloc[:15, :15], cmap="vlag", center=0.0, vmin=-0.08, vmax=0.08, ax=ax1,
                cbar_kws={"label": "DoD Shift (Prob)"}, linewidths=0.5)
    ax1.set_title("A. Progression DoD Network Rewiring ([16w-3w]_PFF - [16w-3w]_Mon)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Target Syllable", fontsize=9)
    ax1.set_ylabel("Source Syllable", fontsize=9)
    
    # 6B: Within-Session Locomotor Fatigue (16w Peak Deficit)
    ax2 = axes[0, 1]
    f16 = df_fatigue[df_fatigue["condition"] == "16w_PI"]
    for g, color in PALETTE.items():
        sub = f16[f16["genotype"] == g]
        means = sub.groupby("minute_center")["velocity_px_s"].mean()
        sems = sub.groupby("minute_center")["velocity_px_s"].sem()
        ax2.plot(means.index, means.values, color=color, linewidth=2.5, label=f"{g} (16w PI)")
        ax2.fill_between(means.index, means.values - sems.values, means.values + sems.values, color=color, alpha=0.18)
    ax2.set_xlabel("Session Elapsed Time (minutes)", fontsize=10)
    ax2.set_ylabel("Velocity (px/s)", fontsize=10)
    ax2.set_title("B. Within-Session Locomotor Decay at Peak Deficit (16w)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    
    # 6C: Within-Session Locomotor Fatigue Under L-DOPA Rescue
    ax3 = axes[1, 0]
    f_ld = df_fatigue[df_fatigue["condition"] == "L_dopa"]
    for g, color in PALETTE.items():
        sub = f_ld[f_ld["genotype"] == g]
        means = sub.groupby("minute_center")["velocity_px_s"].mean()
        sems = sub.groupby("minute_center")["velocity_px_s"].sem()
        ax3.plot(means.index, means.values, color=color, linewidth=2.5, label=f"{g} (L-Dopa)")
        ax3.fill_between(means.index, means.values - sems.values, means.values + sems.values, color=color, alpha=0.18)
    ax3.set_xlabel("Session Elapsed Time (minutes)", fontsize=10)
    ax3.set_ylabel("Velocity (px/s)", fontsize=10)
    ax3.set_title("C. Abolition of Motor Exhaustion Under L-DOPA", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    
    # 6D: Within-Session Transition Entropy Dynamics (16w PI)
    ax4 = axes[1, 1]
    for g, color in PALETTE.items():
        sub = f16[f16["genotype"] == g]
        means = sub.groupby("minute_center")["transition_entropy"].mean()
        sems = sub.groupby("minute_center")["transition_entropy"].sem()
        ax4.plot(means.index, means.values, color=color, linewidth=2.5, label=f"{g} Entropy Rate")
        ax4.fill_between(means.index, means.values - sems.values, means.values + sems.values, color=color, alpha=0.18)
    ax4.set_xlabel("Session Elapsed Time (minutes)", fontsize=10)
    ax4.set_ylabel("Transition Entropy Rate (bits/step)", fontsize=10)
    ax4.set_title("D. Syntax Disorder Over Session Time (16w PI)", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure_6_networks_and_session_fatigue.png")
    plt.close()
    print("Saved Figure 6.")


def main():
    print("=" * 80)
    print("RENDERING PUBLICATION FIGURES FOR MASTER STUDY")
    print("=" * 80)
    render_figure_1()
    render_figure_2()
    render_figure_3()
    render_figure_4()
    render_figure_5()
    render_figure_6()
    print("=" * 80)
    print("ALL 6 FIGURES SUCCESSFULLY GENERATED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
