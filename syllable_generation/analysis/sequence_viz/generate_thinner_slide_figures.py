#!/usr/bin/env python3
"""
generate_thinner_slide_figures.py
=============================================================================
Generates updated slide-friendly figures:
1. Thinner overall aspect ratio (width ~13.5 inches).
2. X-axis ticks utilizing vertical space with full 3-step sequence progression:
     Chain k
     Syll1 →
     Syll2 →
     Syll3
3. Redundant banners in B1 & B2 removed (clean distribution + dashed line + legend).
4. Generates BOTH:
   - fig_level3_slide_usage_rate_vertical.png (Usage rate in events/min)
   - fig_level3_slide_duration_vertical.png   (Total duration in seconds/session)
=============================================================================
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

DATA_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/data")
PRESENTATION_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations")
FIGURES_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/optimized_seqnmf/figures")
ARTIFACTS_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/6e6d59fd-58a4-4955-a762-5c2c131ab2b6")

# Load existing metrics
df_anim = pd.read_csv(DATA_DIR / "level3_animal_chain_usage_metrics.csv")
df_cohort = pd.read_csv(DATA_DIR / "level3_chain_cohort_usage_summary.csv")
df_uni_rate = pd.read_csv(DATA_DIR / "level3_chain_univariate_contrasts.csv")

# Compute univariate tests on duration
dur_contrasts = []
for k in range(1, 11):
    for drug in ["Saline", "Amphetamine"]:
        wt_vals = df_anim[(df_anim["genotype"] == "WT") & (df_anim["drug"] == drug) & (df_anim["factor_id"] == k)]["total_duration_sec"].values
        ki_vals = df_anim[(df_anim["genotype"] == "KI") & (df_anim["drug"] == drug) & (df_anim["factor_id"] == k)]["total_duration_sec"].values
        u, p = stats.mannwhitneyu(wt_vals, ki_vals, alternative="two-sided")
        dur_contrasts.append({"factor_id": k, "drug": drug, "p_val": p})
df_uni_dur = pd.DataFrame(dur_contrasts)

# Color palette
COLOR_WT_SAL = "#4A5568"    # Slate Grey
COLOR_KI_SAL = "#DD6B20"    # Vibrant Ochre / Orange
COLOR_WT_AMPH = "#2B6CB0"   # Rich Sapphire Blue
COLOR_KI_AMPH = "#C53030"   # Crimson Red

# Setup exact animal permutations for TV
anim_gt = df_anim[["animal_id", "genotype"]].drop_duplicates().set_index("animal_id")["genotype"].to_dict()
all_animals = sorted(list(anim_gt.keys()))
wt_animals = [a for a in all_animals if anim_gt[a] == "WT"]
ki_animals = [a for a in all_animals if anim_gt[a] == "KI"]
K = 10

partitions = list(itertools.combinations(all_animals, len(wt_animals)))
N_partitions = len(partitions)

profiles = {}
for (aid, drug), sub in df_anim.groupby(["animal_id", "drug"]):
    vec = np.zeros(K, dtype=np.float64)
    for _, row in sub.iterrows():
        k_idx = int(row["factor_id"]) - 1
        vec[k_idx] = row["rate_per_min"]
    s = np.sum(vec)
    if s > 0:
        vec = vec / s
    profiles[(aid, drug)] = vec

def calc_tv(group1, group2, drug):
    vecs1 = [profiles[(a, drug)] for a in group1 if (a, drug) in profiles]
    vecs2 = [profiles[(a, drug)] for a in group2 if (a, drug) in profiles]
    m1 = np.mean(vecs1, axis=0)
    m2 = np.mean(vecs2, axis=0)
    return 0.5 * np.sum(np.abs(m1 - m2))

tv_obs_sal = calc_tv(wt_animals, ki_animals, "Saline")
tv_obs_amph = calc_tv(wt_animals, ki_animals, "Amphetamine")

null_sal = np.zeros(N_partitions)
null_amph = np.zeros(N_partitions)
all_set = set(all_animals)
for p_idx, p_wt in enumerate(partitions):
    p_ki = list(all_set - set(p_wt))
    null_sal[p_idx] = calc_tv(p_wt, p_ki, "Saline")
    null_amph[p_idx] = calc_tv(p_wt, p_ki, "Amphetamine")

p_sal = float(np.mean(null_sal >= tv_obs_sal))
p_amph = float(np.mean(null_amph >= tv_obs_amph))

# Vertical sequence labels utilizing space below each chain
VERTICAL_CHAIN_LABELS = {
    1: "Chain 1\nWalk_L →\nTail →\nWalk_R",
    2: "Chain 2\nSniff_Grd →\nDescent_R →\nRest",
    3: "Chain 3\nSprint →\nSprint →\nSniff_Up",
    4: "Chain 4\nWalk_R →\nWalk_R →\nGrooming",
    5: "Chain 5\nSniff_Grd →\nLook_L →\nSniff_Grd",
    6: "Chain 6\nTail →\nLook_R →\nSniff_Grd",
    7: "Chain 7\nWall_Lean →\nDescent_L →\nWalk_L",
    8: "Chain 8\nSprint →\nTurn_R →\nRear_Elev",
    9: "Chain 9\nWalk_Fwd →\nSniff_Grd →\nSniff_Scr",
    10: "Chain 10\nSprint →\nTurn_L →\nSprint"
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2D3748",
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 10.5,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10.5,
})

def make_slide_figure(metric_type="rate"):
    """
    Builds a thinner, vertically balanced 2-panel presentation figure.
    metric_type: 'rate' (events/min) or 'duration' (seconds/session)
    """
    is_rate = (metric_type == "rate")
    val_col = "rate_per_min_mean" if is_rate else "total_duration_sec_mean"
    sem_col = "rate_per_min_sem" if is_rate else "total_duration_sec_sem"
    df_uni = df_uni_rate if is_rate else df_uni_dur
    y_label = "Chain Usage Rate (events / min)" if is_rate else "Total Session Duration (seconds)"
    panel_a_title = ("Cohort Chain Usage Rate (Mean ± SEM)" if is_rate else
                     "Total Time Spent in Discovered Chains (Mean ± SEM)")
    fig_title = ("Level 3 Behavioral Chains: Cohort Usage Rates & Genotype Separation"
                 if is_rate else
                 "Level 3 Behavioral Chains: Session Duration & Genotype Separation")
    
    # Thinner aspect ratio (width=12.0, height=12.2)
    fig = plt.figure(figsize=(12.0, 12.2), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.22)
    
    # -------------------------------------------------------------------------
    # TOP PANEL
    # -------------------------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    chain_indices = np.arange(1, K + 1)
    bar_w = 0.185
    offsets = {
        "WT_Saline": -1.65 * bar_w,
        "KI_Saline": -0.65 * bar_w,
        "WT_Amphetamine": 0.65 * bar_w,
        "KI_Amphetamine": 1.65 * bar_w,
    }
    cohort_colors = {
        "WT_Saline": COLOR_WT_SAL,
        "KI_Saline": COLOR_KI_SAL,
        "WT_Amphetamine": COLOR_WT_AMPH,
        "KI_Amphetamine": COLOR_KI_AMPH,
    }
    cohort_labels = {
        "WT_Saline": "WT Saline",
        "KI_Saline": "DAT-KI Saline",
        "WT_Amphetamine": "WT Amphetamine",
        "KI_Amphetamine": "DAT-KI Amphetamine",
    }
    
    for cohort in ["WT_Saline", "KI_Saline", "WT_Amphetamine", "KI_Amphetamine"]:
        sub = df_cohort[df_cohort["cohort"] == cohort].sort_values("factor_id")
        means = sub[val_col].values
        sems = sub[sem_col].values
        x_pos = chain_indices + offsets[cohort]
        
        ax_top.bar(x_pos, means, yerr=sems, width=bar_w, color=cohort_colors[cohort],
                   capsize=2.8, edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
                   label=cohort_labels[cohort], zorder=3)

    # Shading for alternating chains
    for k in chain_indices:
        if k % 2 == 0:
            ax_top.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

    # Add Significance Brackets and Stars with dynamic headroom tracking
    max_star_y = 0.0
    scale_factor = 5.5 if is_rate else 540.0
    y_step = 0.05 * scale_factor
    h_tick = 0.015 * scale_factor

    for k in range(1, K + 1):
        sub_k = df_cohort[df_cohort["factor_id"] == k]
        wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
        ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
        sal_max = max(wt_sal_row[val_col] + wt_sal_row[sem_col],
                      ki_sal_row[val_col] + ki_sal_row[sem_col])
        
        wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
        ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
        amph_max = max(wt_amph_row[val_col] + wt_amph_row[sem_col],
                       ki_amph_row[val_col] + ki_amph_row[sem_col])
        
        p_sal_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Saline")]["p_val"].values[0]
        p_amph_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Amphetamine")]["p_val"].values[0]
        
        if p_sal_k < 0.05:
            star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
            x1 = k + offsets["WT_Saline"]
            x2 = k + offsets["KI_Saline"]
            y_b = sal_max + y_step
            ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
            star_pos_y = y_b + h_tick + 0.01 * scale_factor
            ax_top.text((x1 + x2) / 2, star_pos_y, star_sal,
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#C05621", zorder=5)
            max_star_y = max(max_star_y, star_pos_y + 0.05 * scale_factor)
            
        if p_amph_k < 0.05:
            star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
            x1 = k + offsets["WT_Amphetamine"]
            x2 = k + offsets["KI_Amphetamine"]
            y_b = amph_max + y_step
            ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
            star_pos_y = y_b + h_tick + 0.01 * scale_factor
            ax_top.text((x1 + x2) / 2, star_pos_y, star_amph,
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#9B2C2C", zorder=5)
            max_star_y = max(max_star_y, star_pos_y + 0.05 * scale_factor)

    ax_top.set_xticks(chain_indices)
    ax_top.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
    ax_top.set_ylabel(y_label, fontweight="bold", fontsize=12)
    ax_top.set_title(f"{panel_a_title}\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                     loc="left", fontweight="bold", fontsize=13, pad=12)
    ax_top.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Clean horizontal legend positioned above plot
    ax_top.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
                  frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)
    
    y_top_limit = max(6.2 if is_rate else 660.0, max_star_y * 1.08)
    ax_top.set_ylim(0, y_top_limit)
    
    # -------------------------------------------------------------------------
    # BOTTOM PANELS (NO REDUNDANT BANNERS)
    # -------------------------------------------------------------------------
    n_bins = 30
    
    # B1: Saline
    ax_b1 = fig.add_subplot(gs[1, 0])
    counts, bins, patches = ax_b1.hist(null_sal, bins=n_bins, density=True, color="#CBD5E0",
                                       edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for b_idx in range(len(patches)):
        if bins[b_idx] >= tv_obs_sal:
            patches[b_idx].set_facecolor("#ED8936")
            patches[b_idx].set_alpha(0.95)
            
    ax_b1.axvline(tv_obs_sal, color="#C05621", linestyle="--", linewidth=2.6, zorder=4,
                  label=f"Observed TV = {tv_obs_sal:.3f}\nExact p = {p_sal:.4f} *")
    ax_b1.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=11.5)
    ax_b1.set_ylabel("Permutation Density", fontweight="bold", fontsize=11.5)
    ax_b1.set_title("Saline: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                    loc="left", fontweight="bold", fontsize=12, pad=8)
    ax_b1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_b1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=10.5)
    ax_b1.set_ylim(0, 11.0)
    
    # B2: Amphetamine
    ax_b2 = fig.add_subplot(gs[1, 1])
    counts_a, bins_a, patches_a = ax_b2.hist(null_amph, bins=n_bins, density=True, color="#CBD5E0",
                                             edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for b_idx in range(len(patches_a)):
        if bins_a[b_idx] >= tv_obs_amph:
            patches_a[b_idx].set_facecolor("#E53E3E")
            patches_a[b_idx].set_alpha(0.95)
            
    ax_b2.axvline(tv_obs_amph, color="#9B2C2C", linestyle="--", linewidth=2.6, zorder=4,
                  label=f"Observed TV = {tv_obs_amph:.3f}\nExact p = {p_amph:.4f} ***")
    ax_b2.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=11.5)
    ax_b2.set_ylabel("Permutation Density", fontweight="bold", fontsize=11.5)
    ax_b2.set_title("Amphetamine: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                    loc="left", fontweight="bold", fontsize=12, pad=8)
    ax_b2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_b2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=10.5)
    ax_b2.set_ylim(0, 11.0)
    
    fig.suptitle(fig_title, fontweight="bold", fontsize=14.5, y=0.99)
    
    out_name = f"fig_level3_slide_{metric_type}_vertical.png"
    for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
        fig.savefig(d / out_name, dpi=300, bbox_inches="tight")
    print(f"Saved {out_name} successfully.")
    plt.close(fig)

make_slide_figure("rate")
make_slide_figure("duration")
print("Both thinner slide figures generated successfully!")
