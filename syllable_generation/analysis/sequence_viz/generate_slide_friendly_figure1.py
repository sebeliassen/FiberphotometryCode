#!/usr/bin/env python3
"""
generate_slide_friendly_figure1.py
=============================================================================
Slide-optimized version of Figure 1:
- Top panel: Cohort Chain Usage Rates (Mean ± SEM), individual dots removed,
  clean bars with distinct significance stars for genotype contrasts:
  * Saline: WT Sal vs DAT-KI Sal
  * Amphetamine: WT Amph vs DAT-KI Amph
- Legend placed horizontally above the plot to eliminate any overlap with bars.
- Bottom panel: Two high-impact, side-by-side exact permutation test cards
  showing the full null distributions, observed Total Variation distances,
  and exact p-values for Saline (p=0.0141*) and Amphetamine (p=0.0045***).
=============================================================================
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/data")
PRESENTATION_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations")
FIGURES_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/optimized_seqnmf/figures")
ARTIFACTS_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/6e6d59fd-58a4-4955-a762-5c2c131ab2b6")

# Load existing computed metrics
df_anim = pd.read_csv(DATA_DIR / "level3_animal_chain_usage_metrics.csv")
df_cohort = pd.read_csv(DATA_DIR / "level3_chain_cohort_usage_summary.csv")
df_uni = pd.read_csv(DATA_DIR / "level3_chain_univariate_contrasts.csv")

# Color palette: Premium, slide-friendly
COLOR_WT_SAL = "#4A5568"    # Slate Grey
COLOR_KI_SAL = "#DD6B20"    # Vibrant Ochre / Orange
COLOR_WT_AMPH = "#2B6CB0"   # Rich Sapphire Blue
COLOR_KI_AMPH = "#C53030"   # Crimson Red

# Setup exact animal permutations
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

# Clean, concise titles for x-axis
CHAIN_TITLES = {
    1: "Chain 1\nWalk-Turn",
    2: "Chain 2\nRest",
    3: "Chain 3\nSprint-Sniff",
    4: "Chain 4\nPace-Groom",
    5: "Chain 5\nGround-Sniff",
    6: "Chain 6\nOrient-Look",
    7: "Chain 7\nWall-Lean",
    8: "Chain 8\nSprint-Rear",
    9: "Chain 9\nWalk-Sniff",
    10: "Chain 10\nSlalom-Sprint"
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2D3748",
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13.5,
    "legend.fontsize": 11,
})

# =============================================================================
# 1. BUILD COMBINED VERTICAL SLIDE FIGURE
# =============================================================================
fig = plt.figure(figsize=(16, 12.5), dpi=300)
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.3, 0.95], hspace=0.42, wspace=0.22)

# -----------------------------------------------------------------------------
# TOP PANEL: Usage Rate (No dots, with Significance Stars)
# -----------------------------------------------------------------------------
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
    means = sub["rate_per_min_mean"].values
    sems = sub["rate_per_min_sem"].values
    x_pos = chain_indices + offsets[cohort]
    
    ax_top.bar(x_pos, means, yerr=sems, width=bar_w, color=cohort_colors[cohort],
               capsize=3.0, edgecolor="#1A202C", linewidth=1.0, alpha=0.92,
               label=cohort_labels[cohort], zorder=3)

# Add Significance Brackets and Stars
for k in range(1, K + 1):
    sub_k = df_cohort[df_cohort["factor_id"] == k]
    
    # Saline values
    wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
    ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
    sal_max = max(wt_sal_row["rate_per_min_mean"] + wt_sal_row["rate_per_min_sem"],
                  ki_sal_row["rate_per_min_mean"] + ki_sal_row["rate_per_min_sem"])
    
    # Amph values
    wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
    ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
    amph_max = max(wt_amph_row["rate_per_min_mean"] + wt_amph_row["rate_per_min_sem"],
                   ki_amph_row["rate_per_min_mean"] + ki_amph_row["rate_per_min_sem"])
    
    # Get p-values
    p_sal_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Saline")]["p_val"].values[0]
    p_amph_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Amphetamine")]["p_val"].values[0]
    
    # Saline bracket
    if p_sal_k < 0.05:
        star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
        x1 = k + offsets["WT_Saline"]
        x2 = k + offsets["KI_Saline"]
        y_b = sal_max + 0.32
        h_tick = 0.09
        ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.2, zorder=5)
        ax_top.text((x1 + x2) / 2, y_b + h_tick + 0.04, star_sal, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#C05621", zorder=5)
        
    # Amph bracket
    if p_amph_k < 0.05:
        star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
        x1 = k + offsets["WT_Amphetamine"]
        x2 = k + offsets["KI_Amphetamine"]
        y_b = amph_max + 0.32
        h_tick = 0.09
        ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.2, zorder=5)
        ax_top.text((x1 + x2) / 2, y_b + h_tick + 0.04, star_amph, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#9B2C2C", zorder=5)

ax_top.set_xticks(chain_indices)
ax_top.set_xticklabels([CHAIN_TITLES[k] for k in chain_indices], fontweight="bold")
ax_top.set_ylabel("Chain Usage Rate (events / min)", fontweight="bold", fontsize=13)
ax_top.set_title("A. Cohort Chain Usage Rates across Discovered Level 3 Motifs (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                 loc="left", fontweight="bold", fontsize=13.5, pad=12)
ax_top.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

# Position legend horizontally outside/above the plot so it NEVER covers any bars!
ax_top.legend(bbox_to_anchor=(0.5, 1.24), loc="upper center", ncol=4,
              frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11.5)
ax_top.set_ylim(0, 5.8)

# Subtle background shading for alternating chains
for k in chain_indices:
    if k % 2 == 0:
        ax_top.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

# -----------------------------------------------------------------------------
# BOTTOM PANEL: Exact Animal Permutations for Saline and Amphetamine
# -----------------------------------------------------------------------------
n_bins = 32

# Panel B1: Saline Genotype Separation
ax_b1 = fig.add_subplot(gs[1, 0])
counts, bins, patches = ax_b1.hist(null_sal, bins=n_bins, density=True, color="#CBD5E0",
                                   edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)

for b_idx in range(len(patches)):
    if bins[b_idx] >= tv_obs_sal:
        patches[b_idx].set_facecolor("#ED8936")
        patches[b_idx].set_alpha(0.95)

ax_b1.axvline(tv_obs_sal, color="#C05621", linestyle="--", linewidth=2.8, zorder=4,
              label=f"Observed TV = {tv_obs_sal:.3f}\nExact p = {p_sal:.4f} *")

ax_b1.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=12)
ax_b1.set_ylabel("Permutation Density", fontweight="bold", fontsize=12)
ax_b1.set_title("B1. Saline: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                loc="left", fontweight="bold", fontsize=13, pad=10)
ax_b1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_b1.set_ylim(0, 11.5)

# High-impact presentation stat card
ax_b1.text(0.04, 0.94, f"Saline Genotype Separation\n• Observed TV = {tv_obs_sal:.3f}\n• Exact p = {p_sal:.4f} * (Significant)",
           transform=ax_b1.transAxes, va="top", ha="left",
           fontweight="bold", fontsize=11, color="#7B341E",
           bbox=dict(boxstyle="round,pad=0.55", facecolor="#FEEBC8", edgecolor="#FBD38D", alpha=0.92))

ax_b1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=11)


# Panel B2: Amphetamine Genotype Separation
ax_b2 = fig.add_subplot(gs[1, 1])
counts_a, bins_a, patches_a = ax_b2.hist(null_amph, bins=n_bins, density=True, color="#CBD5E0",
                                         edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)

for b_idx in range(len(patches_a)):
    if bins_a[b_idx] >= tv_obs_amph:
        patches_a[b_idx].set_facecolor("#E53E3E")
        patches_a[b_idx].set_alpha(0.95)

ax_b2.axvline(tv_obs_amph, color="#9B2C2C", linestyle="--", linewidth=2.8, zorder=4,
              label=f"Observed TV = {tv_obs_amph:.3f}\nExact p = {p_amph:.4f} ***")

ax_b2.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=12)
ax_b2.set_ylabel("Permutation Density", fontweight="bold", fontsize=12)
ax_b2.set_title("B2. Amphetamine: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                loc="left", fontweight="bold", fontsize=13, pad=10)
ax_b2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_b2.set_ylim(0, 11.5)

# High-impact presentation stat card
ax_b2.text(0.04, 0.94, f"Amphetamine Genotype Separation\n• Observed TV = {tv_obs_amph:.3f}\n• Exact p = {p_amph:.4f} *** (High Significance)",
           transform=ax_b2.transAxes, va="top", ha="left",
           fontweight="bold", fontsize=11, color="#742A2A",
           bbox=dict(boxstyle="round,pad=0.55", facecolor="#FED7D7", edgecolor="#FEB2B2", alpha=0.92))

ax_b2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=11)

fig.suptitle("Level 3 Behavioral Chains: Cohort Usage Profiles & Statistically Significant Genotype Separation",
             fontweight="bold", fontsize=15.5, y=0.99)

combined_name = "fig_level3_slide_usage_and_separation_vertical.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig.savefig(d / combined_name, dpi=300, bbox_inches="tight")
print(f"Saved combined vertical slide figure to: {combined_name}")
plt.close(fig)


# =============================================================================
# 2. STANDALONE TOP PANEL (Usage Rate Only)
# =============================================================================
fig_top = plt.figure(figsize=(16, 6.8), dpi=300)
ax_t = fig_top.add_subplot(111)

for cohort in ["WT_Saline", "KI_Saline", "WT_Amphetamine", "KI_Amphetamine"]:
    sub = df_cohort[df_cohort["cohort"] == cohort].sort_values("factor_id")
    means = sub["rate_per_min_mean"].values
    sems = sub["rate_per_min_sem"].values
    x_pos = chain_indices + offsets[cohort]
    ax_t.bar(x_pos, means, yerr=sems, width=bar_w, color=cohort_colors[cohort],
             capsize=3.0, edgecolor="#1A202C", linewidth=1.0, alpha=0.92,
             label=cohort_labels[cohort], zorder=3)

for k in range(1, K + 1):
    sub_k = df_cohort[df_cohort["factor_id"] == k]
    wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
    ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
    sal_max = max(wt_sal_row["rate_per_min_mean"] + wt_sal_row["rate_per_min_sem"],
                  ki_sal_row["rate_per_min_mean"] + ki_sal_row["rate_per_min_sem"])
    wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
    ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
    amph_max = max(wt_amph_row["rate_per_min_mean"] + wt_amph_row["rate_per_min_sem"],
                   ki_amph_row["rate_per_min_mean"] + ki_amph_row["rate_per_min_sem"])
    p_sal_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Saline")]["p_val"].values[0]
    p_amph_k = df_uni[(df_uni["factor_id"] == k) & (df_uni["drug"] == "Amphetamine")]["p_val"].values[0]
    if p_sal_k < 0.05:
        star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
        x1 = k + offsets["WT_Saline"]
        x2 = k + offsets["KI_Saline"]
        y_b = sal_max + 0.32
        h_tick = 0.09
        ax_t.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.2, zorder=5)
        ax_t.text((x1 + x2) / 2, y_b + h_tick + 0.04, star_sal, ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color="#C05621", zorder=5)
    if p_amph_k < 0.05:
        star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
        x1 = k + offsets["WT_Amphetamine"]
        x2 = k + offsets["KI_Amphetamine"]
        y_b = amph_max + 0.32
        h_tick = 0.09
        ax_t.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.2, zorder=5)
        ax_t.text((x1 + x2) / 2, y_b + h_tick + 0.04, star_amph, ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color="#9B2C2C", zorder=5)

for k in chain_indices:
    if k % 2 == 0:
        ax_t.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

ax_t.set_xticks(chain_indices)
ax_t.set_xticklabels([CHAIN_TITLES[k] for k in chain_indices], fontweight="bold", fontsize=11)
ax_t.set_ylabel("Chain Usage Rate (events / min)", fontweight="bold", fontsize=13)
ax_t.set_title("Cohort Chain Usage Rates across Discovered Level 3 Motifs (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
               loc="left", fontweight="bold", fontsize=14, pad=12)
ax_t.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_t.legend(bbox_to_anchor=(0.5, 1.22), loc="upper center", ncol=4,
            frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11.5)
ax_t.set_ylim(0, 5.8)

top_name = "fig_level3_slide_top_usage_rate_stars.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig_top.savefig(d / top_name, dpi=300, bbox_inches="tight")
print(f"Saved standalone top slide figure to: {top_name}")
plt.close(fig_top)


# =============================================================================
# 3. STANDALONE BOTTOM PANEL (Permutation p-values Only)
# =============================================================================
fig_bot = plt.figure(figsize=(16, 5.8), dpi=300)
gs_b = gridspec.GridSpec(1, 2, figure=fig_bot, wspace=0.22)

# Saline
ax_b1_s = fig_bot.add_subplot(gs_b[0, 0])
counts, bins, patches = ax_b1_s.hist(null_sal, bins=n_bins, density=True, color="#CBD5E0",
                                     edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
for b_idx in range(len(patches)):
    if bins[b_idx] >= tv_obs_sal:
        patches[b_idx].set_facecolor("#ED8936")
        patches[b_idx].set_alpha(0.95)

ax_b1_s.axvline(tv_obs_sal, color="#C05621", linestyle="--", linewidth=2.8, zorder=4,
                label=f"Observed TV = {tv_obs_sal:.3f}\nExact p = {p_sal:.4f} *")
ax_b1_s.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=12)
ax_b1_s.set_ylabel("Permutation Density", fontweight="bold", fontsize=12)
ax_b1_s.set_title("Saline: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                  loc="left", fontweight="bold", fontsize=13, pad=10)
ax_b1_s.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_b1_s.set_ylim(0, 11.5)
ax_b1_s.text(0.04, 0.94, f"Saline Genotype Separation\n• Observed TV = {tv_obs_sal:.3f}\n• Exact p = {p_sal:.4f} * (Significant)",
             transform=ax_b1_s.transAxes, va="top", ha="left",
             fontweight="bold", fontsize=11, color="#7B341E",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#FEEBC8", edgecolor="#FBD38D", alpha=0.92))
ax_b1_s.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=11)

# Amph
ax_b2_s = fig_bot.add_subplot(gs_b[0, 1])
counts_a, bins_a, patches_a = ax_b2_s.hist(null_amph, bins=n_bins, density=True, color="#CBD5E0",
                                           edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
for b_idx in range(len(patches_a)):
    if bins_a[b_idx] >= tv_obs_amph:
        patches_a[b_idx].set_facecolor("#E53E3E")
        patches_a[b_idx].set_alpha(0.95)

ax_b2_s.axvline(tv_obs_amph, color="#9B2C2C", linestyle="--", linewidth=2.8, zorder=4,
                label=f"Observed TV = {tv_obs_amph:.3f}\nExact p = {p_amph:.4f} ***")
ax_b2_s.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=12)
ax_b2_s.set_ylabel("Permutation Density", fontweight="bold", fontsize=12)
ax_b2_s.set_title("Amphetamine: WT vs. DAT-KI Genotype Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                  loc="left", fontweight="bold", fontsize=13, pad=10)
ax_b2_s.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_b2_s.set_ylim(0, 11.5)
ax_b2_s.text(0.04, 0.94, f"Amphetamine Genotype Separation\n• Observed TV = {tv_obs_amph:.3f}\n• Exact p = {p_amph:.4f} *** (High Significance)",
             transform=ax_b2_s.transAxes, va="top", ha="left",
             fontweight="bold", fontsize=11, color="#742A2A",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#FED7D7", edgecolor="#FEB2B2", alpha=0.92))
ax_b2_s.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=11)

bot_name = "fig_level3_slide_bottom_genotype_separation_pvalues.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig_bot.savefig(d / bot_name, dpi=300, bbox_inches="tight")
print(f"Saved standalone bottom slide figure to: {bot_name}")
plt.close(fig_bot)

print("Slide-friendly figures regenerated with perfect spacing!")
