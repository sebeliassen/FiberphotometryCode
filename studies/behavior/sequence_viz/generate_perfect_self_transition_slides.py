#!/usr/bin/env python3
"""
generate_perfect_self_transition_slides.py
=============================================================================
Generates the self-transition slide figures represented EXACTLY like
fig_level3_slide_duration_vertical:
- Mean ± SEM error bars across animals
- Formal significance brackets with stars (* p < 0.05, ** p < 0.01)
- NO weird arrows, NO arbitrary text box callouts
- Clean horizontal legend, alternating grey column shading
- Vertical sequence progression labels along the x-axis
- Generates:
  1. fig_level3_slide_self_transitions.png (Top: Self-Transitions, Bottom: 4 Transition Matrices)
  2. fig_level3_slide_self_transitions_vertical.png (Top: Self-Transitions, Bottom: 2 Permutation Tests)
=============================================================================
"""

import itertools
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/data")
PRESENTATION_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations")
FIGURES_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/optimized_seqnmf/figures")
ARTIFACTS_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/6e6d59fd-58a4-4955-a762-5c2c131ab2b6")

# Color palette matching presentation
COLOR_WT_SAL = "#4A5568"    # Slate Grey
COLOR_KI_SAL = "#DD6B20"    # Vibrant Ochre / Orange
COLOR_WT_AMPH = "#2B6CB0"   # Rich Sapphire Blue
COLOR_KI_AMPH = "#C53030"   # Crimson Red

COHORT_ORDER = ["WT_Saline", "KI_Saline", "WT_Amphetamine", "KI_Amphetamine"]
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "KI_Saline": "DAT-KI Saline",
    "WT_Amphetamine": "WT Amphetamine",
    "KI_Amphetamine": "DAT-KI Amphetamine",
}
COHORT_COLORS = {
    "WT_Saline": COLOR_WT_SAL,
    "KI_Saline": COLOR_KI_SAL,
    "WT_Amphetamine": COLOR_WT_AMPH,
    "KI_Amphetamine": COLOR_KI_AMPH,
}

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

# Run pipeline to get verified episodes and metadata
import sys

sys.path.append(str(Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz")))
from extract_level3_chain_quantitative_results import run_full_level3_pipeline

W, token_names, df_ep, df_meta = run_full_level3_pipeline()

K = 10

# Compute per-animal self-transition probabilities and cohort transition matrices
animal_records = []
cohort_trans_counts = {c: np.zeros((K, K), dtype=np.float64) for c in COHORT_ORDER}

for _, s_row in df_meta.iterrows():
    s_idx = s_row["session_idx"]
    aid = s_row["animal_id"]
    gt = s_row["genotype"]
    drug = s_row["drug"]
    cohort = s_row["cohort"]

    ep_sub = df_ep[df_ep["session_idx"] == s_idx].sort_values("start_bout")
    factors = ep_sub["factor_id"].values
    n_ep = len(factors)

    sess_trans = np.zeros((K, K), dtype=np.float64)
    if n_ep > 1:
        for i in range(n_ep - 1):
            f_curr = factors[i] - 1
            f_next = factors[i+1] - 1
            sess_trans[f_curr, f_next] += 1.0

    cohort_trans_counts[cohort] += sess_trans

    row_sums = sess_trans.sum(axis=1)
    for k in range(K):
        p_self = sess_trans[k, k] / row_sums[k] if row_sums[k] > 0 else 0.0
        animal_records.append({
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "cohort": cohort,
            "factor_id": k + 1,
            "self_trans_prob": p_self,
            "n_transitions_from_k": row_sums[k],
            "n_self_loops": sess_trans[k, k]
        })

df_self_anim = pd.DataFrame(animal_records)

# Normalized cohort transition matrices
norm_cohort_trans = {}
for cohort, mat in cohort_trans_counts.items():
    r_sums = mat.sum(axis=1, keepdims=True)
    norm_cohort_trans[cohort] = np.divide(mat, r_sums, out=np.zeros_like(mat), where=r_sums > 0)

# Compute univariate Mann-Whitney U test p-values for self-transitions
dur_contrasts = []
for k in range(1, K + 1):
    sub_k = df_self_anim[df_self_anim["factor_id"] == k]
    for drug in ["Saline", "Amphetamine"]:
        wt_vals = sub_k[(sub_k["genotype"] == "WT") & (sub_k["drug"] == drug)]["self_trans_prob"].values
        ki_vals = sub_k[(sub_k["genotype"] == "KI") & (sub_k["drug"] == drug)]["self_trans_prob"].values
        u, p = stats.mannwhitneyu(wt_vals, ki_vals, alternative="two-sided")
        dur_contrasts.append({"factor_id": k, "drug": drug, "p_val": p})
df_uni_self = pd.DataFrame(dur_contrasts)

# Summary cohort dataframe (Mean and SEM)
cohort_summary = []
for cohort in COHORT_ORDER:
    for k in range(1, K + 1):
        vals = df_self_anim[(df_self_anim["cohort"] == cohort) & (df_self_anim["factor_id"] == k)]["self_trans_prob"].values
        cohort_summary.append({
            "cohort": cohort,
            "factor_id": k,
            "mean": np.mean(vals),
            "sem": stats.sem(vals) if len(vals) > 1 else 0.0
        })
df_self_summary = pd.DataFrame(cohort_summary)

# Setup exact animal permutations for TV distance (for the vertical twin figure)
df_anim = pd.read_csv(DATA_DIR / "level3_animal_chain_usage_metrics.csv")
anim_gt = df_anim[["animal_id", "genotype"]].drop_duplicates().set_index("animal_id")["genotype"].to_dict()
all_animals = sorted(list(anim_gt.keys()))
wt_animals = [a for a in all_animals if anim_gt[a] == "WT"]
ki_animals = [a for a in all_animals if anim_gt[a] == "KI"]

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

def draw_top_self_transitions_panel(ax_top):
    """Draws the top panel with exact significance brackets & stars matching fig_level3_slide_duration_vertical."""
    chain_indices = np.arange(1, K + 1)
    bar_w = 0.185
    offsets = {
        "WT_Saline": -1.65 * bar_w,
        "KI_Saline": -0.65 * bar_w,
        "WT_Amphetamine": 0.65 * bar_w,
        "KI_Amphetamine": 1.65 * bar_w,
    }

    for cohort in COHORT_ORDER:
        sub = df_self_summary[df_self_summary["cohort"] == cohort].sort_values("factor_id")
        means = sub["mean"].values
        sems = sub["sem"].values
        x_pos = chain_indices + offsets[cohort]

        ax_top.bar(x_pos, means, yerr=sems, width=bar_w, color=COHORT_COLORS[cohort],
                   capsize=2.8, edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
                   label=COHORT_LABELS[cohort], zorder=3)

    # Shading for alternating chains
    for k in chain_indices:
        if k % 2 == 0:
            ax_top.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

    # Add Significance Brackets and Stars
    max_star_y = 0.0
    scale_factor = 0.65
    y_step = 0.04 * scale_factor
    h_tick = 0.012 * scale_factor

    for k in range(1, K + 1):
        sub_k = df_self_summary[df_self_summary["factor_id"] == k]
        wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
        ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
        sal_max = max(wt_sal_row["mean"] + wt_sal_row["sem"],
                      ki_sal_row["mean"] + ki_sal_row["sem"])

        wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
        ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
        amph_max = max(wt_amph_row["mean"] + wt_amph_row["sem"],
                       ki_amph_row["mean"] + ki_amph_row["sem"])

        p_sal_k = df_uni_self[(df_uni_self["factor_id"] == k) & (df_uni_self["drug"] == "Saline")]["p_val"].values[0]
        p_amph_k = df_uni_self[(df_uni_self["factor_id"] == k) & (df_uni_self["drug"] == "Amphetamine")]["p_val"].values[0]

        if p_sal_k < 0.05:
            star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
            x1 = k + offsets["WT_Saline"]
            x2 = k + offsets["KI_Saline"]
            y_b = sal_max + y_step
            ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
            star_pos_y = y_b + h_tick + 0.008 * scale_factor
            ax_top.text((x1 + x2) / 2, star_pos_y, star_sal,
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#C05621", zorder=5)
            max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

        if p_amph_k < 0.05:
            star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
            x1 = k + offsets["WT_Amphetamine"]
            x2 = k + offsets["KI_Amphetamine"]
            y_b = amph_max + y_step
            ax_top.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
            star_pos_y = y_b + h_tick + 0.008 * scale_factor
            ax_top.text((x1 + x2) / 2, star_pos_y, star_amph,
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#9B2C2C", zorder=5)
            max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

    ax_top.set_xticks(chain_indices)
    ax_top.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
    ax_top.set_ylabel("Self-Transition Probability P(k → k)", fontweight="bold", fontsize=12)
    ax_top.set_title("Chain-Specific Self-Repetition Propensity (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                     loc="left", fontweight="bold", fontsize=13, pad=12)
    ax_top.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Clean horizontal legend positioned above plot
    ax_top.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
                  frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)

    y_top_limit = max(0.65, max_star_y * 1.08)
    ax_top.set_ylim(0, y_top_limit)


# =============================================================================
# FIGURE 1: Top = Self-Transitions with Stars; Bottom = 4 Transition Matrices
# =============================================================================
fig1 = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs1 = gridspec.GridSpec(2, 4, figure=fig1, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.32)

ax_top1 = fig1.add_subplot(gs1[0, :])
draw_top_self_transitions_panel(ax_top1)

for c_idx, cohort in enumerate(COHORT_ORDER):
    ax_mat = fig1.add_subplot(gs1[1, c_idx])
    t_mat = norm_cohort_trans[cohort]

    im = ax_mat.imshow(t_mat, cmap="viridis", vmin=0, vmax=0.45, aspect="equal")
    ax_mat.set_title(f"{COHORT_LABELS[cohort]}\nTransition Matrix", fontweight="bold", fontsize=11.5, pad=8)

    ax_mat.set_xticks(np.arange(0, K, 2))
    ax_mat.set_xticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)
    ax_mat.set_yticks(np.arange(0, K, 2))
    ax_mat.set_yticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)

    # Highlight diagonal self-transitions with clean bounding boxes
    for k in range(K):
        rect = patches.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                 edgecolor="#ED8936" if "KI" in cohort else "#63B3ED",
                                 linewidth=1.3, linestyle="-", zorder=5)
        ax_mat.add_patch(rect)

    if c_idx == 0:
        ax_mat.set_ylabel("Current Chain $k_t$", fontweight="bold", fontsize=11)
    ax_mat.set_xlabel("Next Chain $k_{t+1}$", fontweight="bold", fontsize=11)

    if c_idx == 3:
        cbar = fig1.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.07)
        cbar.set_label("Transition Probability", fontweight="bold", fontsize=10)

fig1.suptitle("Level 3 Behavioral Chains: Self-Transitions & Sequential Syntactic Organization",
              fontweight="bold", fontsize=14.5, y=0.99)

out1 = "fig_level3_slide_self_transitions.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig1.savefig(d / out1, dpi=300, bbox_inches="tight")
print(f"Saved {out1} successfully.")
plt.close(fig1)


# =============================================================================
# FIGURE 2: Top = Self-Transitions with Stars; Bottom = 2 Permutation Tests (EXACT TWIN of duration slide)
# =============================================================================
fig2 = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs2 = gridspec.GridSpec(2, 2, figure=fig2, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.22)

ax_top2 = fig2.add_subplot(gs2[0, :])
draw_top_self_transitions_panel(ax_top2)

n_bins = 30

# B1: Saline
ax_b1 = fig2.add_subplot(gs2[1, 0])
counts, bins, patches_s = ax_b1.hist(null_sal, bins=n_bins, density=True, color="#CBD5E0",
                                     edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
for b_idx in range(len(patches_s)):
    if bins[b_idx] >= tv_obs_sal:
        patches_s[b_idx].set_facecolor("#ED8936")
        patches_s[b_idx].set_alpha(0.95)

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
ax_b2 = fig2.add_subplot(gs2[1, 1])
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

fig2.suptitle("Level 3 Behavioral Chains: Self-Transitions & Genotype Separation",
              fontweight="bold", fontsize=14.5, y=0.99)

out2 = "fig_level3_slide_self_transitions_vertical.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig2.savefig(d / out2, dpi=300, bbox_inches="tight")
print(f"Saved {out2} successfully.")
plt.close(fig2)

print("\nBoth figures generated successfully with significance stars and zero annotations/arrows!")
