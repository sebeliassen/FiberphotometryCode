#!/usr/bin/env python3
"""
generate_slide_friendly_self_transitions.py
=============================================================================
Generates a dedicated, slide-ready figure for Chain Self-Transitions & Perseveration:
1. Top Panel: Chain-Specific Self-Transition Probability P(k -> k) across all 10 chains,
   featuring full vertical syllable labels (e.g. Chain 10: Sprint -> Turn_L -> Sprint).
2. Bottom Panels: 4 Cohort Transition Matrices (WT Saline, DAT-KI Saline, WT Amph, DAT-KI Amph)
   with diagonal self-transition cells highlighted with high-contrast outlines.
3. Completely clean presentation titles (no "A.", "B1.", "B2." academic lettering).
=============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

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

# Recompute transition matrices from raw episodes & session metadata
K = 10
df_ep = pd.read_csv(DATA_DIR / "level3_animal_chain_usage_metrics.csv")
# To get exact consecutive sequence transitions, load raw episodes if available, or compute from session transitions
# Let's check if we can compute from raw sequence dataset or level3_chain_looping_perseveration_summary
import pickle

pkl_path = DATA_DIR / "unified_sequence_dataset.pkl"
has_pkl = pkl_path.exists()

cohort_trans_matrices = {c: np.zeros((K, K), dtype=np.float64) for c in COHORT_ORDER}

# Let's inspect how extract_level3_chain_quantitative_results did it
# It used df_ep from decoded episodes
ep_file = DATA_DIR / "unique_behavioral_episodes_database.csv"
if ep_file.exists():
    df_ep_raw = pd.read_csv(ep_file)
    meta_file = DATA_DIR / "session_metadata_ground_truth.csv"
    df_meta = pd.read_csv(meta_file) if meta_file.exists() else None
    
    # Check if level3_seqnmf_model_results exists
    model_pkl = DATA_DIR / "level3_seqnmf_model_results.pkl"
    with open(model_pkl, "rb") as f:
        res = pickle.load(f)
    # The saved model has W, H
    # Let's load the decoded episodes from extract_level3 script
    # In extract_level3, norm_cohort_trans was computed
    pass

# We can directly run the exact transition calculation from extract_level3
from extract_level3_chain_quantitative_results import (
    run_full_level3_pipeline,
    analyze_chain_looping_and_perseveration
)

W, token_names, df_ep, df_meta = run_full_level3_pipeline()
df_loop, norm_cohort_trans = analyze_chain_looping_and_perseveration(df_ep, df_meta, K=10)

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

fig = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.32)

# -------------------------------------------------------------------------
# TOP PANEL: CHAIN-SPECIFIC SELF-TRANSITION PROBABILITY
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

for c_idx, cohort in enumerate(COHORT_ORDER):
    diag_probs = np.diag(norm_cohort_trans[cohort])
    x_pos = chain_indices + offsets[cohort]
    ax_top.bar(x_pos, diag_probs, width=bar_w, color=COHORT_COLORS[cohort],
               edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
               label=COHORT_LABELS[cohort], zorder=3)

# Shading for alternating chains
for k in chain_indices:
    if k % 2 == 0:
        ax_top.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

# Add highlight callout for Chain 10 (Sprint Slalom Stereotypy)
c10_ki_amph = np.diag(norm_cohort_trans["KI_Amphetamine"])[9]
c10_wt_amph = np.diag(norm_cohort_trans["WT_Amphetamine"])[9]
ax_top.annotate(
    f"Chain 10 Looping:\nDAT-KI: {c10_ki_amph*100:.1f}%\nWT: {c10_wt_amph*100:.1f}%\n(>2× Elevation)",
    xy=(10 + offsets["KI_Amphetamine"], c10_ki_amph),
    xytext=(8.3, 0.45),
    arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.6),
    fontweight="bold", fontsize=10, color="#9B2C2C",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF5F5", edgecolor="#FEB2B2", lw=1.2)
)

ax_top.set_xticks(chain_indices)
ax_top.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
ax_top.set_ylabel("Self-Transition Probability P(k → k)", fontweight="bold", fontsize=12)
ax_top.set_title("Chain-Specific Self-Repetition Propensity (Back-to-Back Looping P(k → k))\nMeasuring Behavioral Perseveration and Stereotypy Across Cohorts",
                 loc="left", fontweight="bold", fontsize=13, pad=12)
ax_top.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_top.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
              frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)
ax_top.set_ylim(0, 0.68)

# -------------------------------------------------------------------------
# BOTTOM PANELS: 4 COHORT TRANSITION MATRICES WITH HIGHLIGHTED DIAGONALS
# -------------------------------------------------------------------------
for c_idx, cohort in enumerate(COHORT_ORDER):
    ax_mat = fig.add_subplot(gs[1, c_idx])
    t_mat = norm_cohort_trans[cohort]
    
    im = ax_mat.imshow(t_mat, cmap="viridis", vmin=0, vmax=0.45, aspect="equal")
    ax_mat.set_title(f"{COHORT_LABELS[cohort]}\nTransition Matrix", fontweight="bold", fontsize=11.5, pad=8)
    
    ax_mat.set_xticks(np.arange(0, K, 2))
    ax_mat.set_xticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)
    ax_mat.set_yticks(np.arange(0, K, 2))
    ax_mat.set_yticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)
    
    # Highlight diagonal self-transitions with clean orange/white bounding boxes
    for k in range(K):
        rect = patches.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                 edgecolor="#ED8936" if "KI" in cohort else "#63B3ED",
                                 linewidth=1.4, linestyle="-", zorder=5)
        ax_mat.add_patch(rect)
        
    if c_idx == 0:
        ax_mat.set_ylabel("Current Chain $k_t$", fontweight="bold", fontsize=11)
    ax_mat.set_xlabel("Next Chain $k_{t+1}$", fontweight="bold", fontsize=11)
    
    if c_idx == 3:
        cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.07)
        cbar.set_label("Transition Probability", fontweight="bold", fontsize=10)

fig.suptitle("Level 3 Behavioral Chains: Self-Transitions & Sequential Syntactic Organization",
             fontweight="bold", fontsize=14.5, y=0.99)

out_name = "fig_level3_slide_self_transitions.png"
for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    fig.savefig(d / out_name, dpi=300, bbox_inches="tight")
print(f"Saved {out_name} successfully to all target directories.")
plt.close(fig)
