#!/usr/bin/env python3
"""
generate_hidden_slide_mann_whitney_plot.py
=============================================================================
Generates publication-grade Mann-Whitney U plots for the hidden Q&A backup slides
across experimental cohorts (WT Sal, WT Amph, KI Sal, KI Amph):
1. Level 1 (Exact Video Chains, 100 syllables)
2. Level 2 (Conservative Synonyms Merged, 93 syllables)
3. Level 3 (Functional Families Model, 22 states)

Titles are formatted as:
   Theme X
   *Step 1 -> Step 2 -> Step 3* (in italic below, no extra prose labels)
=============================================================================
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_level_definitions,
    get_verified_metadata,
)

ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/6e6d59fd-58a4-4955-a762-5c2c131ab2b6")
SEQ_VIZ_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz"

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
})

COLOR_WT_SAL = "#4A5568"    # Slate grey
COLOR_WT_AMPH = "#3182CE"   # Rich blue
COLOR_KI_SAL = "#ED8936"    # Warm orange
COLOR_KI_AMPH = "#E53E3E"   # Deep crimson

COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
COHORT_LABELS = ["WT\nSaline", "WT\nAmph", "DAT-KI\nSaline", "DAT-KI\nAmph"]
COHORT_COLORS = [COLOR_WT_SAL, COLOR_WT_AMPH, COLOR_KI_SAL, COLOR_KI_AMPH]


def format_p(p):
    if p < 0.001: return f"p = {p:.1e} ***"
    if p < 0.01:  return f"p = {p:.3f} **"
    if p < 0.05:  return f"p = {p:.3f} *"
    return f"p = {p:.2f} (n.s.)"


def plot_three_themes(data_dict, subtitles, title, out_paths, y_label="Usage Rate (events / min)"):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), dpi=300)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.15)

    themes = ["Theme A", "Theme B", "Theme C"]

    for ax_idx, theme_key in enumerate(themes):
        ax = axes[ax_idx]
        cohort_vals = data_dict[theme_key]  # list of 4 arrays: [wt_sal, wt_amph, ki_sal, ki_amph]

        means = [np.mean(v) for v in cohort_vals]
        sems = [stats.sem(v) if len(v) > 1 else 0.0 for v in cohort_vals]
        x_pos = np.arange(len(COHORT_ORDER))

        # Bars
        ax.bar(x_pos, means, yerr=sems, capsize=4, width=0.55,
               color=COHORT_COLORS, alpha=0.85, edgecolor="#222222", linewidth=1.2, zorder=2)

        # Scatter individual sessions
        rng = np.random.RandomState(42 + ax_idx)
        for i, vals in enumerate(cohort_vals):
            jitter = rng.uniform(-0.11, 0.11, size=len(vals))
            ax.scatter(x_pos[i] + jitter, vals, color="#111111", edgecolor="#FFFFFF",
                       s=45, linewidth=0.8, alpha=0.9, zorder=3)

        # Mann-Whitney stats
        wt_sal, wt_amph, ki_sal, ki_amph = cohort_vals
        u_sal, p_sal = stats.mannwhitneyu(wt_sal, ki_sal, alternative="two-sided")
        u_amph, p_amph = stats.mannwhitneyu(wt_amph, ki_amph, alternative="two-sided")

        # Limits & layout
        max_val = max(np.max(np.concatenate(cohort_vals)), 0.05)
        ax.set_ylim(0, max_val * 1.35)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(COHORT_LABELS, fontweight="semibold")
        ax.set_ylabel(y_label, fontweight="semibold")

        # Header: "Theme X" bold, and italic chain on the line below
        chain_sub = subtitles[theme_key]
        ax.set_title(theme_key, fontsize=13.5, fontweight="bold", pad=20)
        ax.text(0.5, 1.022, f"{chain_sub}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10.0, fontstyle="italic", color="#2D3748")

        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)

        # Significance brackets
        h1 = max_val * 1.05
        ax.plot([0, 0, 2, 2], [h1, h1 * 1.025, h1 * 1.025, h1], color="#333333", lw=1.2)
        ax.text(1.0, h1 * 1.035, f"Saline: {format_p(p_sal)}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold" if p_sal < 0.05 else "normal",
                color="#C53030" if p_sal < 0.05 else "#4A5568")

        h2 = max_val * 1.18
        ax.plot([1, 1, 3, 3], [h2, h2 * 1.025, h2 * 1.025, h2], color="#333333", lw=1.2)
        ax.text(2.0, h2 * 1.035, f"Amph: {format_p(p_amph)}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold" if p_amph < 0.05 else "normal",
                color="#C53030" if p_amph < 0.05 else "#4A5568")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    for p in out_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {p}")
    plt.close()


def generate_level1():
    print("--- Computing Level 1 (Exact Video Chains) ---")
    res_h5 = h5py.File(REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/results.h5", "r")
    keys = list(res_h5.keys())

    chains = {
        "Theme A": [15, 14, 4, 3],
        "Theme B": [2, 0, 13, 2],
        "Theme C": [8, 32, 8, 32]
    }

    subtitles = {
        "Theme A": "Sniff_Up -> Rear -> Desc_L -> Walk_L",
        "Theme B": "Turn_R -> Run_R -> Dash -> Turn_R",
        "Theme C": "Sniff -> Rest -> Sniff -> Rest"
    }

    records = {t: {c: [] for c in COHORT_ORDER} for t in chains}

    for k in keys:
        sylls = res_h5[k]["syllable"][:]
        mask = np.concatenate([[True], sylls[1:] != sylls[:-1]])
        c_seq = sylls[mask]
        t_min = len(sylls) / (40.0 * 60.0)
        _, geno, drug = get_verified_metadata(k)
        cohort = f"{geno}_{drug}"

        for theme, target in chains.items():
            n = len(target)
            cnt = sum(1 for i in range(len(c_seq) - n + 1) if list(c_seq[i : i + n]) == target)
            records[theme][cohort].append(cnt / t_min)

    data_dict = {
        theme: [np.array(records[theme][c]) for c in COHORT_ORDER]
        for theme in chains
    }

    title = "Video Sequence Usage Across Cohorts (Level 1 Model, N=30 Sessions)"
    out_paths = [
        SEQ_VIZ_DIR / "fig_hidden_slide_video_chains_mann_whitney.png",
        ARTIFACT_DIR / "fig_hidden_slide_video_chains_mann_whitney.png"
    ]
    plot_three_themes(data_dict, subtitles, title, out_paths)


def generate_level2():
    print("--- Computing Level 2 (Conservative Synonym Merging) ---")
    levels = get_level_definitions()
    l2 = levels[1]
    m2 = l2['map_arr']

    res_h5 = h5py.File(REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/results.h5", "r")
    keys = list(res_h5.keys())

    chains_l1 = {
        "Theme A": [15, 14, 4, 3],
        "Theme B": [2, 0, 13, 2],
        "Theme C": [8, 32, 8, 32]
    }

    chains_l2 = {
        theme: [int(m2[s]) for s in seq]
        for theme, seq in chains_l1.items()
    }

    subtitles = {
        "Theme A": "Sniff_Up -> Rear -> Desc_L -> Walk_L",
        "Theme B": "Turn_R -> Run_R -> Dash -> Turn_R",
        "Theme C": "Sniff -> Rest -> Sniff -> Rest"
    }

    records = {t: {c: [] for c in COHORT_ORDER} for t in chains_l2}

    for k in keys:
        sylls = res_h5[k]["syllable"][:]
        mask = np.concatenate([[True], sylls[1:] != sylls[:-1]])
        c_seq1 = sylls[mask]
        c_seq2 = m2[c_seq1]
        mask2 = np.concatenate([[True], c_seq2[1:] != c_seq2[:-1]])
        c_seq2 = c_seq2[mask2]
        t_min = len(sylls) / (40.0 * 60.0)
        _, geno, drug = get_verified_metadata(k)
        cohort = f"{geno}_{drug}"

        for theme, target in chains_l2.items():
            n = len(target)
            cnt = sum(1 for i in range(len(c_seq2) - n + 1) if list(c_seq2[i : i + n]) == target)
            records[theme][cohort].append(cnt / t_min)

    data_dict = {
        theme: [np.array(records[theme][c]) for c in COHORT_ORDER]
        for theme in chains_l2
    }

    title = "Conservative Sequence Usage Across Cohorts (Level 2 Model, N=30 Sessions)"
    out_paths = [
        SEQ_VIZ_DIR / "fig_hidden_slide_level2_mann_whitney.png",
        ARTIFACT_DIR / "fig_hidden_slide_level2_mann_whitney.png"
    ]
    plot_three_themes(data_dict, subtitles, title, out_paths)


def generate_level3():
    print("--- Computing Level 3 (Functional Families Model) ---")
    df = pd.read_csv(REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/level3_animal_chain_usage_metrics.csv")

    factor_map = {
        "Theme A": 7,   # Wall_Lean -> Descent_L -> Walk_L
        "Theme B": 10,  # Sprint -> Turn_L -> Sprint
        "Theme C": 2    # Sniff_Ground -> Descent_R -> Rest
    }

    subtitles = {
        "Theme A": "Wall_Lean -> Descent_L -> Walk_L",
        "Theme B": "Sprint -> Turn_L -> Sprint",
        "Theme C": "Sniff_Ground -> Descent_R -> Rest"
    }

    data_dict = {}
    for theme, fid in factor_map.items():
        sub = df[df["factor_id"] == fid]
        cohort_vals = [
            sub[sub["cohort"] == c]["rate_per_min"].values
            for c in COHORT_ORDER
        ]
        data_dict[theme] = cohort_vals

    title = "Functional Sequence Usage Across Cohorts (Level 3 Model, N=30 Sessions)"
    out_paths = [
        SEQ_VIZ_DIR / "fig_hidden_slide_level3_mann_whitney.png",
        ARTIFACT_DIR / "fig_hidden_slide_level3_mann_whitney.png"
    ]
    plot_three_themes(data_dict, subtitles, title, out_paths)


def main():
    generate_level1()
    generate_level2()
    generate_level3()
    print("All hidden slide Mann-Whitney plots generated successfully.")


if __name__ == "__main__":
    main()
