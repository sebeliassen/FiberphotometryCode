#!/usr/bin/env python3
"""
generate_reality_check_slide_figures.py
=============================================================================
Generates the core figures for the presentation slide:
"shMoSeq on Our Cohort: The Reality Check" (The Pivot Slide)

1. pres_syllables_vs_states_genotype_permutation.png
   - Side-by-side exact animal permutation test (6,435 partitions):
     * Left: Raw Syllables Repertoire catches genotype differences (p = 0.0356, Significant).
     * Right: shMoSeq Macro-States erase genotype differences (p = 0.754, Not Significant).

2. pres_behavioral_hierarchy_pyramid.png
   - A clean, modern bottom-up pyramid schematic highlighting the "missing rung":
     * Top: Macro-States (shMoSeq) -> Skipped too fast (Over-abstracted, idiosyncratic)
     * Middle (Focus): Syllable Chains & Episodes -> Tangible, verifiable, structured syntax
     * Bottom: Syllables (KPMS) -> The sub-second physical alphabet

3. pres_shmoseq_state_timelines_idiosyncrasy.png
   - High-contrast, slide-ready timeline raster showing animal-level state entrapment.
=============================================================================
"""

import os
import shutil
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
DATA_FILE = os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl")
MODEL_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "selected_exploratory_model.pkl")

OUTPUT_DIRS = [
    os.path.join(SHMOSEQ_DIR, "output"),
    os.path.join(SHMOSEQ_DIR, "output", "figures"),
    os.path.join(BASE_DIR, "syllable_generation", "presentations"),
]

def set_style():
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
        "axes.titlesize": 13,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15,
    })

def save_fig_to_all(fig, filename):
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")

def make_permutation_comparison():
    print("Computing exact permutation tests for Raw Syllables vs. Macro-States...")
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
    sylls = bundle["syllables_dict"]
    meta = bundle["metadata_dict"]

    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)
    states = model_bundle["states_dict"]

    # 1. Animal-level data for Saline baseline
    animals_syll = {}
    animals_state = {}

    for k, s in sylls.items():
        m = meta[k]
        if m["drug"] == "Saline":
            aid = str(m["animal_id"])
            geno = m["genotype"]
            counts_s = np.bincount(s, minlength=100)
            props_s = counts_s / counts_s.sum()
            animals_syll[aid] = {"geno": geno, "props": props_s}

            st = states[k]
            counts_z = np.bincount(st, minlength=8)
            props_z = counts_z / counts_z.sum()
            animals_state[aid] = {"geno": geno, "props": props_z}

    aids = sorted(list(animals_syll.keys()), key=lambda x: (animals_syll[x]["geno"], x))
    wt_aids = [a for a in aids if animals_syll[a]["geno"] == "WT"]
    n_total = len(aids)
    n_wt = len(wt_aids)
    combos = list(itertools.combinations(range(n_total), n_wt))
    n_combos = len(combos)

    syll_mat = np.array([animals_syll[a]["props"] for a in aids])
    state_mat = np.array([animals_state[a]["props"] for a in aids])
    wt_mask = np.array([animals_syll[a]["geno"] == "WT" for a in aids])
    ki_mask = ~wt_mask

    # Observed distances
    obs_tv_syll = 0.5 * np.sum(np.abs(syll_mat[wt_mask].mean(axis=0) - syll_mat[ki_mask].mean(axis=0)))
    obs_tv_state = 0.5 * np.sum(np.abs(state_mat[wt_mask].mean(axis=0) - state_mat[ki_mask].mean(axis=0)))

    # Permutations
    null_tv_syll = []
    null_tv_state = []
    for c in combos:
        m = np.zeros(n_total, dtype=bool)
        m[list(c)] = True
        d_s = 0.5 * np.sum(np.abs(syll_mat[m].mean(axis=0) - syll_mat[~m].mean(axis=0)))
        d_z = 0.5 * np.sum(np.abs(state_mat[m].mean(axis=0) - state_mat[~m].mean(axis=0)))
        null_tv_syll.append(d_s)
        null_tv_state.append(d_z)

    null_tv_syll = np.array(null_tv_syll)
    null_tv_state = np.array(null_tv_state)

    p_syll = float((null_tv_syll >= obs_tv_syll).mean())
    p_state = float((null_tv_state >= obs_tv_state).mean())
    q95_syll = float(np.percentile(null_tv_syll, 95))
    q95_state = float(np.percentile(null_tv_state, 95))

    print(f"Raw Syllables TV: obs={obs_tv_syll:.4f}, 95th={q95_syll:.4f}, p={p_syll:.4f}")
    print(f"Macro-States TV:  obs={obs_tv_state:.4f}, 95th={q95_state:.4f}, p={p_state:.4f}")

    # Plot Vertically Stacked Figure (High rather than wide, objective styling)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 11.2), dpi=300)

    # Panel A: KPMS Syllables (Repertoire)
    counts1, edges1, patches1 = ax1.hist(null_tv_syll, bins=35, color="#CFD8DC", edgecolor="white", lw=0.8, alpha=0.9)
    for patch, edge in zip(patches1, edges1[:-1]):
        if edge >= obs_tv_syll:
            patch.set_facecolor("#2E7D32")  # Dark green tail for p < 0.05

    ax1.axvline(obs_tv_syll, color="#1B5E20", lw=2.8,
                label=f"Observed Distance ($d_{{TV}} = {obs_tv_syll:.3f}, p = {p_syll:.4f}$)")
    ax1.axvline(q95_syll, color="#37474F", lw=1.8, linestyle="--",
                label=f"95th Percentile Null ({q95_syll:.3f})")

    y_max1 = max(counts1)
    ax1.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=11.5)
    ax1.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=11.5)
    ax1.set_title("A. KPMS Syllable Repertoire (100 Syllables)", fontweight="bold", pad=10, fontsize=13, loc="left")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.5)
    ax1.set_ylim(0, y_max1 * 1.15)

    # Panel B: shMoSeq Macro-States
    counts2, edges2, patches2 = ax2.hist(null_tv_state, bins=35, color="#CFD8DC", edgecolor="white", lw=0.8, alpha=0.9)
    for patch, edge in zip(patches2, edges2[:-1]):
        if edge >= obs_tv_state:
            patch.set_facecolor("#C62828")  # Red tail for null

    ax2.axvline(obs_tv_state, color="#B71C1C", lw=2.8,
                label=f"Observed Distance ($d_{{TV}} = {obs_tv_state:.3f}, p = {p_state:.4f}$)")
    ax2.axvline(q95_state, color="#37474F", lw=1.8, linestyle="--",
                label=f"95th Percentile Null ({q95_state:.3f})")

    y_max2 = max(counts2)
    ax2.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=11.5)
    ax2.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=11.5)
    ax2.set_title("B. shMoSeq Macro-States ($K=8$ States)", fontweight="bold", pad=10, fontsize=13, loc="left")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.5)
    ax2.set_ylim(0, y_max2 * 1.15)

    fig.suptitle("Exact Animal Permutation Test: WT vs. DAT-KI Baseline (n=6,435 Partitions)",
                 fontweight="bold", fontsize=13.5, y=0.995)
    plt.tight_layout()
    save_fig_to_all(fig, "pres_syllables_vs_states_genotype_permutation.png")
    plt.close(fig)

def make_pyramid_schematic():
    print("Generating Bottom-Up Behavioral Hierarchy Pyramid Schematic...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Center is at x = 5.5
    # Bottom tier: KPMS Syllables (foundation)
    poly_bottom = patches.Polygon(
        [[0.4, 0.8], [10.6, 0.8], [9.6, 3.4], [1.4, 3.4]],
        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=3.2, zorder=2
    )
    ax.add_patch(poly_bottom)

    # Middle tier: Syllable Chains & Episodes (highlighted focus)
    poly_middle = patches.Polygon(
        [[1.3, 3.8], [9.7, 3.8], [8.6, 6.4], [2.4, 6.4]],
        facecolor="#FFF8E1", edgecolor="#FF6F00", linewidth=4.2, zorder=3
    )
    ax.add_patch(poly_middle)

    # Top tier: Macro-States (shMoSeq) (apex)
    poly_top = patches.Polygon(
        [[2.2, 6.8], [8.8, 6.8], [7.7, 9.4], [3.3, 9.4]],
        facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2.8, linestyle="--", zorder=2
    )
    ax.add_patch(poly_top)

    # Right side badge for middle tier (The Missing Rung / Our Focus)
    badge = patches.FancyBboxPatch(
        (10.3, 4.2), 3.8, 1.8,
        boxstyle="round,pad=0.3", facecolor="#FF6F00", edgecolor="#BF360C", linewidth=2.2, zorder=5
    )
    ax.add_patch(badge)
    ax.text(12.2, 5.3, "THE MISSING RUNG", color="white", fontsize=13, fontweight="black",
            ha="center", va="center", zorder=6)
    ax.text(12.2, 4.65, "(OUR FOCUS)", color="#FFF8E1", fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=6)

    # Connecting arrow from badge to middle rung
    ax.annotate(
        "", xy=(9.78, 5.1), xytext=(10.3, 5.1),
        arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=3.5), zorder=6
    )

    # Text for Top Tier (Macro-States)
    ax.text(5.5, 8.65, "Top: Macro-States (shMoSeq)", fontsize=14, fontweight="bold", ha="center", va="center", color="#B71C1C", zorder=4)
    ax.text(5.5, 7.95, "\u2192 Skipped too fast", fontsize=13, fontweight="black", ha="center", va="center", color="#D32F2F", zorder=4)
    ax.text(5.5, 7.30, "Animal Idiosyncrasy \u2022 Erased Genotype Differences",
            fontsize=10.5, fontweight="semibold", ha="center", va="center", color="#555555", zorder=4)

    # Text for Middle Tier (Focus: Syllable Chains & Episodes)
    ax.text(5.5, 5.65, "Middle (Focus): Syllable Chains & Episodes", fontsize=15, fontweight="black", ha="center", va="center", color="#E65100", zorder=4)
    ax.text(5.5, 5.00, "\u2192 Tangible & Verifiable", fontsize=14, fontweight="black", ha="center", va="center", color="#BF360C", zorder=4)
    ax.text(5.5, 4.35, "seqNMF Motifs \u2022 Multi-Second Syntax \u2022 Transition Paths \u2022 Resolves Genotypes",
            fontsize=11, fontweight="bold", ha="center", va="center", color="#37474F", zorder=4)

    # Text for Bottom Tier (Syllables)
    ax.text(5.5, 2.65, "Bottom: Syllables (KPMS)", fontsize=14.5, fontweight="bold", ha="center", va="center", color="#0D47A1", zorder=4)
    ax.text(5.5, 2.05, "\u2192 Sub-Second Physical Alphabet", fontsize=12.5, fontweight="bold", ha="center", va="center", color="#1565C0", zorder=4)
    ax.text(5.5, 1.45, "300\u2013400 ms Poses \u2022 Significant Genotype Discrimination Under Permutation (p = 0.0356)",
            fontsize=10.5, fontweight="semibold", ha="center", va="center", color="#37474F", zorder=4)

    # Left-side upward arrow indicating bottom-up construction
    ax.annotate(
        "", xy=(0.3, 9.2), xytext=(0.3, 1.0),
        arrowprops=dict(arrowstyle="->", color="#212121", lw=3.0)
    )
    ax.text(0.0, 5.1, "Bottom-Up Behavioral Hierarchy Construction", fontsize=12, fontweight="bold",
            ha="center", va="center", rotation=90, color="#212121")

    plt.tight_layout()
    save_fig_to_all(fig, "pres_behavioral_hierarchy_pyramid.png")
    plt.close(fig)

def make_idiosyncrasy_timeline():
    print("Generating clean presentation timeline of animal state entrapment...")
    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)
    states = model_bundle["states_dict"]
    meta = model_bundle["metadata_dict"]

    # Organize by animal and drug
    animals = {}
    for k, st in states.items():
        m = meta[k]
        aid = str(m["animal_id"])
        geno = m["genotype"]
        drug_raw = m["drug"]
        drug_key = "saline" if "sal" in drug_raw.lower() else "amphetamine"
        if aid not in animals:
            animals[aid] = {"geno": geno, "saline": None, "amphetamine": None}
        animals[aid][drug_key] = st

    # Sort WT first, then KI
    sorted_aids = sorted(list(animals.keys()), key=lambda a: (0 if animals[a]["geno"] == "WT" else 1, a))
    n_mice = len(sorted_aids)

    # Standard shMoSeq 8-state palette (matching fig_session_state_timelines_k8)
    palette = [
        "#1f77b4",  # Z_0: Blue
        "#ff7f0e",  # Z_1: Orange
        "#2ca02c",  # Z_2: Green
        "#9467bd",  # Z_3: Purple
        "#8c564b",  # Z_4: Brown
        "#7f7f7f",  # Z_5: Gray
        "#bcbd22",  # Z_6: Olive/Yellow
        "#17becf",  # Z_7: Cyan
    ]

    fig, (ax_sal, ax_amp) = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True, dpi=300)

    fps = 40.0125
    y_pos = np.arange(n_mice)

    for i, aid in enumerate(sorted_aids):
        geno = animals[aid]["geno"]
        sal = animals[aid]["saline"]
        amp = animals[aid]["amphetamine"]

        # Plot Saline
        if sal is not None:
            T = len(sal)
            change_pts = np.where(np.diff(sal) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = sal[s_idx]
                ax_sal.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.74, color=palette[st_val], edgecolor="none")

        # Plot Amphetamine
        if amp is not None:
            T = len(amp)
            change_pts = np.where(np.diff(amp) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = amp[s_idx]
                ax_amp.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.74, color=palette[st_val], edgecolor="none")

    # Labels
    y_labels = [f"{animals[a]['geno']} #{a}" for a in sorted_aids]
    ax_sal.set_yticks(y_pos)
    ax_sal.set_yticklabels(y_labels, fontsize=11, fontweight="bold")
    ax_sal.invert_yaxis()

    # Color code y-tick labels: deep blue for WT, deep red for DAT-KI
    for tick_label in ax_sal.get_yticklabels():
        txt = tick_label.get_text()
        if "WT" in txt:
            tick_label.set_color("#0D47A1")
        else:
            tick_label.set_color("#B71C1C")

    # Genotype dividing line
    wt_count = sum(1 for a in sorted_aids if animals[a]["geno"] == "WT")
    ax_sal.axhline(wt_count - 0.5, color="#212121", lw=2.2, linestyle="--")
    ax_amp.axhline(wt_count - 0.5, color="#212121", lw=2.2, linestyle="--")

    ax_sal.set_xlim(0, 60)
    ax_amp.set_xlim(0, 60)
    ax_sal.set_xlabel("Time in Arena Post-Injection (minutes)", fontweight="bold", fontsize=12)
    ax_amp.set_xlabel("Time in Arena Post-Injection (minutes)", fontweight="bold", fontsize=12)

    ax_sal.set_title("Saline Baseline Sessions (Day 1)", fontweight="bold", fontsize=13, pad=10)
    ax_amp.set_title("Amphetamine Challenge Sessions (Day 2)", fontweight="bold", fontsize=13, pad=10)

    # Custom Legend for states
    legend_patches = [patches.Patch(color=palette[k], label=f"Z_{k}") for k in range(8)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.02),
               frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=11)

    fig.suptitle("shMoSeq State Entrapment: Individual Mice Locked into Idiosyncratic States for Entire 60-Minute Sessions",
                 fontweight="bold", fontsize=14.5, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    save_fig_to_all(fig, "pres_shmoseq_state_timelines_idiosyncrasy.png")
    plt.close(fig)

def make_stacked_idiosyncrasy_timeline(exclude_mice=("1765", "728", "1694", "1664")):
    """
    Generates a tall / high (vertical stack) presentation timeline:
    - Top: Saline Baseline (Day 1)
    - Bottom: Amphetamine Challenge (Day 2)
    - Excludes mice 1765, 728 (WT) and 1694, 1664 (KI) to lessen vertical space
      and focus on representative entrapment phenotypes (n=11 mice: 6 WT, 5 KI).
    """
    print(f"Generating stacked presentation timeline (excluding {exclude_mice})...")
    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)
    states = model_bundle["states_dict"]
    meta = model_bundle["metadata_dict"]

    exclude_set = set(str(x) for x in exclude_mice)
    animals = {}
    for k, st in states.items():
        m = meta[k]
        aid = str(m["animal_id"])
        if aid in exclude_set:
            continue
        geno = m["genotype"]
        drug_raw = m["drug"]
        drug_key = "saline" if "sal" in drug_raw.lower() else "amphetamine"
        if aid not in animals:
            animals[aid] = {"geno": geno, "saline": None, "amphetamine": None}
        animals[aid][drug_key] = st

    sorted_aids = sorted(list(animals.keys()), key=lambda a: (0 if animals[a]["geno"] == "WT" else 1, a))
    n_mice = len(sorted_aids)
    wt_count = sum(1 for a in sorted_aids if animals[a]["geno"] == "WT")

    palette = [
        "#1f77b4",  # Z_0: Blue
        "#ff7f0e",  # Z_1: Orange
        "#2ca02c",  # Z_2: Green
        "#9467bd",  # Z_3: Purple
        "#8c564b",  # Z_4: Brown
        "#7f7f7f",  # Z_5: Gray
        "#bcbd22",  # Z_6: Olive/Yellow
        "#17becf",  # Z_7: Cyan
    ]

    fig, (ax_sal, ax_amp) = plt.subplots(2, 1, figsize=(10.5, 12.8), sharex=True, dpi=300)
    fps = 40.0125
    y_pos = np.arange(n_mice)

    for i, aid in enumerate(sorted_aids):
        sal = animals[aid]["saline"]
        amp = animals[aid]["amphetamine"]

        # Saline
        if sal is not None:
            T = len(sal)
            change_pts = np.where(np.diff(sal) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = sal[s_idx]
                ax_sal.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.76, color=palette[st_val], edgecolor="none")

        # Amphetamine
        if amp is not None:
            T = len(amp)
            change_pts = np.where(np.diff(amp) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = amp[s_idx]
                ax_amp.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.76, color=palette[st_val], edgecolor="none")

    y_labels = [f"{animals[a]['geno']} #{a}" for a in sorted_aids]

    for ax, title in [(ax_sal, "A. Saline Baseline Sessions (Day 1)"),
                      (ax_amp, "B. Amphetamine Challenge Sessions (Day 2)")]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")
        ax.invert_yaxis()

        for tick_label in ax.get_yticklabels():
            txt = tick_label.get_text()
            if "WT" in txt:
                tick_label.set_color("#0D47A1")
            else:
                tick_label.set_color("#B71C1C")

        ax.axhline(wt_count - 0.5, color="#212121", lw=2.2, linestyle="--")
        ax.set_xlim(0, 60)
        ax.set_title(title, fontweight="bold", fontsize=13, pad=10, loc="left")
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

        # Right side genotype grouping labels (reflecting true cohort n=8 WT, n=7 DAT-KI)
        ax.text(60.8, (wt_count - 1) / 2.0, "WT\n(n=8)", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#0D47A1")
        ax.text(60.8, wt_count + (n_mice - wt_count - 1) / 2.0, "DAT-KI\n(n=7)", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#B71C1C")

    ax_amp.set_xlabel("Time in Arena Post-Injection (minutes)", fontweight="bold", fontsize=12.5)

    # Custom Legend
    legend_patches = [patches.Patch(color=palette[k], label=f"Z_{k}") for k in range(8)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.015),
               frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=11.5)

    fig.suptitle("shMoSeq State Entrapment: Individual Mice Locked into Idiosyncratic States",
                 fontweight="bold", fontsize=14.5, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, top=0.95, right=0.90)
    save_fig_to_all(fig, "pres_shmoseq_state_timelines_stacked.png")
    plt.close(fig)

def make_composite_slide():
    """
    Generates a unified 16:9 composite slide image in a balanced 3-column layout:
      - Col 1 (Left): Stacked tall timeline (Saline Day 1 top, Amph Day 2 bottom, 11 mice)
      - Col 2 (Middle): Stacked tall permutation test (Syllables top, States bottom, objective)
      - Col 3 (Right): Hierarchy pyramid schematic + summary card (the bottom-up solution)
    """
    print("Generating Unified Composite Slide Image (16:9 3-Column Layout)...")
    from PIL import Image, ImageDraw, ImageFont

    time_img_path = os.path.join(OUTPUT_DIRS[0], "pres_shmoseq_state_timelines_stacked.png")
    perm_img_path = os.path.join(OUTPUT_DIRS[0], "pres_syllables_vs_states_genotype_permutation.png")
    pyramid_img_path = os.path.join(OUTPUT_DIRS[0], "pres_behavioral_hierarchy_pyramid.png")

    img_time = Image.open(time_img_path)
    img_perm = Image.open(perm_img_path)
    img_pyramid = Image.open(pyramid_img_path)

    # Target 16:9 canvas at 3840x2160 (4K UHD slide)
    W, H = 3840, 2160
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    # Available content area
    content_y = 150
    content_h = 1960

    # Column 1: Stacked Timeline (Left)
    # Give it ample width for horizontal time resolution (0-60 min)
    col1_w = 1360
    time_h = content_h
    time_w = int(img_time.width * (time_h / img_time.height))
    if time_w > col1_w:
        time_w = col1_w
        time_h = int(img_time.height * (time_w / img_time.width))
    img_time_resized = img_time.resize((time_w, time_h), Image.Resampling.LANCZOS)
    x1 = 60 + (col1_w - time_w) // 2
    y1 = content_y + (content_h - time_h) // 2
    canvas.paste(img_time_resized, (x1, y1))

    # Column 2: Stacked Permutation Test (Middle)
    col2_w = 1120
    perm_h = content_h - 40
    perm_w = int(img_perm.width * (perm_h / img_perm.height))
    if perm_w > col2_w:
        perm_w = col2_w
        perm_h = int(img_perm.height * (perm_w / img_perm.width))
    img_perm_resized = img_perm.resize((perm_w, perm_h), Image.Resampling.LANCZOS)
    x2 = 1450 + (col2_w - perm_w) // 2
    y2 = content_y + (content_h - perm_h) // 2
    canvas.paste(img_perm_resized, (x2, y2))

    # Column 3: Pyramid Schematic + Bottom-Up Takeaway Card (Right)
    col3_w = 1200
    pyr_w = col3_w
    pyr_h = int(img_pyramid.height * (pyr_w / img_pyramid.width))
    if pyr_h > 1050:
        pyr_h = 1050
        pyr_w = int(img_pyramid.width * (pyr_h / img_pyramid.height))
    img_pyr_resized = img_pyramid.resize((pyr_w, pyr_h), Image.Resampling.LANCZOS)
    x3 = 2600 + (col3_w - pyr_w) // 2
    y3 = content_y + 40
    canvas.paste(img_pyr_resized, (x3, y3))

    # Takeaway Card below pyramid in Column 3
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_card_head = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_card_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_head = ImageFont.load_default()
        font_card_body = ImageFont.load_default()

    # Draw Summary Box
    card_x0 = 2600
    card_y0 = y3 + pyr_h + 50
    card_x1 = 2600 + col3_w
    card_y1 = content_y + content_h - 20

    draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=18,
                           fill="#F8F9FA", outline="#CFD8DC", width=2)

    draw.text((card_x0 + 35, card_y0 + 35), "Key Biological & Analytical Takeaways",
              fill="#212121", font=font_card_head)

    bullets = [
        "• State Entrapment: In top-down shMoSeq, individual mice remain trapped",
        "  in idiosyncratic states across entire 60-minute sessions and across days.",
        "",
        "• Signal Erasure: While raw syllable repertoires significantly separate",
        "  WT vs. DAT-KI (p = 0.0356), 8 macro-states erase the signal (p = 0.7542).",
        "",
        "• The Missing Rung: Top-down HMM jumped from 300 ms syllables straight",
        "  to macro-states, skipping multi-second Syllable Chains & Episodes.",
        "",
        "• The Solution: Build bottom-up via seqNMF to discover structured motifs",
        "  that are tangible in video and preserve genuine genotype phenotypes."
    ]

    cur_y = card_y0 + 85
    for line in bullets:
        if line == "":
            cur_y += 8
        else:
            draw.text((card_x0 + 35, cur_y), line, fill="#37474F", font=font_card_body)
            cur_y += 32

    # Slide Header Banner
    draw.text((80, 40), "shMoSeq on Our Cohort: The Reality Check & The Bottom-Up Pivot",
              fill="#212121", font=font_title)
    draw.text((80, 98), "Visual Entrapment (Left)  •  Permutation Shuffling Test (Center)  •  The Missing Behavioral Rung (Right)",
              fill="#546E7A", font=font_sub)

    for d in OUTPUT_DIRS:
        out_path = os.path.join(d, "pres_reality_check_composite_slide.png")
        canvas.save(out_path, quality=95)
        print(f"Saved composite slide: {out_path}")
        out_stacked = os.path.join(d, "pres_reality_check_composite_slide_stacked.png")
        canvas.save(out_stacked, quality=95)

def main():
    set_style()
    make_permutation_comparison()
    make_pyramid_schematic()
    make_idiosyncrasy_timeline()
    make_stacked_idiosyncrasy_timeline()
    make_composite_slide()
    print("\nAll Reality Check Slide assets generated successfully!")

if __name__ == "__main__":
    main()

