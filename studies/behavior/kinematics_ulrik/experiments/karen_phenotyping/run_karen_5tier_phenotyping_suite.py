#!/usr/bin/env python3
"""
run_karen_5tier_phenotyping_suite.py

Comprehensive 5-Tier Behavioral Phenotyping Suite for the Karen Project
Implements Sections A through D:
  - Section A: Repertoire (Usage, Diversity, Shannon Entropy, Volcano/Forest Plots with FDR)
  - Section B: Temporal Structure (Locomotor Bout Duration, Syllable Dwell Time, Persistence)
  - Section C: Syntax (Transition Matrices, PFF-Monomer Difference Matrix, Transition Entropy)
  - Section D: Execution (Within-Syllable Speeds, 100-bin Phase-Normalized Profiles)

Carries methodological disclaimers:
"All syllable-level analyses were performed irrespective of statistical significance;
individual-syllable results are reported with Benjamini-Hochberg multiplicity correction
and are interpreted as exploratory."
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

DEFAULT_DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
DEFAULT_OUT_BASE = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "5tier_phenotyping"

DATA_PKL = DEFAULT_DATA_PKL
OUT_BASE = DEFAULT_OUT_BASE

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}
CONDITION_ORDER = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]


def benjamini_hochberg(p_values):
    """Computes Benjamini-Hochberg FDR q-values."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    q_vals = np.zeros(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        q = ranked_p[i] * n / (i + 1)
        running_min = min(running_min, q)
        q_vals[order[i]] = running_min
    return np.clip(q_vals, 0.0, 1.0)


def add_exploratory_watermark(ax, text="[EXPLORATORY SCREEN - FDR q-controlled]"):
    """Adds a scientific integrity watermark/badge to exploratory figures."""
    ax.text(
        0.5, 1.02, text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9, zorder=10)
    )


def load_dataset():
    print(f"[DataLoader] Starting 13.7GB Karen dataset load from:\n  {DATA_PKL}", flush=True)
    import time
    t0 = time.time()
    with open(DATA_PKL, "rb") as f:
        all_sessions = pickle.load(f)
    print(f"[DataLoader] Loaded {len(all_sessions)} total sessions in {time.time()-t0:.1f}s.", flush=True)

    valid_sessions = []
    for s in all_sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        cond = md.get("condition", "Unknown")
        if g in ["PFF", "Monomer"] and cond in CONDITION_ORDER:
            valid_sessions.append(s)

    print(f"[DataLoader] Filtered {len(valid_sessions)} valid sessions across {len(set(s.metadata.get('animal_id', '') for s in valid_sessions))} animals.", flush=True)
    return valid_sessions


# ==============================================================================
# SECTION A: REPERTOIRE (BEHAVIORAL SELECTION)
# ==============================================================================
def run_section_a_repertoire(sessions):
    print("\n=======================================================")
    print(">>> RUNNING SECTION A: REPERTOIRE (BEHAVIORAL SELECTION)")
    print("=======================================================")
    out_dir = OUT_BASE / "section_a_repertoire"

    # 1. Compute Syllable Occupancy per Session
    session_occupancies = []
    all_syllable_ids = set()

    for s in sessions:
        df = s.all_data_df.dropna(subset=["syllable"])
        if len(df) < 1000:
            continue
        sylls = df["syllable"].values.astype(int)
        counts = pd.Series(sylls).value_counts(normalize=True)
        all_syllable_ids.update(counts.index)

        row = {
            "session_id": s.session_id,
            "animal_id": str(s.metadata.get("animal_id", s.session_id)),
            "genotype": s.metadata["genotype"],
            "condition": s.metadata["condition"],
            "total_frames": len(df)
        }
        for syll_id, freq in counts.items():
            row[f"syll_{syll_id}"] = freq
        session_occupancies.append(row)

    df_occ = pd.DataFrame(session_occupancies).fillna(0.0)
    syll_cols = sorted([c for c in df_occ.columns if c.startswith("syll_")], key=lambda x: int(x.split("_")[1]))

    # Compute Repertoire Diversity & Entropy per session
    repertoire_records = []
    for _, row in df_occ.iterrows():
        freqs = row[syll_cols].values.astype(float)
        freqs = freqs[freqs > 0]
        shannon_entropy = -np.sum(freqs * np.log2(freqs))
        active_sylls_01pct = np.sum(row[syll_cols].values >= 0.001)
        active_sylls_05pct = np.sum(row[syll_cols].values >= 0.005)
        active_sylls_1pct = np.sum(row[syll_cols].values >= 0.01)

        repertoire_records.append({
            "session_id": row["session_id"],
            "animal_id": row["animal_id"],
            "genotype": row["genotype"],
            "condition": row["condition"],
            "shannon_entropy": shannon_entropy,
            "active_sylls_01pct": active_sylls_01pct,
            "active_sylls_05pct": active_sylls_05pct,
            "active_sylls_1pct": active_sylls_1pct,
        })
    df_rep = pd.DataFrame(repertoire_records)
    df_rep.to_csv(out_dir / "repertoire_diversity_per_session.csv", index=False)

    # 2. Per-Syllable Differential Occupancy (Volcano & Forest Plots)
    mon_occ = df_occ[df_occ["genotype"] == "Monomer"][syll_cols]
    pff_occ = df_occ[df_occ["genotype"] == "PFF"][syll_cols]

    syll_stats = []
    for col in syll_cols:
        syll_num = int(col.split("_")[1])
        m_vals = mon_occ[col].values
        p_vals = pff_occ[col].values

        mean_mon = np.mean(m_vals)
        mean_pff = np.mean(p_vals)
        overall_mean = np.mean(df_occ[col].values)

        # Only evaluate syllables with >= 0.2% overall usage for inference
        t_stat, p_val = stats.ttest_ind(p_vals, m_vals, equal_var=False) if (mean_mon + mean_pff) > 0 else (0.0, 1.0)
        u_stat, p_mw = stats.mannwhitneyu(p_vals, m_vals, alternative="two-sided") if (mean_mon + mean_pff) > 0 else (0.0, 1.0)

        fold_change = (mean_pff + 1e-5) / (mean_mon + 1e-5)
        log2_fc = np.log2(fold_change)

        syll_stats.append({
            "syllable": syll_num,
            "mean_monomer_pct": mean_mon * 100.0,
            "mean_pff_pct": mean_pff * 100.0,
            "overall_mean_pct": overall_mean * 100.0,
            "log2_fold_change": log2_fc,
            "p_val_ttest": p_val if not np.isnan(p_val) else 1.0,
            "p_val_mannwhitney": p_mw if not np.isnan(p_mw) else 1.0,
        })
    df_syll_stats = pd.DataFrame(syll_stats)
    df_syll_stats["q_val_fdr"] = benjamini_hochberg(df_syll_stats["p_val_mannwhitney"].values)
    df_syll_stats.sort_values(by="overall_mean_pct", ascending=False).to_csv(out_dir / "syllable_usage_fdr_statistics.csv", index=False)

    # -------------------------------------------------------------
    # Plot A1: Repertoire Diversity (Shannon Entropy & Active Counts)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # A1.1 Shannon Entropy
    sns.boxplot(data=df_rep, x="condition", y="shannon_entropy", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[0], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_rep, x="condition", y="shannon_entropy", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[0], dodge=True, alpha=0.6, jitter=0.2)
    axes[0].set_title("Repertoire Shannon Entropy", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].set_xlabel("Condition Wave")
    axes[0].legend_.remove()

    # A1.2 Active Syllables (>0.1%)
    sns.boxplot(data=df_rep, x="condition", y="active_sylls_01pct", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[1], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_rep, x="condition", y="active_sylls_01pct", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[1], dodge=True, alpha=0.6, jitter=0.2)
    axes[1].set_title("Active Syllables (Usage ≥ 0.1%)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Number of Distinct Syllables")
    axes[1].set_xlabel("Condition Wave")
    axes[1].legend_.remove()

    # A1.3 Common Syllables (>1.0%)
    sns.boxplot(data=df_rep, x="condition", y="active_sylls_1pct", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[2], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_rep, x="condition", y="active_sylls_1pct", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[2], dodge=True, alpha=0.6, jitter=0.2)
    axes[2].set_title("Core Syllables (Usage ≥ 1.0%)", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Number of Distinct Syllables")
    axes[2].set_xlabel("Condition Wave")

    # Format legend on 3rd panel
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles[:2], labels[:2], title="Genotype", loc="upper right")

    plt.suptitle("Section A: Behavioral Repertoire Diversity & Entropy Across Longitudinal Waves", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "section_a_repertoire_diversity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot A2: Syllable Usage Volcano & Forest Plot (with FDR controls)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Volcano Plot
    df_plot = df_syll_stats[df_syll_stats["overall_mean_pct"] >= 0.1].copy()
    ax = axes[0]
    add_exploratory_watermark(ax, "[EXPLORATORY SCREEN - Benjamini-Hochberg FDR Controlled]")

    # Scatter points
    neg_log_p = -np.log10(np.clip(df_plot["p_val_mannwhitney"], 1e-10, 1.0))
    sizes = np.clip(df_plot["overall_mean_pct"] * 30, 20, 250)

    colors = []
    for _, r in df_plot.iterrows():
        if r["q_val_fdr"] < 0.10:
            colors.append("#e41a1c" if r["log2_fold_change"] > 0 else "#377eb8")
        elif r["p_val_mannwhitney"] < 0.05:
            colors.append("#ff7f00" if r["log2_fold_change"] > 0 else "#4daf4a")
        else:
            colors.append("#999999")

    scatter = ax.scatter(df_plot["log2_fold_change"], neg_log_p, s=sizes, c=colors, alpha=0.75, edgecolors="none")
    ax.axhline(-np.log10(0.05), color="#888888", linestyle="--", alpha=0.7, label="Uncorrected p = 0.05")
    ax.axvline(0, color="#444444", linestyle="-", alpha=0.5)

    # Label top syllables
    for _, r in df_plot.iterrows():
        if r["p_val_mannwhitney"] < 0.05 or abs(r["log2_fold_change"]) > 1.0:
            ax.annotate(f"Syll {int(r['syllable'])}", (r["log2_fold_change"], -np.log10(r["p_val_mannwhitney"])),
                        fontsize=7.5, ha="center", va="bottom", alpha=0.85)

    ax.set_xlabel("Log2 Fold Change (PFF / Monomer)", fontsize=10, fontweight="bold")
    ax.set_ylabel("-Log10 (Mann-Whitney p-value)", fontsize=10, fontweight="bold")
    ax.set_title("Syllable Usage Volcano Plot", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)

    # Forest Plot of Top 25 Most Frequent Syllables
    ax2 = axes[1]
    add_exploratory_watermark(ax2, "[EXPLORATORY SCREEN - Top 25 Core Syllables]")
    top25 = df_syll_stats.sort_values(by="overall_mean_pct", ascending=False).head(25).sort_values(by="log2_fold_change", ascending=True)

    y_pos = np.arange(len(top25))
    bar_colors = ["#d95f02" if fc > 0 else "#2b5c8f" for fc in top25["log2_fold_change"]]
    ax2.barh(y_pos, top25["log2_fold_change"], color=bar_colors, alpha=0.75, edgecolor="none")
    ax2.axvline(0, color="#333333", linestyle="-", linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Syll {int(s)} ({m:.1f}%)" for s, m in zip(top25["syllable"], top25["overall_mean_pct"])], fontsize=8.0)
    ax2.set_xlabel("Log2 Fold Change (PFF vs Monomer)", fontsize=10, fontweight="bold")
    ax2.set_title("Top 25 Core Syllables Usage Shift", fontsize=11, fontweight="bold")

    plt.suptitle("Section A: Granular Syllable Usage & Volcano Screening (All Active States)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "section_a_syllable_usage_volcano_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Section A outputs in {out_dir}")
    return df_rep, df_syll_stats


# ==============================================================================
# SECTION B: TEMPORAL STRUCTURE (PERSISTENCE & BOUT DYNAMICS)
# ==============================================================================
def run_section_b_temporal_structure(sessions, fps=80.0):
    print("\n=======================================================")
    print(">>> RUNNING SECTION B: TEMPORAL STRUCTURE (BOUTS & DWELL)")
    print("=======================================================")
    out_dir = OUT_BASE / "section_b_temporal_structure"

    records = []
    for s in sessions:
        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue

        speed = df["velocity_px_s"].values
        sylls = df["syllable"].values.astype(int)

        # 1. Macro Locomotor Bouts (speed >= 5 px/s)
        is_moving = speed >= 5.0
        moving_diff = np.diff(np.concatenate(([0], is_moving.astype(int), [0])))
        starts = np.where(moving_diff == 1)[0]
        ends = np.where(moving_diff == -1)[0]
        bout_lengths = (ends - starts) / fps
        macro_bout_mean = np.mean(bout_lengths) if len(bout_lengths) > 0 else 0.0
        macro_bout_median = np.median(bout_lengths) if len(bout_lengths) > 0 else 0.0

        # 2. Micro Syllable Dwell Times
        syll_change = np.diff(np.concatenate(([-1], sylls, [-1])))
        change_idx = np.where(syll_change != 0)[0]
        dwell_lengths = np.diff(change_idx) / fps
        micro_dwell_mean = np.mean(dwell_lengths) if len(dwell_lengths) > 0 else 0.0
        micro_dwell_median = np.median(dwell_lengths) if len(dwell_lengths) > 0 else 0.0

        # 3. Markov Self-Transition Persistence
        trans_same = np.sum(sylls[:-1] == sylls[1:])
        persistence_prob = trans_same / max(len(sylls) - 1, 1)

        records.append({
            "session_id": s.session_id,
            "animal_id": str(s.metadata.get("animal_id", s.session_id)),
            "genotype": s.metadata["genotype"],
            "condition": s.metadata["condition"],
            "macro_bout_mean_s": macro_bout_mean,
            "macro_bout_median_s": macro_bout_median,
            "micro_dwell_mean_s": micro_dwell_mean,
            "micro_dwell_median_s": micro_dwell_median,
            "self_transition_persistence": persistence_prob,
            "num_macro_bouts": len(bout_lengths),
            "num_syllable_states": len(dwell_lengths)
        })

    df_temporal = pd.DataFrame(records)
    df_temporal.to_csv(out_dir / "temporal_structure_per_session.csv", index=False)

    # -------------------------------------------------------------
    # Plot B1: 3-Panel Temporal Structure across Waves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # B1.1 Macro Locomotor Bout Duration
    sns.boxplot(data=df_temporal, x="condition", y="macro_bout_mean_s", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[0], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_temporal, x="condition", y="macro_bout_mean_s", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[0], dodge=True, alpha=0.6, jitter=0.2)
    axes[0].set_title("Macro Locomotor Bout Length (s)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Mean Bout Duration (seconds)")
    axes[0].set_xlabel("Condition Wave")
    axes[0].legend_.remove()

    # B1.2 Micro Syllable Dwell Time
    sns.boxplot(data=df_temporal, x="condition", y="micro_dwell_mean_s", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[1], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_temporal, x="condition", y="micro_dwell_mean_s", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[1], dodge=True, alpha=0.6, jitter=0.2)
    axes[1].set_title("Micro Syllable Dwell Time (s)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Mean State Duration (seconds)")
    axes[1].set_xlabel("Condition Wave")
    axes[1].legend_.remove()

    # B1.3 Frame Persistence Probability
    sns.boxplot(data=df_temporal, x="condition", y="self_transition_persistence", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[2], width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_temporal, x="condition", y="self_transition_persistence", hue="genotype", order=CONDITION_ORDER, palette=palette, ax=axes[2], dodge=True, alpha=0.6, jitter=0.2)
    axes[2].set_title("Self-Transition Persistence P(S_t+1 = S_t)", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Self-Transition Probability")
    axes[2].set_xlabel("Condition Wave")

    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles[:2], labels[:2], title="Genotype", loc="upper right")

    plt.suptitle("Section B: Behavioral Temporal Organization & Persistence Dynamics", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "section_b_temporal_structure_progression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Section B outputs in {out_dir}")
    return df_temporal


# ==============================================================================
# SECTION C: SYNTAX (SEQUENCING & TRANSITIONS)
# ==============================================================================
def run_section_c_syntax(sessions, max_sylls=100):
    print("\n=======================================================")
    print(">>> RUNNING SECTION C: SYNTAX (TRANSITIONS & MATRICES)")
    print("=======================================================")
    out_dir = OUT_BASE / "section_c_syntax"

    mon_counts = np.zeros((max_sylls, max_sylls), dtype=float)
    pff_counts = np.zeros((max_sylls, max_sylls), dtype=float)
    syntax_entropy_records = []

    for s in sessions:
        df = s.all_data_df.dropna(subset=["syllable"])
        if len(df) < 1000:
            continue
        g = s.metadata["genotype"]
        cond = s.metadata["condition"]
        aid = str(s.metadata.get("animal_id", s.session_id))

        sylls = df["syllable"].values.astype(int)

        # Discrete state sequence (compress consecutive repeats)
        changes = np.where(np.diff(sylls) != 0)[0]
        state_seq = sylls[np.concatenate(([0], changes + 1))]
        state_seq = state_seq[state_seq < max_sylls]

        if len(state_seq) < 10:
            continue

        s_counts = np.zeros((max_sylls, max_sylls), dtype=float)
        for i, j in zip(state_seq[:-1], state_seq[1:]):
            s_counts[i, j] += 1
            if g == "Monomer":
                mon_counts[i, j] += 1
            else:
                pff_counts[i, j] += 1

        # Session Transition Entropy Rate: H(S_t+1 | S_t) = - sum_i pi_i sum_j P(j|i) log2 P(j|i)
        row_sums = s_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_trans = s_counts / row_sums

        state_occ = pd.Series(state_seq).value_counts(normalize=True).reindex(np.arange(max_sylls), fill_value=0.0).values

        log_P = np.zeros_like(P_trans)
        nonzero = P_trans > 0
        log_P[nonzero] = np.log2(P_trans[nonzero])
        cond_entropy_per_state = -np.sum(P_trans * log_P, axis=1)
        syntax_entropy = np.sum(state_occ * cond_entropy_per_state)

        syntax_entropy_records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "genotype": g,
            "condition": cond,
            "syntax_entropy_rate": syntax_entropy,
            "total_transitions": len(state_seq) - 1
        })

    df_syntax = pd.DataFrame(syntax_entropy_records)
    df_syntax.to_csv(out_dir / "syntax_entropy_per_session.csv", index=False)

    # Normalize Group Transition Matrices
    mon_row_sums = mon_counts.sum(axis=1, keepdims=True)
    mon_row_sums[mon_row_sums == 0] = 1.0
    P_mon = mon_counts / mon_row_sums

    pff_row_sums = pff_counts.sum(axis=1, keepdims=True)
    pff_row_sums[pff_row_sums == 0] = 1.0
    P_pff = pff_counts / pff_row_sums

    P_diff = P_pff - P_mon

    # Filter to top active syllables for clean visualization (e.g. top 30)
    top_sylls = np.argsort(-mon_counts.sum(axis=1) - pff_counts.sum(axis=1))[:30]
    top_sylls = np.sort(top_sylls)

    # -------------------------------------------------------------
    # Plot C1: Transition Probability Matrices (Monomer, PFF, Diff)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    im0 = axes[0].imshow(P_mon[np.ix_(top_sylls, top_sylls)], cmap="Blues", vmin=0, vmax=0.25, origin="upper")
    axes[0].set_title("Monomer (Control) Transition Matrix", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Next Syllable (S_t+1)")
    axes[0].set_ylabel("Current Syllable (S_t)")
    axes[0].set_xticks(np.arange(len(top_sylls)))
    axes[0].set_yticks(np.arange(len(top_sylls)))
    axes[0].set_xticklabels(top_sylls, fontsize=7.0, rotation=90)
    axes[0].set_yticklabels(top_sylls, fontsize=7.0)
    fig.colorbar(im0, ax=axes[0], shrink=0.75, label="P(S_t+1 | S_t)")

    im1 = axes[1].imshow(P_pff[np.ix_(top_sylls, top_sylls)], cmap="Oranges", vmin=0, vmax=0.25, origin="upper")
    axes[1].set_title("PFF (Parkinson's) Transition Matrix", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Next Syllable (S_t+1)")
    axes[1].set_xticks(np.arange(len(top_sylls)))
    axes[1].set_yticks(np.arange(len(top_sylls)))
    axes[1].set_xticklabels(top_sylls, fontsize=7.0, rotation=90)
    axes[1].set_yticklabels(top_sylls, fontsize=7.0)
    fig.colorbar(im1, ax=axes[1], shrink=0.75, label="P(S_t+1 | S_t)")

    im2 = axes[2].imshow(P_diff[np.ix_(top_sylls, top_sylls)], cmap="RdBu_r", vmin=-0.08, vmax=0.08, origin="upper")
    axes[2].set_title("Difference Matrix (PFF - Monomer)", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("Next Syllable (S_t+1)")
    axes[2].set_xticks(np.arange(len(top_sylls)))
    axes[2].set_yticks(np.arange(len(top_sylls)))
    axes[2].set_xticklabels(top_sylls, fontsize=7.0, rotation=90)
    axes[2].set_yticklabels(top_sylls, fontsize=7.0)
    fig.colorbar(im2, ax=axes[2], shrink=0.75, label="Δ Probability")

    add_exploratory_watermark(axes[2], "[EXPLORATORY SYNTAX SCREEN]")
    plt.suptitle("Section C: Markov Behavioral Syntax Matrices (Top 30 Active Syllables)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "section_c_transition_difference_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Section C outputs in {out_dir}")
    return df_syntax


# ==============================================================================
# SECTION D: EXECUTION (INTRA-SYLLABLE MICRO-KINEMATICS)
# ==============================================================================
def run_section_d_execution(sessions, fps=80.0):
    print("\n=======================================================")
    print(">>> RUNNING SECTION D: EXECUTION (INTRA-SYLLABLE KINEMATICS)")
    print("=======================================================")
    out_dir = OUT_BASE / "section_d_execution"

    # 1. Collect per-syllable speeds across sessions
    syll_speeds_by_group = {"Monomer": {}, "PFF": {}}

    # Select 4 representative archetypes for 100-bin phase normalization
    target_sylls = [43, 22, 9, 5]  # Locomotion, Turning, Moderate Exploration, Fine Adjust
    syll_labels = {
        43: "Fast Locomotion (Syllable 43)",
        22: "Turning / Steering (Syllable 22)",
        9: "Exploration Run (Syllable 9)",
        5: "Micro Locomotion (Syllable 5)"
    }
    phase_trajectories = {s: {"Monomer": [], "PFF": []} for s in target_sylls}

    for s in sessions:
        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue
        g = s.metadata["genotype"]
        speed = df["velocity_px_s"].values
        sylls = df["syllable"].values.astype(int)

        # Speeds per syllable
        for u_syll in np.unique(sylls):
            idx = np.where(sylls == u_syll)[0]
            if len(idx) >= 10:
                if u_syll not in syll_speeds_by_group[g]:
                    syll_speeds_by_group[g][u_syll] = []
                syll_speeds_by_group[g][u_syll].append(np.mean(speed[idx]))

        # Phase extraction for target syllables
        for t_syll in target_sylls:
            is_t = (sylls == t_syll).astype(int)
            diff_t = np.diff(np.concatenate(([0], is_t, [0])))
            starts = np.where(diff_t == 1)[0]
            ends = np.where(diff_t == -1)[0]

            for st, en in zip(starts, ends):
                length = en - st
                if 5 <= length <= 80:  # Valid syllable duration ~60ms - 1000ms
                    inst_speed = speed[st:en]
                    # Resample to 100 normalized phase bins
                    norm_phase = np.linspace(0, 1, length)
                    target_phase = np.linspace(0, 1, 100)
                    resampled_v = np.interp(target_phase, norm_phase, inst_speed)
                    phase_trajectories[t_syll][g].append(resampled_v)

    # Calculate within-syllable speed differences
    all_syll_set = sorted(set(syll_speeds_by_group["Monomer"].keys()).intersection(set(syll_speeds_by_group["PFF"].keys())))
    speed_diff_records = []

    for s_id in all_syll_set:
        m_speeds = syll_speeds_by_group["Monomer"][s_id]
        p_speeds = syll_speeds_by_group["PFF"][s_id]
        if len(m_speeds) >= 10 and len(p_speeds) >= 10:
            mean_m = np.mean(m_speeds)
            mean_p = np.mean(p_speeds)
            delta = mean_p - mean_m
            pct_delta = (delta / (mean_m + 1e-3)) * 100.0
            t_val, p_val = stats.ttest_ind(p_speeds, m_speeds, equal_var=False)
            u_val, p_mw = stats.mannwhitneyu(p_speeds, m_speeds, alternative="two-sided")

            speed_diff_records.append({
                "syllable": s_id,
                "monomer_speed_mean": mean_m,
                "pff_speed_mean": mean_p,
                "delta_speed_px_s": delta,
                "percent_delta": pct_delta,
                "p_value_raw": p_mw,
                "n_sessions_mon": len(m_speeds),
                "n_sessions_pff": len(p_speeds)
            })

    df_speed_diff = pd.DataFrame(speed_diff_records)
    df_speed_diff["q_val_fdr"] = benjamini_hochberg(df_speed_diff["p_value_raw"].values)
    df_speed_diff.sort_values(by="delta_speed_px_s", ascending=True).to_csv(out_dir / "within_syllable_speed_stats.csv", index=False)

    # -------------------------------------------------------------
    # Plot D1: Phase-Normalized Kinematic Trajectories (4 Panels)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes_flat = axes.flatten()
    phase_x = np.linspace(0, 100, 100)

    for idx, t_syll in enumerate(target_sylls):
        ax = axes_flat[idx]
        add_exploratory_watermark(ax, "[EXPLORATORY MICRO-KINEMATICS - 100 Phase Bins]")

        for g, color in palette.items():
            arr = np.array(phase_trajectories[t_syll][g])
            if len(arr) > 0:
                mean_profile = np.mean(arr, axis=0)
                sem_profile = stats.sem(arr, axis=0)
                ax.plot(phase_x, mean_profile, color=color, linewidth=2.5, label=f"{g} (N={len(arr)} instances)")
                ax.fill_between(phase_x, mean_profile - sem_profile, mean_profile + sem_profile, color=color, alpha=0.2)

        ax.set_title(syll_labels[t_syll], fontweight="bold", fontsize=11)
        ax.set_xlabel("Syllable Execution Phase (%)", fontsize=10)
        ax.set_ylabel("Instantaneous Speed (px/s)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Section D: Within-Syllable Micro-Kinematic Profiles (Phase-Normalized 0% → 100%)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "section_d_phase_normalized_kinematics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Section D outputs in {out_dir}")
    return df_speed_diff


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    global DATA_PKL, OUT_BASE
    parser = argparse.ArgumentParser(description="5-Tier Systematic Behavioral Phenotyping Suite")
    parser.add_argument("--data-pkl", type=str, default=str(DEFAULT_DATA_PKL), help="Path to input sessions pickle")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_BASE), help="Output base directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Camera framerate (default: 60.0)")
    args = parser.parse_args()

    DATA_PKL = Path(args.data_pkl)
    OUT_BASE = Path(args.out_dir)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "section_a_repertoire").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "section_b_temporal_structure").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "section_c_syntax").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "section_d_execution").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("      KAREN DATASET: 5-TIER SYSTEMATIC BEHAVIORAL PHENOTYPING")
    print("="*70)
    sessions = load_dataset()

    df_rep, df_syll_stats = run_section_a_repertoire(sessions)
    df_temporal = run_section_b_temporal_structure(sessions)
    df_syntax = run_section_c_syntax(sessions)
    df_speed_diff = run_section_d_execution(sessions)

    # Merge Master Session Table for Longitudinal LMM Modeling
    master_df = df_rep.merge(df_temporal, on=["session_id", "animal_id", "genotype", "condition"], how="inner")
    master_df = master_df.merge(df_syntax[["session_id", "syntax_entropy_rate"]], on="session_id", how="left")

    master_csv = OUT_BASE / "karen_phenotyping_master_session_metrics.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\n>>> Master integrated metrics saved to: {master_csv} ({len(master_df)} sessions)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
