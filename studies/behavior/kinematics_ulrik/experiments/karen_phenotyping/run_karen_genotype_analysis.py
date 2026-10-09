#!/usr/bin/env python3
"""
run_karen_genotype_analysis.py

Comprehensive behavioral comparison between PFF (Parkinson's Model, N=56)
and Monomer (Control, N=43) in the Karen dataset.

Analyzes and outputs figures and detailed statistical CSVs across 5 dimensions:
1. Global Locomotor & Spatial Kinematics (Speed, Acceleration, Immobility, Turning, Path)
2. Syllable Repertoire & Usage Differences (Volcano plot, Fold Change, Entropy, UMAP)
3. Within-Syllable Micro-Kinematics (Execution speed, duration per syllable)
4. Behavioral Syntax, Transitions & Bigrams (Markov transition matrix differences)
5. Temporal Dynamics & Fatigue Across Session Time (Minute-by-minute progression)
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
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}

import argparse

DEFAULT_DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison"

DATA_PKL = DEFAULT_DATA_PKL
OUT_DIR = DEFAULT_OUT_DIR


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


def load_and_filter_sessions():
    print(f"Loading dataset from {DATA_PKL}...")
    with open(DATA_PKL, "rb") as f:
        all_sessions = pickle.load(f)

    # Filter to PFF and Monomer
    valid_sessions = []
    for s in all_sessions:
        g = getattr(s, "metadata", {}).get("genotype", "Unknown")
        if g in ["PFF", "Monomer"]:
            valid_sessions.append(s)

    print(f"Filtered {len(valid_sessions)} valid sessions:")
    print(f"  PFF: {sum(1 for s in valid_sessions if s.metadata['genotype'] == 'PFF')} sessions")
    print(f"  Monomer: {sum(1 for s in valid_sessions if s.metadata['genotype'] == 'Monomer')} sessions")
    return valid_sessions


# ==============================================================================
# DIMENSION 1: GLOBAL LOCOMOTOR & SPATIAL KINEMATICS
# ==============================================================================
def analyze_global_kinematics(sessions, fps=80.0):
    print("\n--- [Dimension 1] Global Locomotor & Spatial Kinematics ---")
    d1_dir = OUT_DIR / "01_global_kinematics"
    d1_dir.mkdir(parents=True, exist_ok=True)

    session_rows = []
    for s in sessions:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id
        aid = s.metadata.get("animal_id", sid)

        # Valid tracking slice
        vdf = df.dropna(subset=["centroid_x", "centroid_y", "velocity_px_s"])
        if len(vdf) < 1000:
            continue

        speed = vdf["velocity_px_s"].values
        ang_vel = vdf["angular_velocity"].values if "angular_velocity" in vdf.columns else np.zeros_like(speed)
        x = vdf["centroid_x"].values
        y = vdf["centroid_y"].values

        # Metrics
        mean_speed = np.mean(speed)
        median_speed = np.median(speed)
        p95_speed = np.percentile(speed, 95)
        immobility_frac = np.mean(speed < 5.0)  # stationary threshold 5 px/s
        active_speed_mean = np.mean(speed[speed >= 5.0]) if np.sum(speed >= 5.0) > 0 else 0.0

        # Acceleration & Jerk
        accel = np.abs(np.diff(speed)) * fps
        mean_accel = np.mean(accel)

        # Angular speed
        abs_ang_vel_mean = np.mean(np.abs(ang_vel))

        # Spatial exploration / center occupancy (distance from arena center ~200, 200)
        dist_from_center = np.sqrt((x - 200)**2 + (y - 200)**2)
        center_occupancy_frac = np.mean(dist_from_center < 100.0)  # within center circle

        # Total distance (px)
        total_dist_px = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

        session_rows.append({
            "session_id": sid,
            "animal_id": aid,
            "genotype": g,
            "total_frames": len(vdf),
            "duration_min": len(vdf) / (fps * 60.0),
            "mean_speed_px_s": mean_speed,
            "median_speed_px_s": median_speed,
            "p95_speed_px_s": p95_speed,
            "immobility_fraction": immobility_frac,
            "active_mean_speed_px_s": active_speed_mean,
            "mean_accel_px_s2": mean_accel,
            "abs_ang_vel_mean_rad_s": abs_ang_vel_mean,
            "center_occupancy_fraction": center_occupancy_frac,
            "total_distance_px": total_dist_px,
        })

    df_global = pd.DataFrame(session_rows)
    df_global.to_csv(d1_dir / "global_kinematics_per_session.csv", index=False)

    # Statistical comparison table
    stat_rows = []
    numeric_cols = [c for c in df_global.columns if c not in ["session_id", "animal_id", "genotype", "total_frames"]]

    for c in numeric_cols:
        mon_vals = df_global[df_global["genotype"] == "Monomer"][c].dropna()
        pff_vals = df_global[df_global["genotype"] == "PFF"][c].dropna()

        # t-test & Mann-Whitney U
        t_stat, t_pval = stats.ttest_ind(pff_vals, mon_vals, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(pff_vals, mon_vals, alternative="two-sided")

        # Effect sizes
        mon_mean, pff_mean = mon_vals.mean(), pff_vals.mean()
        mon_std, pff_std = mon_vals.std(), pff_vals.std()
        pooled_std = np.sqrt(((len(mon_vals)-1)*mon_std**2 + (len(pff_vals)-1)*pff_std**2) / (len(mon_vals) + len(pff_vals) - 2))
        cohens_d = (pff_mean - mon_mean) / (pooled_std + 1e-12)
        pct_change = ((pff_mean - mon_mean) / (mon_mean + 1e-12)) * 100.0

        stat_rows.append({
            "metric": c,
            "Monomer_mean": mon_mean,
            "Monomer_std": mon_std,
            "PFF_mean": pff_mean,
            "PFF_std": pff_std,
            "pct_change_PFF_vs_Monomer": pct_change,
            "cohens_d": cohens_d,
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "mannwhitney_u": u_stat,
            "mwu_pvalue": u_pval,
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats["t_qvalue_fdr"] = benjamini_hochberg(df_stats["t_pvalue"])
    df_stats["mwu_qvalue_fdr"] = benjamini_hochberg(df_stats["mwu_pvalue"])
    df_stats.to_csv(d1_dir / "global_kinematics_statistical_summary.csv", index=False)
    print(f"Saved {d1_dir / 'global_kinematics_statistical_summary.csv'}")

    # Plotting Grid
    plot_metrics = [
        ("mean_speed_px_s", "Mean Speed (px/s)"),
        ("immobility_fraction", "Immobility Fraction (<5 px/s)"),
        ("active_mean_speed_px_s", "Active Speed (px/s)"),
        ("mean_accel_px_s2", "Mean Acceleration (px/s²)"),
        ("abs_ang_vel_mean_rad_s", "Mean Angular Speed (|rad/s|)"),
        ("center_occupancy_fraction", "Center Arena Occupancy Fraction"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (m, label) in enumerate(plot_metrics):
        ax = axes[i]
        sns.boxplot(data=df_global, x="genotype", y=m, palette=palette, ax=ax, width=0.4, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_global, x="genotype", y=m, color="black", alpha=0.6, jitter=0.2, size=5, ax=ax)

        # Annotate p-value
        row = df_stats[df_stats["metric"] == m].iloc[0]
        pval = row["mwu_pvalue"]
        qval = row["mwu_qvalue_fdr"]
        sig_str = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.set_title(f"{label}\n(p={pval:.4f} [{sig_str}], q={qval:.4f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)

    plt.tight_layout()
    fig.savefig(d1_dir / "global_kinematics_boxplots.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d1_dir / 'global_kinematics_boxplots.png'}")
    return df_global, df_stats


# ==============================================================================
# DIMENSION 2: SYLLABLE REPERTOIRE & DIFFERENTIAL USAGE
# ==============================================================================
def analyze_syllable_usage(sessions, fps=80.0):
    print("\n--- [Dimension 2] Syllable Repertoire & Differential Usage ---")
    d2_dir = OUT_DIR / "02_syllable_usage"
    d2_dir.mkdir(parents=True, exist_ok=True)

    # Per-session syllable usage matrix (sessions x 100 syllables)
    sess_usage = []
    sess_meta = []

    for s in sessions:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id

        s_series = df["syllable"].dropna()
        if len(s_series) < 1000:
            continue

        counts = s_series.value_counts()
        total = counts.sum()
        freqs = np.zeros(100)
        for syll_id, count in counts.items():
            if 0 <= int(syll_id) < 100:
                freqs[int(syll_id)] = count / total

        sess_usage.append(freqs)

        # Entropy & Diversity metrics
        probs = freqs[freqs > 0]
        shannon_entropy = -(probs * np.log2(probs)).sum()
        part_ratio = 1.0 / (freqs**2).sum()

        sess_meta.append({
            "session_id": sid,
            "genotype": g,
            "shannon_entropy_bits": shannon_entropy,
            "participation_ratio": part_ratio,
        })

    usage_mat = np.array(sess_usage)  # N x 100
    df_meta = pd.DataFrame(sess_meta)

    # Save usage matrix CSV
    df_usage_all = pd.DataFrame(usage_mat, columns=[f"syllable_{i}" for i in range(100)])
    df_usage_all.insert(0, "genotype", df_meta["genotype"])
    df_usage_all.insert(0, "session_id", df_meta["session_id"])
    df_usage_all.to_csv(d2_dir / "syllable_usage_matrix_per_session.csv", index=False)

    # Differential usage per syllable statistics
    mon_mask = (df_meta["genotype"] == "Monomer").values
    pff_mask = (df_meta["genotype"] == "PFF").values

    diff_rows = []
    for syll in range(100):
        mon_vals = usage_mat[mon_mask, syll]
        pff_vals = usage_mat[pff_mask, syll]

        mon_mean = np.mean(mon_vals)
        pff_mean = np.mean(pff_vals)
        delta = pff_mean - mon_mean
        log2_fc = np.log2((pff_mean + 1e-5) / (mon_mean + 1e-5))

        # Statistical test
        u_stat, u_pval = stats.mannwhitneyu(pff_vals, mon_vals, alternative="two-sided")
        t_stat, t_pval = stats.ttest_ind(pff_vals, mon_vals, equal_var=False)

        diff_rows.append({
            "syllable": syll,
            "Monomer_mean_usage": mon_mean,
            "PFF_mean_usage": pff_mean,
            "usage_delta_PFF_minus_Monomer": delta,
            "log2_fold_change": log2_fc,
            "mannwhitney_u": u_stat,
            "pvalue": u_pval,
            "t_pvalue": t_pval,
        })

    df_diff = pd.DataFrame(diff_rows)
    df_diff["qvalue_fdr"] = benjamini_hochberg(df_diff["pvalue"])
    df_diff["significant_raw_p05"] = df_diff["pvalue"] < 0.05
    df_diff["significant_fdr_q10"] = df_diff["qvalue_fdr"] < 0.10
    df_diff = df_diff.sort_values("pvalue", ascending=True)
    df_diff.to_csv(d2_dir / "differential_syllable_usage_stats.csv", index=False)
    print(f"Saved {d2_dir / 'differential_syllable_usage_stats.csv'}")

    num_sig_raw = df_diff["significant_raw_p05"].sum()
    num_sig_fdr = df_diff["significant_fdr_q10"].sum()
    print(f"  Differential Syllables: {num_sig_raw}/100 significant at p<0.05 (uncorrected), {num_sig_fdr}/100 at FDR q<0.10")

    # 1. VOLCANO PLOT
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df_diff["log2_fold_change"], -np.log10(df_diff["pvalue"]), c="#888888", alpha=0.6, s=40, label="Non-significant")

    # Highlight significant
    sig_pff = df_diff[(df_diff["pvalue"] < 0.05) & (df_diff["log2_fold_change"] > 0)]
    sig_mon = df_diff[(df_diff["pvalue"] < 0.05) & (df_diff["log2_fold_change"] < 0)]

    ax.scatter(sig_pff["log2_fold_change"], -np.log10(sig_pff["pvalue"]), color=palette["PFF"], s=70, label="Upregulated in PFF (p<0.05)")
    ax.scatter(sig_mon["log2_fold_change"], -np.log10(sig_mon["pvalue"]), color=palette["Monomer"], s=70, label="Downregulated in PFF (p<0.05)")

    # Annotate top 5 differential syllables
    for _, row in df_diff.head(8).iterrows():
        ax.annotate(f"Syll {int(row['syllable'])}", (row["log2_fold_change"], -np.log10(row["pvalue"]) + 0.1),
                    fontsize=9, fontweight="bold", ha="center")

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.5, label="p = 0.05 threshold")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Log2 Fold Change (PFF / Monomer)", fontsize=12)
    ax.set_ylabel("-Log10 (p-value)", fontsize=12)
    ax.set_title("Volcano Plot: Syllable Usage Differential between PFF and Monomer", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig.savefig(d2_dir / "volcano_syllable_usage.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d2_dir / 'volcano_syllable_usage.png'}")

    # 2. TOP DIFFERENTIAL SYLLABLES BARCHART
    top_diff = pd.concat([df_diff[df_diff["log2_fold_change"] > 0].head(6), df_diff[df_diff["log2_fold_change"] < 0].head(6)]).sort_values("log2_fold_change")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [palette["Monomer"] if fc < 0 else palette["PFF"] for fc in top_diff["log2_fold_change"]]
    y_pos = np.arange(len(top_diff))
    bars = ax.barh(y_pos, top_diff["log2_fold_change"], color=colors, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Syllable {int(s)} (p={p:.3f})" for s, p in zip(top_diff["syllable"], top_diff["pvalue"])], fontsize=10)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Log2 Fold Change (PFF / Monomer)", fontsize=11, fontweight="bold")
    ax.set_title("Top Differential Behavioral Syllables (PFF vs Monomer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(d2_dir / "top_differential_syllables_barchart.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d2_dir / 'top_differential_syllables_barchart.png'}")

    # 3. REPERTOIRE DIVERSITY (Entropy & Participation Ratio)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(data=df_meta, x="genotype", y="shannon_entropy_bits", palette=palette, ax=ax1, width=0.4, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_meta, x="genotype", y="shannon_entropy_bits", color="black", alpha=0.6, jitter=0.2, ax=ax1)
    e_pval = stats.mannwhitneyu(df_meta[df_meta["genotype"]=="PFF"]["shannon_entropy_bits"], df_meta[df_meta["genotype"]=="Monomer"]["shannon_entropy_bits"]).pvalue
    ax1.set_title(f"Syllable Shannon Entropy\n(p = {e_pval:.4f})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Entropy (bits)")

    sns.boxplot(data=df_meta, x="genotype", y="participation_ratio", palette=palette, ax=ax2, width=0.4, boxprops=dict(alpha=0.7))
    sns.stripplot(data=df_meta, x="genotype", y="participation_ratio", color="black", alpha=0.6, jitter=0.2, ax=ax2)
    pr_pval = stats.mannwhitneyu(df_meta[df_meta["genotype"]=="PFF"]["participation_ratio"], df_meta[df_meta["genotype"]=="Monomer"]["participation_ratio"]).pvalue
    ax2.set_title(f"Participation Ratio (Effective Syllables)\n(p = {pr_pval:.4f})", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Effective Number of Syllables")

    plt.tight_layout()
    fig.savefig(d2_dir / "repertoire_entropy_and_diversity.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d2_dir / 'repertoire_entropy_and_diversity.png'}")

    # 4. PCA / UMAP of Syllable Usage Vectors
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(usage_mat)

    fig, ax = plt.subplots(figsize=(8, 6))
    for g, col in palette.items():
        mask = (df_meta["genotype"] == g).values
        ax.scatter(pca_proj[mask, 0], pca_proj[mask, 1], c=col, label=f"{g} (N={mask.sum()})", s=60, alpha=0.8, edgecolors="none")
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("PCA Projection of Animal Syllable Repertoires", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig.savefig(d2_dir / "pca_syllable_repertoire_projection.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d2_dir / 'pca_syllable_repertoire_projection.png'}")

    return df_diff, df_meta


# ==============================================================================
# DIMENSION 3: WITHIN-SYLLABLE MICRO-KINEMATICS
# ==============================================================================
def analyze_micro_kinematics(sessions, fps=80.0):
    print("\n--- [Dimension 3] Within-Syllable Micro-Kinematics ---")
    d3_dir = OUT_DIR / "03_micro_kinematics"
    d3_dir.mkdir(parents=True, exist_ok=True)

    # Collect speed and duration per syllable occurrence per session
    syll_records = []

    for s in sessions:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id

        # Valid slice
        vdf = df.dropna(subset=["syllable", "velocity_px_s"])
        if len(vdf) < 1000:
            continue

        # Group by contiguous syllable run
        s_arr = vdf["syllable"].values
        speed_arr = vdf["velocity_px_s"].values
        ang_arr = vdf["angular_velocity"].values if "angular_velocity" in vdf.columns else np.zeros_like(speed_arr)

        # Run-length grouping
        changes = np.concatenate(([True], s_arr[1:] != s_arr[:-1], [True]))
        idx_boundaries = np.where(changes)[0]

        for start_i, end_i in zip(idx_boundaries[:-1], idx_boundaries[1:]):
            syll = int(s_arr[start_i])
            dur_frames = end_i - start_i
            dur_ms = (dur_frames / fps) * 1000.0
            mean_run_speed = np.mean(speed_arr[start_i:end_i])
            mean_run_ang = np.mean(np.abs(ang_arr[start_i:end_i]))

            syll_records.append({
                "session_id": sid,
                "genotype": g,
                "syllable": syll,
                "duration_ms": dur_ms,
                "speed_px_s": mean_run_speed,
                "abs_ang_vel": mean_run_ang,
            })

    df_runs = pd.DataFrame(syll_records)

    # Per-session aggregated means for each syllable to do proper between-animal stats
    sess_syll_stats = df_runs.groupby(["session_id", "genotype", "syllable"]).agg({
        "speed_px_s": "mean",
        "duration_ms": "median",
        "abs_ang_vel": "mean",
    }).reset_index()

    micro_stats = []
    for syll in range(100):
        sub = sess_syll_stats[sess_syll_stats["syllable"] == syll]
        mon_speeds = sub[sub["genotype"] == "Monomer"]["speed_px_s"].dropna()
        pff_speeds = sub[sub["genotype"] == "PFF"]["speed_px_s"].dropna()
        mon_durs = sub[sub["genotype"] == "Monomer"]["duration_ms"].dropna()
        pff_durs = sub[sub["genotype"] == "PFF"]["duration_ms"].dropna()

        if len(mon_speeds) >= 5 and len(pff_speeds) >= 5:
            # Speed test
            t_sp_stat, t_sp_pval = stats.ttest_ind(pff_speeds, mon_speeds, equal_var=False)
            u_sp_stat, u_sp_pval = stats.mannwhitneyu(pff_speeds, mon_speeds, alternative="two-sided")
            # Duration test
            t_dur_stat, t_dur_pval = stats.ttest_ind(pff_durs, mon_durs, equal_var=False)
            u_dur_stat, u_dur_pval = stats.mannwhitneyu(pff_durs, mon_durs, alternative="two-sided")

            micro_stats.append({
                "syllable": syll,
                "Monomer_n_sessions": len(mon_speeds),
                "PFF_n_sessions": len(pff_speeds),
                "Monomer_speed_mean": mon_speeds.mean(),
                "PFF_speed_mean": pff_speeds.mean(),
                "speed_delta_px_s": pff_speeds.mean() - mon_speeds.mean(),
                "speed_pvalue": u_sp_pval,
                "Monomer_duration_median_ms": mon_durs.median(),
                "PFF_duration_median_ms": pff_durs.median(),
                "duration_delta_ms": pff_durs.median() - mon_durs.median(),
                "duration_pvalue": u_dur_pval,
            })

    df_micro = pd.DataFrame(micro_stats)
    df_micro["speed_qvalue_fdr"] = benjamini_hochberg(df_micro["speed_pvalue"])
    df_micro["duration_qvalue_fdr"] = benjamini_hochberg(df_micro["duration_pvalue"])
    df_micro = df_micro.sort_values("speed_pvalue", ascending=True)
    df_micro.to_csv(d3_dir / "micro_kinematics_within_syllable_stats.csv", index=False)
    print(f"Saved {d3_dir / 'micro_kinematics_within_syllable_stats.csv'}")

    # Plot top speed difference syllables
    top_speed_diff = df_micro.sort_values("speed_delta_px_s").head(6)
    top_speed_diff = pd.concat([top_speed_diff, df_micro.sort_values("speed_delta_px_s", ascending=False).head(6)]).sort_values("speed_delta_px_s")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [palette["Monomer"] if d < 0 else palette["PFF"] for d in top_speed_diff["speed_delta_px_s"]]
    y_pos = np.arange(len(top_speed_diff))
    ax.barh(y_pos, top_speed_diff["speed_delta_px_s"], color=colors, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Syllable {int(s)} (p={p:.3f})" for s, p in zip(top_speed_diff["syllable"], top_speed_diff["speed_pvalue"])], fontsize=10)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Within-Syllable Execution Speed Delta (PFF - Monomer) [px/s]", fontsize=11, fontweight="bold")
    ax.set_title("Within-Syllable Execution Speed Differences (PFF vs Monomer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(d3_dir / "within_syllable_speed_deltas.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d3_dir / 'within_syllable_speed_deltas.png'}")
    return df_micro


# ==============================================================================
# DIMENSION 4: BEHAVIORAL SYNTAX & TRANSITIONS
# ==============================================================================
def analyze_behavioral_syntax(sessions):
    print("\n--- [Dimension 4] Behavioral Syntax & Transition Bigrams ---")
    d4_dir = OUT_DIR / "04_syntax_and_motifs"
    d4_dir.mkdir(parents=True, exist_ok=True)

    # Bigram transition counts per session
    bigram_counts = {}
    sess_bigram_freqs = []

    for s in sessions:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id

        s_series = df["syllable"].dropna().values
        if len(s_series) < 1000:
            continue

        # Extract syllable sequence of onsets (collapse consecutive duplicates)
        collapsed = s_series[np.concatenate(([True], s_series[1:] != s_series[:-1]))]
        if len(collapsed) < 2:
            continue

        src = collapsed[:-1].astype(int)
        dst = collapsed[1:].astype(int)

        # Valid syllable IDs [0..99]
        valid_mask = (src >= 0) & (src < 100) & (dst >= 0) & (dst < 100)
        src, dst = src[valid_mask], dst[valid_mask]

        total_transitions = len(src)
        if total_transitions == 0:
            continue

        pairs = list(zip(src, dst))
        counts = pd.Series(pairs).value_counts()
        for pair, count in counts.items():
            bigram_counts[pair] = bigram_counts.get(pair, 0) + count

        sess_dict = {f"{p[0]}->{p[1]}": count / total_transitions for p, count in counts.items()}
        sess_dict["session_id"] = sid
        sess_dict["genotype"] = g
        sess_bigram_freqs.append(sess_dict)

    df_bigrams = pd.DataFrame(sess_bigram_freqs).fillna(0.0)

    # Filter to frequent bigrams (occurring at least 50 times across the dataset)
    frequent_pairs = [p for p, c in bigram_counts.items() if c >= 50]
    print(f"  Analyzing {len(frequent_pairs)} frequent behavioral bigram transitions...")

    trans_stats = []
    for p in frequent_pairs:
        col = f"{p[0]}->{p[1]}"
        if col not in df_bigrams.columns:
            continue

        mon_vals = df_bigrams[df_bigrams["genotype"] == "Monomer"][col]
        pff_vals = df_bigrams[df_bigrams["genotype"] == "PFF"][col]

        mon_mean = mon_vals.mean()
        pff_mean = pff_vals.mean()
        delta = pff_mean - mon_mean
        log2_fc = np.log2((pff_mean + 1e-5) / (mon_mean + 1e-5))

        u_stat, u_pval = stats.mannwhitneyu(pff_vals, mon_vals, alternative="two-sided")

        trans_stats.append({
            "from_syllable": p[0],
            "to_syllable": p[1],
            "transition": col,
            "total_occurrences": bigram_counts[p],
            "Monomer_mean_prob": mon_mean,
            "PFF_mean_prob": pff_mean,
            "prob_delta_PFF_minus_Monomer": delta,
            "log2_fold_change": log2_fc,
            "pvalue": u_pval,
        })

    df_syntax = pd.DataFrame(trans_stats)
    df_syntax["qvalue_fdr"] = benjamini_hochberg(df_syntax["pvalue"])
    df_syntax = df_syntax.sort_values("pvalue", ascending=True)
    df_syntax.to_csv(d4_dir / "differential_syntax_transitions.csv", index=False)
    print(f"Saved {d4_dir / 'differential_syntax_transitions.csv'}")

    num_sig_trans = (df_syntax["pvalue"] < 0.05).sum()
    print(f"  Differential Syntax: {num_sig_trans}/{len(df_syntax)} bigrams significant at p<0.05")

    # Plot top differential transitions
    top_trans = pd.concat([df_syntax[df_syntax["log2_fold_change"] > 0].head(6), df_syntax[df_syntax["log2_fold_change"] < 0].head(6)]).sort_values("log2_fold_change")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [palette["Monomer"] if d < 0 else palette["PFF"] for d in top_trans["log2_fold_change"]]
    y_pos = np.arange(len(top_trans))
    ax.barh(y_pos, top_trans["log2_fold_change"], color=colors, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Syll {r['from_syllable']} → {r['to_syllable']} (p={r['pvalue']:.3f})" for _, r in top_trans.iterrows()], fontsize=10)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Log2 Fold Change in Transition Probability (PFF / Monomer)", fontsize=11, fontweight="bold")
    ax.set_title("Top Differential Behavioral Transitions / Syntax (PFF vs Monomer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(d4_dir / "top_differential_syntax_transitions.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d4_dir / 'top_differential_syntax_transitions.png'}")
    return df_syntax


# ==============================================================================
# DIMENSION 5: TEMPORAL DYNAMICS & FATIGUE OVER SESSION TIME
# ==============================================================================
def analyze_temporal_dynamics(sessions, fps=80.0, max_minutes=45):
    print("\n--- [Dimension 5] Temporal Dynamics & Fatigue Across Session Time ---")
    d5_dir = OUT_DIR / "05_temporal_dynamics"
    d5_dir.mkdir(parents=True, exist_ok=True)

    bin_size_frames = int(60.0 * fps)  # 1-minute bins
    temporal_rows = []

    for s in sessions:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id

        vdf = df.dropna(subset=["velocity_px_s", "syllable"])
        if len(vdf) < 1000:
            continue

        n_bins = min(max_minutes, len(vdf) // bin_size_frames)
        for b in range(n_bins):
            start = b * bin_size_frames
            end = start + bin_size_frames
            chunk = vdf.iloc[start:end]

            # Speed
            speed_m = chunk["velocity_px_s"].mean()
            immobile_m = (chunk["velocity_px_s"] < 5.0).mean()

            # Syllable switching rate (transitions per min)
            s_arr = chunk["syllable"].values
            transitions = (s_arr[1:] != s_arr[:-1]).sum()

            temporal_rows.append({
                "session_id": sid,
                "genotype": g,
                "minute": b + 1,
                "speed_px_s": speed_m,
                "immobility_frac": immobile_m,
                "syllable_transitions_per_min": transitions,
            })

    df_temp = pd.DataFrame(temporal_rows)
    df_temp.to_csv(d5_dir / "temporal_binned_per_session.csv", index=False)

    # Group summary across minutes
    summary = df_temp.groupby(["minute", "genotype"]).agg({
        "speed_px_s": ["mean", "sem", "count"],
        "immobility_frac": ["mean", "sem"],
        "syllable_transitions_per_min": ["mean", "sem"],
    }).reset_index()
    summary.to_csv(d5_dir / "temporal_dynamics_summary.csv", index=False)
    print(f"Saved {d5_dir / 'temporal_dynamics_summary.csv'}")

    # Line plots over session time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for g, col in palette.items():
        sub = df_temp[df_temp["genotype"] == g]
        sns.lineplot(data=sub, x="minute", y="speed_px_s", color=col, label=g, ax=ax1, errorbar="se", linewidth=2.5)
        sns.lineplot(data=sub, x="minute", y="immobility_frac", color=col, label=g, ax=ax2, errorbar="se", linewidth=2.5)
        sns.lineplot(data=sub, x="minute", y="syllable_transitions_per_min", color=col, label=g, ax=ax3, errorbar="se", linewidth=2.5)

    ax1.set_title("Locomotor Speed Progression Over Session", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Speed (px/s)", fontsize=11)
    ax1.legend(loc="upper right")

    ax2.set_title("Immobility Fraction (<5 px/s) Over Session", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Immobility Fraction", fontsize=11)
    ax2.legend(loc="upper left")

    ax3.set_title("Syllable Transition / Switching Rate Over Session", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Session Time (Minutes)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Syllable Switches / Min", fontsize=11)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(d5_dir / "temporal_dynamics_progression_over_time.png", dpi=300)
    plt.close(fig)
    print(f"Saved {d5_dir / 'temporal_dynamics_progression_over_time.png'}")
    return df_temp


# ==============================================================================
# MAIN ORCHESTRATOR & REPORT SYNTHESIS
# ==============================================================================
def main():
    global DATA_PKL, OUT_DIR
    parser = argparse.ArgumentParser(description="Karen Genotype Behavioral Comparison")
    parser.add_argument("--data-pkl", type=str, default=str(DEFAULT_DATA_PKL), help="Path to input sessions pickle")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Camera framerate (default: 60.0)")
    args = parser.parse_args()

    DATA_PKL = Path(args.data_pkl)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    sessions = load_and_filter_sessions()

    df_global, df_global_stats = analyze_global_kinematics(sessions, fps=fps)
    df_diff_sylls, df_entropy = analyze_syllable_usage(sessions, fps=fps)
    df_micro = analyze_micro_kinematics(sessions, fps=fps)
    df_syntax = analyze_behavioral_syntax(sessions)
    df_temp = analyze_temporal_dynamics(sessions, fps=fps)

    # Synthesize comprehensive Markdown Report
    report_path = OUT_DIR / "KAREN_GENOTYPE_BEHAVIORAL_REPORT.md"
    print(f"\nWriting comprehensive biological synthesis report to {report_path}...")

    sig_global = df_global_stats[df_global_stats["mwu_pvalue"] < 0.05]
    sig_sylls = df_diff_sylls[df_diff_sylls["pvalue"] < 0.05]
    sig_micro_sp = df_micro[df_micro["speed_pvalue"] < 0.05]
    sig_syntax = df_syntax[df_syntax["pvalue"] < 0.05]

    with open(report_path, "w") as f:
        f.write("# Karen Dataset Genotype Behavioral Analysis Report\n")
        f.write("**Comparison**: PFF (Parkinson's Disease Alpha-Synuclein Model, N=56) vs. Monomer (Control, N=43)\n\n")

        f.write("## Executive Summary\n")
        f.write(f"- **Global Kinematics**: Found **{len(sig_global)} / {len(df_global_stats)}** global locomotor/spatial metrics showing significant differences at p < 0.05.\n")
        f.write(f"- **Syllable Repertoire**: Found **{len(sig_sylls)} / 100** syllables with significant differential occupancy (p < 0.05).\n")
        f.write(f"- **Within-Syllable Micro-Kinematics**: Found **{len(sig_micro_sp)}** syllables with significant execution speed modulation.\n")
        f.write(f"- **Behavioral Syntax & Transitions**: Found **{len(sig_syntax)}** bigram transitions significantly altered between genotypes.\n\n")

        f.write("## 1. Global Locomotor & Spatial Kinematics\n")
        f.write("| Metric | Monomer (Mean ± Std) | PFF (Mean ± Std) | % Change | Cohen's d | p-value (MWU) | FDR q-value |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in df_global_stats.iterrows():
            f.write(f"| **{r['metric']}** | {r['Monomer_mean']:.2f} ± {r['Monomer_std']:.2f} | {r['PFF_mean']:.2f} ± {r['PFF_std']:.2f} | {r['pct_change_PFF_vs_Monomer']:+.1f}% | {r['cohens_d']:+.2f} | **{r['mwu_pvalue']:.4f}** | {r['mwu_qvalue_fdr']:.4f} |\n")
        f.write("\n\n")

        f.write("## 2. Top Differential Behavioral Syllables (p < 0.05)\n")
        f.write("| Syllable ID | Monomer Usage | PFF Usage | Log2 Fold Change | p-value (MWU) | Direction |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in sig_sylls.head(15).iterrows():
            direction = "🔺 Upregulated in PFF" if r["log2_fold_change"] > 0 else "🔻 Downregulated in PFF"
            f.write(f"| **Syllable {int(r['syllable'])}** | {r['Monomer_mean_usage']*100:.2f}% | {r['PFF_mean_usage']*100:.2f}% | {r['log2_fold_change']:+.2f} | **{r['pvalue']:.4f}** | {direction} |\n")
        f.write("\n\n")

        f.write("## 3. Top Altered Behavioral Syntax / Transitions\n")
        f.write("| Transition | Monomer Prob | PFF Prob | Log2 Fold Change | p-value (MWU) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in sig_syntax.head(10).iterrows():
            f.write(f"| **Syllable {r['from_syllable']} → {r['to_syllable']}** | {r['Monomer_mean_prob']*100:.3f}% | {r['PFF_mean_prob']*100:.3f}% | {r['log2_fold_change']:+.2f} | **{r['pvalue']:.4f}** |\n")
        f.write("\n")

    print(f"Analysis complete. Report generated at {report_path}")


if __name__ == "__main__":
    main()
