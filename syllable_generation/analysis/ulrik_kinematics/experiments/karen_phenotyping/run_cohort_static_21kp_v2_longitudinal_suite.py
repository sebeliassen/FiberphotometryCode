#!/usr/bin/env python3
"""
run_cohort_static_21kp_v2_longitudinal_suite.py

Comprehensive longitudinal phenotyping and cohort-split analysis suite
for the Cohort Static 21kp V2 Seed 101 model.

Loads the unified session pickle:
  sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl (60 FPS)

Produces publication-grade, space-efficient aggregated figures & tables in:
  07_longitudinal_cohorts/
  1. Master 6-Panel Timecourse Grid (Speed, Sprint v95, Jerk, Bout Dur, Immobility, Syntax Entropy)
  2. Bout Dynamics & Freezing Timecourse (Bout Dur, Immobility, Initiation Rate)
  3. Action Syntax & Repertoire Entropy Timecourse (H_cond, H_rep, Switch Rate)
  4. Longitudinal Progression with Animal Traces & L-Dopa Rescue Bar Comparison
  5. High-Density Syllable Differences Across Cohorts Heatmap (log2FC per wave with significance stars)
  6. Top Differential Syllables Longitudinal Trajectories Grid
  7. Exported CSVs: session data, condition summaries, and per-syllable cohort metrics.
  8. JSON export of per-syllable cohort stats for the interactive Labeling Gallery & Catalog.
"""

import sys
import os
import argparse
import json
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}
condition_order = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]
condition_labels = [
    "3w PI\n(Baseline)",
    "8w PI\n(Mid)",
    "16w PI\n(Peak)",
    "24w PI\n(Late)",
    "L-Dopa\n(+Benserazide)"
]
condition_labels_short = ["3w PI", "8w PI", "16w PI", "24w PI", "L-Dopa"]


def get_p_star(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "#"
    return "ns"


def plot_timecourse_grid(df_data, panels, nrows, ncols, figsize, out_path, suptitle):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    x_pos = np.arange(len(condition_order))

    for idx, (col, title, ylabel) in enumerate(panels):
        ax = axes_flat[idx]

        # Drug rescue shading
        ax.axvspan(3.5, 4.5, color="#f1f5f9", alpha=0.9, zorder=0)
        ax.axvline(x=3.5, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)

        ax.text(
            4.0, 0.93, "L-Dopa Rescue",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            color="#334155", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9, zorder=5)
        )

        # Plot individual traces and group means
        for g, color in palette.items():
            sub = df_data[df_data["genotype"] == g]
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map and not np.isnan(a_map[c])]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.1, zorder=2)

            means = sub.groupby("condition")[col].mean().reindex(condition_order).values
            sems = sub.groupby("condition")[col].sem().reindex(condition_order).values

            ax.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)" if idx == 0 else None,
                        marker="o", markersize=7.0, linewidth=2.5, capsize=4, zorder=4)
            ax.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":",
                        marker="o", markersize=7.0, linewidth=1.8, capsize=4, zorder=4)

        # Annotate per-timepoint stats
        for i, cond in enumerate(condition_order):
            sub_c = df_data[df_data["condition"] == cond]
            m_vals = sub_c[sub_c["genotype"] == "Monomer"][col].dropna()
            p_vals = sub_c[sub_c["genotype"] == "PFF"][col].dropna()
            if len(m_vals) >= 2 and len(p_vals) >= 2:
                try:
                    _, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                    _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                    min_p = min(p_t, p_u)
                except Exception:
                    min_p = 1.0

                y_top = max(
                    m_vals.mean() + (m_vals.sem() if len(m_vals) > 1 else 0),
                    p_vals.mean() + (p_vals.sem() if len(p_vals) > 1 else 0)
                )

                if min_p < 0.05:
                    star = get_p_star(min_p)
                    ax.text(i, y_top * 1.08, f"{star}\np={min_p:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#d95f02")
                elif min_p < 0.10:
                    ax.text(i, y_top * 1.06, f"p={min_p:.3f}\n(trend)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#b45309")
                else:
                    ax.text(i, y_top * 1.05, "ns", ha="center", va="bottom", fontsize=7.0, color="#94a3b8")

        # Untreated 3w-24w stats
        untreated = df_data[df_data["condition"] != "L_dopa"]
        m_all = untreated[untreated["genotype"] == "Monomer"][col].dropna()
        p_all = untreated[untreated["genotype"] == "PFF"][col].dropna()
        if len(m_all) >= 2 and len(p_all) >= 2:
            try:
                _, p_all_t = stats.ttest_ind(p_all, m_all, equal_var=False)
                _, p_all_u = stats.mannwhitneyu(p_all, m_all)
                min_all_p = min(p_all_t, p_all_u)
                star_all = get_p_star(min_all_p)
            except Exception:
                min_all_p, star_all = 1.0, "ns"

            ax.text(
                0.03, 0.04, f"Untreated (3w-24w):\n{star_all} p = {min_all_p:.4f}",
                transform=ax.transAxes,
                fontsize=8.0, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9)
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_labels, fontsize=9.0, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Post-Injection", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0] * 0.95 if y_lims[0] > 0 else 0, y_lims[1] * 1.25)

    if axes_flat[0].get_legend_handles_labels()[0]:
        axes_flat[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper left")

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run 21kp V2 Longitudinal & Cohort Phenotyping Suite")
    parser.add_argument("--data-pkl", type=str,
                        default=str(REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl"))
    parser.add_argument("--out-dir", type=str,
                        default=str(REPO_ROOT / "syllable_generation/analysis/ulrik_kinematics/output/cohort_static_21kp_v2_seed101_phenotyping/07_longitudinal_cohorts"))
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    data_pkl = Path(args.data_pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    print("=" * 80)
    print("📈 RUNNING COHORT STATIC 21kp V2 SEED 101 LONGITUDINAL PHENOTYPING SUITE")
    print(f"Data Pickle: {data_pkl}")
    print(f"Output Dir:  {out_dir}")
    print(f"Framerate:   {fps} FPS")
    print("=" * 80)

    print("\n[1/5] Loading serialized sessions...", flush=True)
    with open(data_pkl, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} sessions.")

    # -------------------------------------------------------------
    # 1. EXTRACT SESSION KINEMATICS & SYLLABLE USAGE
    # -------------------------------------------------------------
    print("\n[2/5] Extracting session kinematics, syntax entropy, and per-syllable distributions...", flush=True)
    session_records = []
    syll_session_records = []  # (session_id, animal_id, condition, genotype, syll_id, usage_pct, count, dur_mean)

    for s in sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        aid = str(md.get("animal_id", ""))
        cond = md.get("condition", "Unknown")
        drug = md.get("drug", "None")
        is_drug = md.get("is_drug_session", False)
        weeks_pi = md.get("weeks_post_injection", np.nan)
        ts_str = md.get("timestamp", "")
        date_str = ts_str.split("T")[0] if "T" in ts_str else ts_str

        if g not in ["PFF", "Monomer"] or cond not in condition_order:
            continue

        if not hasattr(s, "all_data_df") or s.all_data_df is None:
            continue

        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue

        speed = df["velocity_px_s"].values
        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        norm_jerk = np.mean(np.abs(jerk)) / (np.mean(speed) + 1e-3)

        # Active movement bouts (>5 px/s)
        is_active = (speed > 5.0).astype(int)
        active_changes = np.diff(np.concatenate(([0], is_active, [0])))
        starts = np.where(active_changes == 1)[0]
        ends = np.where(active_changes == -1)[0]
        bout_durations = (ends - starts) / fps
        mean_bout_dur = np.mean(bout_durations) if len(bout_durations) > 0 else np.nan
        immobility_pct = np.mean(speed < 5.0) * 100.0
        session_min = len(speed) / (fps * 60.0)
        bouts_per_min = len(bout_durations) / session_min

        # Syllables & Transitions
        sylls = df["syllable"].values.astype(int)
        changes = np.concatenate(([True], sylls[1:] != sylls[:-1]))
        collapsed = sylls[changes]
        switch_rate_hz = len(collapsed) / (len(speed) / fps)

        if len(collapsed) >= 50:
            src = collapsed[:-1]
            dst = collapsed[1:]
            pairs = pd.Series(list(zip(src, dst))).value_counts()
            p_pairs = pairs / pairs.sum()
            p_src = pd.Series(src).value_counts() / len(src)
            h_cond = 0.0
            for (i, j), pij in p_pairs.items():
                pi = p_src.get(i, 1e-12)
                h_cond -= pij * np.log2((pij / pi) + 1e-12)
            p_rep = pd.Series(collapsed).value_counts(normalize=True).values
            h_rep = -np.sum(p_rep * np.log2(p_rep + 1e-12))
        else:
            h_cond = np.nan
            h_rep = np.nan

        session_records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "condition": cond,
            "condition_idx": condition_order.index(cond),
            "date": date_str,
            "drug": drug,
            "is_drug_session": is_drug,
            "weeks_post_injection": weeks_pi,
            "genotype": g,
            "mean_speed_px_s": np.mean(speed),
            "p95_speed_px_s": np.percentile(speed, 95),
            "p99_speed_px_s": np.percentile(speed, 99),
            "norm_jerk": norm_jerk,
            "mean_bout_dur_s": mean_bout_dur,
            "immobility_pct": immobility_pct,
            "bouts_per_min": bouts_per_min,
            "switch_rate_hz": switch_rate_hz,
            "transition_entropy_rate": h_cond,
            "repertoire_entropy": h_rep,
        })

        # Per-syllable stats in this session
        u_sylls, u_counts = np.unique(sylls, return_counts=True)
        tot_frames = len(sylls)
        # Bout starts & durations per syllable
        syll_starts = np.where(changes)[0]
        syll_ends = np.concatenate((syll_starts[1:], [len(sylls)]))
        syll_bouts_id = sylls[syll_starts]
        syll_bouts_dur = (syll_ends - syll_starts) / fps

        bouts_df = pd.DataFrame({"syll": syll_bouts_id, "dur": syll_bouts_dur})
        dur_means = bouts_df.groupby("syll")["dur"].mean().to_dict()
        dur_counts = bouts_df.groupby("syll")["dur"].count().to_dict()

        for s_id, s_cnt in zip(u_sylls, u_counts):
            usage_pct = (s_cnt / tot_frames) * 100.0
            syll_session_records.append({
                "session_id": s.session_id,
                "animal_id": aid,
                "condition": cond,
                "genotype": g,
                "syllable": int(s_id),
                "usage_pct": usage_pct,
                "frame_count": int(s_cnt),
                "bout_count": dur_counts.get(s_id, 0),
                "mean_duration_s": dur_means.get(s_id, np.nan)
            })

    df_sessions = pd.DataFrame(session_records)
    df_sessions.to_csv(out_dir / "longitudinal_data_per_session.csv", index=False)
    print(f"Saved {out_dir / 'longitudinal_data_per_session.csv'} ({len(df_sessions)} sessions)")

    # Condition & Genotype Summary Table
    cond_summary = df_sessions.groupby(["condition", "genotype"]).agg({
        "mean_speed_px_s": ["mean", "std", "count"],
        "p95_speed_px_s": ["mean", "std"],
        "norm_jerk": ["mean", "std"],
        "mean_bout_dur_s": ["mean", "std"],
        "immobility_pct": ["mean", "std"],
        "transition_entropy_rate": ["mean", "std"],
    }).reindex(index=condition_order, level=0).reset_index()
    cond_summary.to_csv(out_dir / "longitudinal_condition_summary.csv", index=False)
    print(f"Saved {out_dir / 'longitudinal_condition_summary.csv'}")

    df_syll_sessions = pd.DataFrame(syll_session_records)

    # -------------------------------------------------------------
    # 2. GENERATE LONGITUDINAL MULTI-PANEL KINEMATIC FIGURES
    # -------------------------------------------------------------
    print("\n[3/5] Generating longitudinal multi-panel figures...", flush=True)

    # 2A. MASTER 6-PANEL FIGURE
    master_panels = [
        ("mean_speed_px_s", "1. Locomotor Speed (px/s)", "Speed (px/s)"),
        ("p95_speed_px_s", "2. Peak Burst Sprint Speed ($v_{95}$)", "Burst Speed (px/s)"),
        ("norm_jerk", "3. Movement Jerkiness (Norm. Jerk)", "Norm. Jerk $\\langle|\\dddot{x}|\\rangle/\\langle v\\rangle$"),
        ("mean_bout_dur_s", "4. Active Movement Bout Duration (s)", "Continuous Bout Length (s)"),
        ("immobility_pct", "5. Freezing / Immobility (<5 px/s)", "Stationary Time (%)"),
        ("transition_entropy_rate", "6. Transition Entropy Rate $H(S_{t+1}|S_t)$", "Syntax Disorder (bits)"),
    ]
    plot_timecourse_grid(
        df_sessions, master_panels, 2, 3, (18, 10),
        out_dir / "karen_longitudinal_6panel_master.png",
        "Cohort Static 21kp V2: Longitudinal Kinematic & Behavioral Phenotyping (3w -> 24w PI + L-Dopa)"
    )

    # 2B. BOUT DYNAMICS & FREEZING 3-PANEL
    bout_panels = [
        ("mean_bout_dur_s", "A. Active Movement Bout Duration (s)", "Continuous Run Length (s)"),
        ("immobility_pct", "B. Freezing / Immobility (<5 px/s)", "Stationary / Freezing (%)"),
        ("bouts_per_min", "C. Movement Initiation Rate (Starts/min)", "Bout Frequency (starts/min)"),
    ]
    plot_timecourse_grid(
        df_sessions, bout_panels, 1, 3, (18, 5.5),
        out_dir / "karen_bout_dynamics_timecourse.png",
        "Cohort Static 21kp V2: Bout Dynamics & Freezing Progression"
    )

    # 2C. SYNTAX & ENTROPY 3-PANEL
    syntax_panels = [
        ("transition_entropy_rate", "A. Transition Entropy Rate $H(S_{t+1}|S_t)$", "Syntax Disorder (bits/step)"),
        ("repertoire_entropy", "B. Behavioral Repertoire Entropy $H(S)$", "Repertoire Diversity (bits)"),
        ("switch_rate_hz", "C. Syllable Switching Rate (Hz)", "Switching Frequency (Hz)"),
    ]
    plot_timecourse_grid(
        df_sessions, syntax_panels, 1, 3, (18, 5.5),
        out_dir / "karen_syntax_entropy_timecourse.png",
        "Cohort Static 21kp V2: Behavioral Syntax & Entropy Dynamics"
    )

    # 2D. 4-PANEL EXECUTIVE SUMMARY WITH SPAGHETTI TRACES
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    exec_metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)", "Speed (px/s)", axes[0, 0]),
        ("p95_speed_px_s", "Peak Burst Sprint Speed (px/s)", "Top 5% Speed (px/s)", axes[0, 1]),
        ("norm_jerk", "Movement Jerkiness (Norm. Jerk)", "Norm. Jerk $\\langle|\\dddot{x}|\\rangle/\\langle v\\rangle$", axes[1, 0]),
        ("transition_entropy_rate", "Transition Entropy Rate (bits)", "Entropy (bits)", axes[1, 1]),
    ]
    x_pos = np.arange(len(condition_order))

    for col, title, ylabel, ax in exec_metrics:
        ax.axvspan(3.5, 4.5, color="#f1f5f9", alpha=0.9, zorder=0)
        ax.axvline(x=3.5, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)
        ax.text(
            4.0, 0.93, "L-Dopa Rescue",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            color="#334155", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9, zorder=5)
        )

        for g, color in palette.items():
            sub = df_sessions[df_sessions["genotype"] == g]
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map and not np.isnan(a_map[c])]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.1, zorder=2)

            means = sub.groupby("condition")[col].mean().reindex(condition_order).values
            sems = sub.groupby("condition")[col].sem().reindex(condition_order).values

            ax.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)",
                        marker="o", markersize=7.5, linewidth=2.5, capsize=4, zorder=4)
            ax.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":",
                        marker="o", markersize=7.5, linewidth=1.8, capsize=4, zorder=4)

        for i, cond in enumerate(condition_order):
            sub_c = df_sessions[df_sessions["condition"] == cond]
            m_vals = sub_c[sub_c["genotype"] == "Monomer"][col].dropna()
            p_vals = sub_c[sub_c["genotype"] == "PFF"][col].dropna()
            if len(m_vals) >= 2 and len(p_vals) >= 2:
                _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                y_top = max(
                    m_vals.mean() + (m_vals.sem() if len(m_vals) > 1 else 0),
                    p_vals.mean() + (p_vals.sem() if len(p_vals) > 1 else 0)
                )
                if p_u < 0.05:
                    ax.text(i, y_top * 1.08, f"{get_p_star(p_u)}\np={p_u:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#d95f02")
                elif p_u < 0.10:
                    ax.text(i, y_top * 1.06, f"p={p_u:.3f}\n(trend)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#b45309")
                else:
                    ax.text(i, y_top * 1.05, "ns", ha="center", va="bottom", fontsize=7.0, color="#94a3b8")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_labels, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "longitudinal_cohort_progression.png", dpi=300)
    plt.close(fig)
    print(f"Saved {out_dir / 'longitudinal_cohort_progression.png'}")

    # 2E. L-DOPA RESCUE SPECIFIC COMPARISON (24w vs L-Dopa)
    fig_rescue, axes_r = plt.subplots(1, 3, figsize=(16, 5.2))
    rescue_metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)", axes_r[0]),
        ("p95_speed_px_s", "Peak Sprint Speed (px/s)", axes_r[1]),
        ("norm_jerk", "Movement Jerkiness (Norm. Jerk)", axes_r[2]),
    ]
    rescue_df = df_sessions[df_sessions["condition"].isin(["24w_PI", "L_dopa"])].copy()

    for col, title, ax in rescue_metrics:
        sns.barplot(data=rescue_df, x="condition", y=col, hue="genotype", palette=palette, ax=ax, capsize=0.1, err_kws={"linewidth": 1.5})
        sns.stripplot(data=rescue_df, x="condition", y=col, hue="genotype", palette=palette, dodge=True, ax=ax, alpha=0.7, jitter=0.2, edgecolor="black", linewidth=0.5)
        ax.set_title(f"L-Dopa Rescue: {title}", fontsize=11, fontweight="bold")
        ax.set_xticklabels(["24w PI (Pre-Drug Baseline)", "L-Dopa + Benserazide"], fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Genotype", frameon=True, facecolor="white")

    plt.tight_layout()
    fig_rescue.savefig(out_dir / "ldopa_rescue_comparison.png", dpi=300)
    plt.close(fig_rescue)
    print(f"Saved {out_dir / 'ldopa_rescue_comparison.png'}")

    # -------------------------------------------------------------
    # 3. COMPUTE COHORT-SPLIT SYLLABLE METRICS & LOG2FC
    # -------------------------------------------------------------
    print("\n[4/5] Computing per-syllable stats across all cohorts (3w, 8w, 16w, 24w, L-Dopa)...", flush=True)
    all_syll_ids = sorted(df_syll_sessions["syllable"].unique())

    # Build per-syllable, per-condition, per-genotype matrix
    syll_cohort_records = []
    syll_cohort_dict = {}

    for s_id in all_syll_ids:
        sub_s = df_syll_sessions[df_syll_sessions["syllable"] == s_id]

        s_entry = {
            "syllable_id": int(s_id),
            "waves": {}
        }

        # Overall untreated
        untreated_sub = sub_s[sub_s["condition"] != "L_dopa"]
        pff_all = untreated_sub[untreated_sub["genotype"] == "PFF"]["usage_pct"].values
        mono_all = untreated_sub[untreated_sub["genotype"] == "Monomer"]["usage_pct"].values
        mean_pff_all = float(np.mean(pff_all)) if len(pff_all) > 0 else 0.0
        mean_mono_all = float(np.mean(mono_all)) if len(mono_all) > 0 else 0.0
        log2fc_all = float(np.log2((mean_pff_all + 1e-4) / (mean_mono_all + 1e-4)))
        p_val_all = float(stats.mannwhitneyu(pff_all, mono_all)[1]) if (len(pff_all) > 0 and len(mono_all) > 0) else 1.0

        s_entry["untreated_overall"] = {
            "mean_pff": mean_pff_all,
            "mean_monomer": mean_mono_all,
            "log2fc": log2fc_all,
            "p_val": p_val_all,
            "star": get_p_star(p_val_all)
        }

        for cond in condition_order:
            cond_sub = sub_s[sub_s["condition"] == cond]
            pff_vals = cond_sub[cond_sub["genotype"] == "PFF"]["usage_pct"].values
            mono_vals = cond_sub[cond_sub["genotype"] == "Monomer"]["usage_pct"].values

            mean_pff = float(np.mean(pff_vals)) if len(pff_vals) > 0 else 0.0
            sem_pff = float(stats.sem(pff_vals)) if len(pff_vals) > 1 else 0.0
            mean_mono = float(np.mean(mono_vals)) if len(mono_vals) > 0 else 0.0
            sem_mono = float(stats.sem(mono_vals)) if len(mono_vals) > 1 else 0.0

            log2fc = float(np.log2((mean_pff + 1e-4) / (mean_mono + 1e-4)))
            delta_pct = float(mean_pff - mean_mono)

            if len(pff_vals) >= 2 and len(mono_vals) >= 2:
                try:
                    _, p_u = stats.mannwhitneyu(pff_vals, mono_vals)
                except Exception:
                    p_u = 1.0
            else:
                p_u = 1.0

            star = get_p_star(p_u)

            s_entry["waves"][cond] = {
                "mean_pff": mean_pff,
                "sem_pff": sem_pff,
                "mean_monomer": mean_mono,
                "sem_mono": sem_mono,
                "log2fc": log2fc,
                "delta_pct": delta_pct,
                "p_val": float(p_u),
                "star": star
            }

            syll_cohort_records.append({
                "syllable": int(s_id),
                "condition": cond,
                "mean_pff_usage": mean_pff,
                "sem_pff_usage": sem_pff,
                "mean_mono_usage": mean_mono,
                "sem_mono_usage": sem_mono,
                "log2fc_pff_mono": log2fc,
                "delta_usage_pct": delta_pct,
                "p_value": float(p_u),
                "significance": star
            })

        syll_cohort_dict[str(s_id)] = s_entry

    df_syll_cohorts = pd.DataFrame(syll_cohort_records)
    df_syll_cohorts.to_csv(out_dir / "syllable_differences_across_cohorts.csv", index=False)
    print(f"Saved {out_dir / 'syllable_differences_across_cohorts.csv'}")

    with open(out_dir / "syllable_cohort_breakdown.json", "w") as f:
        json.dump(syll_cohort_dict, f, indent=2)
    print(f"Saved {out_dir / 'syllable_cohort_breakdown.json'}")

    # -------------------------------------------------------------
    # 4. GENERATE AGGREGATED SYLLABLE HEATMAP & SMALL-MULTIPLES
    # -------------------------------------------------------------
    print("\n[5/5] Generating space-efficient aggregated syllable figures...", flush=True)

    # 4A. HIGH-DENSITY HEATMAP OF SYLLABLE DIFFERENCES (LOG2FC) ACROSS COHORTS
    # Filter to active syllables (overall usage > 0.05% or at least 1 significant wave)
    piv_log2fc = df_syll_cohorts.pivot(index="syllable", columns="condition", values="log2fc_pff_mono")[condition_order]
    piv_pvals = df_syll_cohorts.pivot(index="syllable", columns="condition", values="p_value")[condition_order]

    # Calculate overall usage prevalence per syllable
    overall_usage = df_syll_sessions.groupby("syllable")["usage_pct"].mean()
    active_sylls = [s for s in all_syll_ids if (overall_usage.get(s, 0) >= 0.10 or (piv_pvals.loc[s] < 0.05).any())]

    # Sort active syllables by 16w PI (peak deficit) fold change
    if "16w_PI" in piv_log2fc.columns:
        active_sylls = sorted(active_sylls, key=lambda s: piv_log2fc.loc[s, "16w_PI"], reverse=True)

    heat_data = piv_log2fc.loc[active_sylls]
    p_data = piv_pvals.loc[active_sylls]

    # Create annot text matrix (showing effect size and significance stars)
    annot_matrix = []
    for s in active_sylls:
        row_txt = []
        for c in condition_order:
            fc = heat_data.loc[s, c]
            pval = p_data.loc[s, c]
            star = get_p_star(pval)
            if star != "ns":
                row_txt.append(f"{fc:+.2f}\n{star}")
            else:
                row_txt.append(f"{fc:+.2f}")
        annot_matrix.append(row_txt)

    fig_h, ax_h = plt.subplots(figsize=(10, max(12, len(active_sylls) * 0.28)))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        heat_data,
        annot=np.array(annot_matrix),
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-2.0,
        vmax=2.0,
        cbar_kws={"label": "$\log_2$ Fold Change (PFF / Monomer)", "shrink": 0.6},
        ax=ax_h,
        linewidths=0.5,
        linecolor="#e2e8f0"
    )

    ax_h.set_title("Longitudinal Syllable Repertoire Dynamics Across Cohorts\n($\log_2$ Fold Change PFF vs Monomer, * p<0.05, ** p<0.01, *** p<0.001)",
                   fontsize=12, fontweight="bold", pad=15)
    ax_h.set_xticklabels(condition_labels_short, fontsize=11, fontweight="bold")
    ax_h.set_ylabel("Syllable ID (Ordered by 16w PI Peak Deficit)", fontsize=11, fontweight="bold")
    ax_h.set_xlabel("Experimental Cohort / Wave", fontsize=11, fontweight="bold")
    plt.tight_layout()

    heatmap_path = out_dir / "syllable_differences_across_cohorts_heatmap.png"
    fig_h.savefig(heatmap_path, dpi=300)
    plt.close(fig_h)
    print(f"Saved {heatmap_path.name} ({len(active_sylls)} active syllables)")

    # 4B. TOP DIFFERENTIAL SYLLABLES LONGITUDINAL TRAJECTORIES (Small Multiples Grid)
    # Select top 9 most interesting / differential syllables
    # e.g. 61 (upregulated), 92, 94, 38, 03 (downregulated), 42, 15 (high prevalence), 12, 19
    key_sylls = [61, 92, 94, 38, 3, 12, 19, 41, 42]
    # Filter to only those present in the model
    key_sylls = [s for s in key_sylls if s in all_syll_ids]

    n_rows, n_cols = 3, 3
    fig_traj, axes_t = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes_t_flat = axes_t.flatten()

    for idx, s_id in enumerate(key_sylls):
        ax = axes_t_flat[idx]
        ax.axvspan(3.5, 4.5, color="#f1f5f9", alpha=0.9, zorder=0)
        ax.axvline(x=3.5, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)

        sub_s = df_syll_sessions[df_syll_sessions["syllable"] == s_id]

        for g, color in palette.items():
            g_sub = sub_s[sub_s["genotype"] == g]
            for aid in g_sub["animal_id"].unique():
                a_sub = g_sub[g_sub["animal_id"] == aid]
                a_map = {row["condition"]: row["usage_pct"] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.1, zorder=2)

            means = [syll_cohort_dict[str(s_id)]["waves"][c][f"mean_{'pff' if g=='PFF' else 'monomer'}"] for c in condition_order]
            sems = [syll_cohort_dict[str(s_id)]["waves"][c][f"sem_{'pff' if g=='PFF' else 'mono'}"] for c in condition_order]

            ax.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)" if idx == 0 else None,
                        marker="o", markersize=6.5, linewidth=2.2, capsize=3.5, zorder=4)
            ax.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":",
                        marker="o", markersize=6.5, linewidth=1.6, capsize=3.5, zorder=4)

        # Star annotations per wave
        for i, cond in enumerate(condition_order):
            w_info = syll_cohort_dict[str(s_id)]["waves"][cond]
            star = w_info["star"]
            pval = w_info["p_val"]
            if star != "ns":
                y_max = max(w_info["mean_pff"] + w_info["sem_pff"], w_info["mean_monomer"] + w_info["sem_mono"])
                ax.text(i, y_max * 1.10, f"{star}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#d95f02")

        fc_16 = syll_cohort_dict[str(s_id)]["waves"]["16w_PI"]["log2fc"]
        ax.set_title(f"Syllable {s_id:02d} (16w $\\log_2$FC: {fc_16:+.2f})", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_labels_short, fontsize=9.0, fontweight="bold")
        ax.set_ylabel("Usage %", fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    if axes_t_flat[0].get_legend_handles_labels()[0]:
        axes_t_flat[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper left")

    for j in range(len(key_sylls), len(axes_t_flat)):
        axes_t_flat[j].set_visible(False)

    fig_traj.suptitle("Longitudinal Usage Trajectories of Key Phenotype Syllables Across Waves\n(3w -> 8w -> 16w -> 24w PI + L-Dopa Rescue)",
                      fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    trajs_path = out_dir / "top_syllables_longitudinal_trajectories.png"
    fig_traj.savefig(trajs_path, dpi=300)
    plt.close(fig_traj)
    print(f"Saved {trajs_path.name}")

    print("\n" + "=" * 80)
    print("🎉 ALL LONGITUDINAL COHORT ANALYSES COMPLETED SUCCESSFULLY!")
    print(f"Output files in: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
