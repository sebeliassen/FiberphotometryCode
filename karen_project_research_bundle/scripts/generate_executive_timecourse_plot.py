#!/usr/bin/env python3
"""
generate_executive_timecourse_plot.py

Creates clean, publication-ready longitudinal timecourse figures
tracking the mice across time post-injection (3w -> 8w -> 16w -> 24w -> L-Dopa)
with statistical significance stars (*, **, ns) and clean annotations.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

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

DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "07_longitudinal_cohorts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_p_star(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "#"  # Trend
    else:
        return "ns"


def main():
    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)
        
    fps = 80.0
    records = []
    
    for s in sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        aid = str(md.get("animal_id", ""))
        cond = md.get("condition", "Unknown")
        is_drug = md.get("is_drug_session", False)
        
        if g not in ["PFF", "Monomer"] or cond not in condition_order:
            continue
            
        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue
            
        speed = df["velocity_px_s"].values
        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        norm_jerk = np.mean(np.abs(jerk)) / (np.mean(speed) + 1e-3)
        
        records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "condition": cond,
            "genotype": g,
            "mean_speed": np.mean(speed),
            "p95_speed": np.percentile(speed, 95),
            "norm_jerk": norm_jerk,
        })
        
    df_data = pd.DataFrame(records)
    
    # -------------------------------------------------------------
    # 3-PANEL FIGURE WITH SIGNIFICANCE STARS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    
    panels = [
        ("mean_speed", "A. Mean Locomotor Speed (px/s)", "Locomotor Velocity (px/s)", axes[0]),
        ("p95_speed", "B. Peak Burst Sprint Speed ($v_{95}$, px/s)", "Top 5% Sprint Speed (px/s)", axes[1]),
        ("norm_jerk", "C. Movement Jerkiness (Norm. Jerk)", "Normalized Jerk $\\langle|\\dddot{x}|\\rangle / \\langle v \\rangle$", axes[2]),
    ]
    
    x_pos = np.arange(len(condition_order))
    
    for col, title, ylabel, ax in panels:
        # Shaded background for L-Dopa drug rescue
        ax.axvspan(3.5, 4.5, color="#f2f2f2", alpha=0.9, zorder=0)
        ax.axvline(x=3.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)
        
        # Drug rescue banner at top
        ax.text(
            4.0, 0.93, "L-Dopa Rescue\n(Deficit Abolished)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            color="#444444", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.9, zorder=5)
        )
        
        max_y = 0
        min_y = float("inf")
        
        for g, color in palette.items():
            sub = df_data[df_data["genotype"] == g]
            
            # Individual mouse faint spaghetti traces
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.1, zorder=2)
                    max_y = max(max_y, max(y_pts))
                    min_y = min(min_y, min(y_pts))
            
            # Group Mean & SEM
            means = sub.groupby("condition")[col].mean().reindex(condition_order).values
            sems = sub.groupby("condition")[col].sem().reindex(condition_order).values
            
            # Solid line for drug-free (0 to 3), dotted line connecting to L-Dopa (3 to 4)
            ax.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)" if col == "mean_speed" else None, marker="o", markersize=8, linewidth=2.8, capsize=4.5, zorder=4)
            ax.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":", marker="o", markersize=8, linewidth=2.0, capsize=4.5, zorder=4)
            
        # Calculate & display statistical tests for each timepoint
        for i, cond in enumerate(condition_order):
            sub_c = df_data[df_data["condition"] == cond]
            m_vals = sub_c[sub_c["genotype"] == "Monomer"][col].dropna()
            p_vals = sub_c[sub_c["genotype"] == "PFF"][col].dropna()
            
            if len(m_vals) >= 2 and len(p_vals) >= 2:
                t_stat, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                u_stat, p_u = stats.mannwhitneyu(p_vals, m_vals)
                
                # Top y-position for stars
                y_top = max(m_vals.mean() + (m_vals.sem() if len(m_vals)>1 else 0), p_vals.mean() + (p_vals.sem() if len(p_vals)>1 else 0))
                
                # Annotate stars or p-values
                if p_t < 0.05 or p_u < 0.05:
                    star = "*" if min(p_t, p_u) >= 0.01 else ("**" if min(p_t, p_u) >= 0.001 else "***")
                    ax.text(i, y_top * 1.08, f"{star}\np={min(p_t, p_u):.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#d95f02")
                elif p_t < 0.10 or p_u < 0.10:
                    ax.text(i, y_top * 1.06, f"p={min(p_t, p_u):.3f}\n(trend)", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#884400")
                else:
                    ax.text(i, y_top * 1.05, "ns", ha="center", va="bottom", fontsize=7.5, color="#888888")

        # Global untreated annotation (Drug-free 3w-24w)
        untreated = df_data[df_data["condition"] != "L_dopa"]
        m_all = untreated[untreated["genotype"] == "Monomer"][col].dropna()
        p_all = untreated[untreated["genotype"] == "PFF"][col].dropna()
        t_all, p_all_t = stats.ttest_ind(p_all, m_all, equal_var=False)
        u_all, p_all_u = stats.mannwhitneyu(p_all, m_all)
        
        star_all = "*" if p_all_t < 0.05 else ("#" if p_all_t < 0.1 else "ns")
        ax.text(
            0.03, 0.04, f"Untreated (3w-24w):\n{star_all} p = {p_all_t:.4f}",
            transform=ax.transAxes,
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85)
        )
        
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Weeks Post-Injection & Treatment", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Give enough headroom for stars
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0] * 0.95 if y_lims[0] > 0 else 0, y_lims[1] * 1.22)
        
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper left")
    
    plt.tight_layout()
    out_path = OUT_DIR / "karen_longitudinal_3panel_timecourse.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved updated 3-panel figure with stars to: {out_path}")
    
    # -------------------------------------------------------------
    # STANDALONE 1-PANEL PRIMARY FIGURE (SPEED)
    # -------------------------------------------------------------
    fig_s, ax_s = plt.subplots(figsize=(9, 6))
    ax_s.axvspan(3.5, 4.5, color="#f2f2f2", alpha=0.9, zorder=0)
    ax_s.axvline(x=3.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)
    
    ax_s.text(
        4.0, 0.93, "Pharmacological Rescue\n(L-Dopa + Benserazide)",
        transform=ax_s.get_xaxis_transform(),
        ha="center", va="top",
        color="#333333", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.9, zorder=5)
    )
    
    for g, color in palette.items():
        sub = df_data[df_data["genotype"] == g]
        for aid in sub["animal_id"].unique():
            a_sub = sub[sub["animal_id"] == aid]
            a_map = {row["condition"]: row["mean_speed"] for _, row in a_sub.iterrows()}
            x_pts = [i for i, c in enumerate(condition_order) if c in a_map]
            y_pts = [a_map[condition_order[i]] for i in x_pts]
            if len(x_pts) >= 2:
                ax_s.plot(x_pts, y_pts, color=color, alpha=0.18, linewidth=1.2, zorder=2)
                
        means = sub.groupby("condition")["mean_speed"].mean().reindex(condition_order).values
        sems = sub.groupby("condition")["mean_speed"].sem().reindex(condition_order).values
        
        ax_s.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)", marker="o", markersize=9, linewidth=3.0, capsize=5, zorder=4)
        ax_s.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":", marker="o", markersize=9, linewidth=2.2, capsize=5, zorder=4)

    # Annotate stats at each point
    for i, cond in enumerate(condition_order):
        sub_c = df_data[df_data["condition"] == cond]
        m_vals = sub_c[sub_c["genotype"] == "Monomer"]["mean_speed"].dropna()
        p_vals = sub_c[sub_c["genotype"] == "PFF"]["mean_speed"].dropna()
        if len(m_vals) >= 2 and len(p_vals) >= 2:
            t_stat, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
            y_top = max(m_vals.mean() + m_vals.sem(), p_vals.mean() + p_vals.sem())
            if p_t < 0.05:
                ax_s.text(i, y_top * 1.10, f"*\np={p_t:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#d95f02")
            elif p_t < 0.10:
                ax_s.text(i, y_top * 1.08, f"p={p_t:.3f}\n(trend)", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#884400")
            else:
                ax_s.text(i, y_top * 1.05, "ns", ha="center", va="bottom", fontsize=8.5, color="#888888")

    # Global untreated
    untreated = df_data[df_data["condition"] != "L_dopa"]
    m_all = untreated[untreated["genotype"] == "Monomer"]["mean_speed"].dropna()
    p_all = untreated[untreated["genotype"] == "PFF"]["mean_speed"].dropna()
    t_all, p_all_t = stats.ttest_ind(p_all, m_all, equal_var=False)
    
    ax_s.text(
        0.03, 0.04, f"Untreated (3w-24w):\n* p = {p_all_t:.4f} (Speed -21.2%)",
        transform=ax_s.transAxes,
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9)
    )

    ax_s.set_title("Longitudinal Locomotor Decline and L-Dopa Rescue (PFF vs Monomer)", fontsize=13, fontweight="bold", pad=12)
    ax_s.set_xticks(x_pos)
    ax_s.set_xticklabels(condition_labels, fontsize=10, fontweight="bold")
    ax_s.set_ylabel("Mean Locomotor Speed (px/s)", fontsize=12, fontweight="bold")
    ax_s.set_xlabel("Time Post-Injection & Drug Challenge", fontsize=12, fontweight="bold")
    ax_s.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="upper left")
    ax_s.grid(True, linestyle=":", alpha=0.6)
    y_lims = ax_s.get_ylim()
    ax_s.set_ylim(y_lims[0] * 0.95 if y_lims[0] > 0 else 0, y_lims[1] * 1.25)
    
    plt.tight_layout()
    out_single = OUT_DIR / "karen_locomotor_speed_timecourse.png"
    fig_s.savefig(out_single, dpi=300)
    plt.close(fig_s)
    print(f"Saved updated single figure with stars to: {out_single}")


if __name__ == "__main__":
    main()
