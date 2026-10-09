#!/usr/bin/env python3
"""
generate_all_metric_timecourses.py

Generates publication-ready timecourse figures across ALL key behavioral,
kinematic, bout dynamic, and syntax entropy dimensions with significance stars:
1. Master 6-Panel Timecourse Grid
2. Bout Dynamics & Freezing Panel
3. Action Syntax & Entropy Dynamics Panel
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


def plot_timecourse_grid(df_data, panels, nrows, ncols, figsize, out_name, suptitle):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    x_pos = np.arange(len(condition_order))
    
    for idx, (col, title, ylabel) in enumerate(panels):
        ax = axes_flat[idx]
        
        # Shaded background for L-Dopa drug rescue
        ax.axvspan(3.5, 4.5, color="#f2f2f2", alpha=0.9, zorder=0)
        ax.axvline(x=3.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)
        
        # Drug rescue banner at top
        ax.text(
            4.0, 0.93, "L-Dopa Rescue",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            color="#444444", fontsize=8.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.9, zorder=5)
        )
        
        for g, color in palette.items():
            sub = df_data[df_data["genotype"] == g]
            
            # Individual mouse faint spaghetti traces
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.0, zorder=2)
            
            # Group Mean & SEM
            means = sub.groupby("condition")[col].mean().reindex(condition_order).values
            sems = sub.groupby("condition")[col].sem().reindex(condition_order).values
            
            ax.errorbar(x_pos[:4], means[:4], yerr=sems[:4], color=color, label=f"{g} (Mean ± SEM)" if idx == 0 else None, marker="o", markersize=7.0, linewidth=2.5, capsize=4, zorder=4)
            ax.errorbar(x_pos[3:], means[3:], yerr=sems[3:], color=color, linestyle=":", marker="o", markersize=7.0, linewidth=1.8, capsize=4, zorder=4)

        # Annotate per-timepoint stats
        for i, cond in enumerate(condition_order):
            sub_c = df_data[df_data["condition"] == cond]
            m_vals = sub_c[sub_c["genotype"] == "Monomer"][col].dropna()
            p_vals = sub_c[sub_c["genotype"] == "PFF"][col].dropna()
            if len(m_vals) >= 2 and len(p_vals) >= 2:
                t_stat, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                u_stat, p_u = stats.mannwhitneyu(p_vals, m_vals)
                min_p = min(p_t, p_u)
                y_top = max(m_vals.mean() + (m_vals.sem() if len(m_vals)>1 else 0), p_vals.mean() + (p_vals.sem() if len(p_vals)>1 else 0))
                
                if min_p < 0.05:
                    star = "*" if min_p >= 0.01 else ("**" if min_p >= 0.001 else "***")
                    ax.text(i, y_top * 1.08, f"{star}\np={min_p:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#d95f02")
                elif min_p < 0.10:
                    ax.text(i, y_top * 1.06, f"p={min_p:.3f}\n(trend)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#884400")
                else:
                    ax.text(i, y_top * 1.05, "ns", ha="center", va="bottom", fontsize=7.0, color="#888888")

        # Global untreated test (3w-24w)
        untreated = df_data[df_data["condition"] != "L_dopa"]
        m_all = untreated[untreated["genotype"] == "Monomer"][col].dropna()
        p_all = untreated[untreated["genotype"] == "PFF"][col].dropna()
        t_all, p_all_t = stats.ttest_ind(p_all, m_all, equal_var=False)
        u_all, p_all_u = stats.mannwhitneyu(p_all, m_all)
        star_all = "*" if min(p_all_t, p_all_u) < 0.05 else ("#" if min(p_all_t, p_all_u) < 0.1 else "ns")
        
        ax.text(
            0.03, 0.04, f"Untreated (3w-24w):\n{star_all} p = {min(p_all_t, p_all_u):.4f}",
            transform=ax.transAxes,
            fontsize=8.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85)
        )
        
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_labels, fontsize=9.0, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Post-Injection", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0] * 0.95 if y_lims[0] > 0 else 0, y_lims[1] * 1.25)
        
    axes_flat[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper left")
    
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    if suptitle:
        fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
        
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_name} with significance stars")


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
        drug = md.get("drug", "None")
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
        
        # Bout dynamics
        is_active = (speed > 5.0).astype(int)
        active_changes = np.diff(np.concatenate(([0], is_active, [0])))
        starts = np.where(active_changes == 1)[0]
        ends = np.where(active_changes == -1)[0]
        bout_durations = (ends - starts) / fps
        mean_bout_dur = np.mean(bout_durations) if len(bout_durations) > 0 else np.nan
        immobility_pct = np.mean(speed < 5.0) * 100.0
        session_min = len(speed) / (fps * 60.0)
        bouts_per_min = len(bout_durations) / session_min
        
        # Syllables & Syntax
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
            
        records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "condition": cond,
            "genotype": g,
            "mean_speed": np.mean(speed),
            "p95_speed": np.percentile(speed, 95),
            "p99_speed": np.percentile(speed, 99),
            "norm_jerk": norm_jerk,
            "mean_bout_dur": mean_bout_dur,
            "immobility_pct": immobility_pct,
            "bouts_per_min": bouts_per_min,
            "switch_rate_hz": switch_rate_hz,
            "syntax_entropy_rate": h_cond,
            "repertoire_entropy": h_rep,
        })
        
    df_data = pd.DataFrame(records)
    
    # 1. MASTER 6-PANEL FIGURE
    master_panels = [
        ("mean_speed", "1. Locomotor Speed (px/s)", "Speed (px/s)"),
        ("p95_speed", "2. Peak Burst Sprint Speed ($v_{95}$)", "Burst Speed (px/s)"),
        ("norm_jerk", "3. Movement Jerkiness (Norm. Jerk)", "Norm. Jerk $\\langle|\\dddot{x}|\\rangle/\\langle v\\rangle$"),
        ("mean_bout_dur", "4. Active Movement Bout Duration (s)", "Continuous Bout Length (s)"),
        ("immobility_pct", "5. Freezing / Immobility (<5 px/s)", "Stationary Time (%)"),
        ("syntax_entropy_rate", "6. Transition Entropy Rate $H(S_{t+1}|S_t)$", "Syntax Disorder (bits)"),
    ]
    plot_timecourse_grid(df_data, master_panels, 2, 3, (18, 10), "karen_longitudinal_6panel_master.png", "Comprehensive Longitudinal Kinematic & Behavioral Phenotyping (3w -> 24w PI + L-Dopa)")
    
    # 2. BOUT DYNAMICS & FREEZING 3-PANEL FIGURE
    bout_panels = [
        ("mean_bout_dur", "A. Active Movement Bout Duration (s)", "Continuous Run Length (s)"),
        ("immobility_pct", "B. Freezing / Immobility (<5 px/s)", "Stationary / Freezing (%)"),
        ("bouts_per_min", "C. Movement Initiation Rate (Starts/min)", "Bout Frequency (starts/min)"),
    ]
    plot_timecourse_grid(df_data, bout_panels, 1, 3, (18, 5.5), "karen_bout_dynamics_timecourse.png", "Longitudinal Bout Dynamics & Freezing Progression")
    
    # 3. SYNTAX & ENTROPY 3-PANEL FIGURE
    syntax_panels = [
        ("syntax_entropy_rate", "A. Transition Entropy Rate $H(S_{t+1}|S_t)$", "Syntax Disorder (bits/step)"),
        ("repertoire_entropy", "B. Behavioral Repertoire Entropy $H(S)$", "Repertoire Diversity (bits)"),
        ("switch_rate_hz", "C. Syllable Switching Rate (Hz)", "Switching Frequency (Hz)"),
    ]
    plot_timecourse_grid(df_data, syntax_panels, 1, 3, (18, 5.5), "karen_syntax_entropy_timecourse.png", "Longitudinal Behavioral Syntax & Entropy Dynamics")


if __name__ == "__main__":
    main()
