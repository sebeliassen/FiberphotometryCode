#!/usr/bin/env python3
"""
generate_unified_executive_presentation_plot.py

Creates publication-ready unified figures that combine:
1. Longitudinal Disease Progression Timecourse (3w -> 8w -> 16w -> 24w PI)
2. Pooled Untreated Summary (All 3w-24w) with significance bracket
3. Pharmacological Rescue (L-Dopa + Benserazide)
With explicit ns and p-values annotated for every single week across:
- Master 6-Panel Unified Grid (All 6 key metrics)
- Primary 3-Panel Unified Figure
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

import argparse

DEFAULT_DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "07_longitudinal_cohorts"

DATA_PKL = DEFAULT_DATA_PKL
OUT_DIR = DEFAULT_OUT_DIR


def plot_unified_grid(df_data, panels, nrows, ncols, figsize, out_name, suptitle):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    
    timecourse_conds = ["3w_PI", "8w_PI", "16w_PI", "24w_PI"]
    tc_x = np.array([0, 1, 2, 3])
    untreated_x = 4.4
    ldopa_x = 5.6
    
    for idx, (col, title, ylabel) in enumerate(panels):
        ax = axes_flat[idx]
        
        # Background shading for Untreated Summary vs L-Dopa Rescue
        ax.axvspan(3.8, 5.0, color="#f4f7fa", alpha=0.9, zorder=0)
        ax.axvspan(5.0, 6.2, color="#fef9f3", alpha=0.9, zorder=0)
        
        # Vertical divider lines
        ax.axvline(x=3.8, color="#cccccc", linestyle="-", linewidth=1.5, zorder=1)
        ax.axvline(x=5.0, color="#cccccc", linestyle="--", linewidth=1.5, zorder=1)
        
        # Section Header Badges
        ax.text(1.5, 0.96, "Disease Progression", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.2, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#dddddd", alpha=0.85, zorder=3))
        ax.text(untreated_x, 0.96, "Pooled\nUntreated", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.8, fontweight="bold", color="#1b4f72")
        ax.text(ldopa_x, 0.96, "L-Dopa\nRescue", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.8, fontweight="bold", color="#b9770e")
        
        # 1. Plot Timecourse traces (3w -> 8w -> 16w -> 24w)
        for g, color in palette.items():
            sub = df_data[df_data["genotype"] == g]
            
            # Faint spaghetti traces
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(timecourse_conds) if c in a_map]
                y_pts = [a_map[timecourse_conds[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.14, linewidth=1.0, zorder=2)
                    
            # Means & SEMs for timecourse
            means_tc = sub.groupby("condition")[col].mean().reindex(timecourse_conds).values
            sems_tc = sub.groupby("condition")[col].sem().reindex(timecourse_conds).values
            ax.errorbar(tc_x, means_tc, yerr=sems_tc, color=color, label=f"{g} (Mean ± SEM)" if idx == 0 else None,
                        marker="o", markersize=7.0, linewidth=2.5, capsize=3.8, zorder=4)
            
            # 2. Plot Pooled Untreated point
            sub_untreated = sub[sub["condition"] != "L_dopa"][col].dropna()
            u_mean = sub_untreated.mean()
            u_sem = sub_untreated.sem()
            x_offset = -0.15 if g == "Monomer" else 0.15
            ax.errorbar([untreated_x + x_offset], [u_mean], yerr=[u_sem], color=color,
                        marker="s", markersize=8.0, linewidth=2.5, capsize=4.0, zorder=4)
            
            # 3. Plot L-Dopa Rescue point
            sub_ldopa = sub[sub["condition"] == "L_dopa"][col].dropna()
            d_mean = sub_ldopa.mean()
            d_sem = sub_ldopa.sem()
            ax.errorbar([ldopa_x + x_offset], [d_mean], yerr=[d_sem], color=color,
                        marker="D", markersize=7.5, linewidth=2.5, capsize=4.0, zorder=4)
            
        # Add explicit p-values / ns annotations for EVERY single week (3w, 8w, 16w, 24w)
        for i, cond in enumerate(timecourse_conds):
            sub_c = df_data[df_data["condition"] == cond]
            m_v = sub_c[sub_c["genotype"] == "Monomer"][col].dropna()
            p_v = sub_c[sub_c["genotype"] == "PFF"][col].dropna()
            
            if len(m_v) >= 2 and len(p_v) >= 2:
                t_stat, p_val = stats.ttest_ind(p_v, m_v, equal_var=False)
                y_top = max(m_v.mean() + (m_v.sem() if len(m_v)>1 else 0), p_v.mean() + (p_v.sem() if len(p_v)>1 else 0))
                
                if p_val < 0.05:
                    lbl = f"*\np={p_val:.3f}"
                    clr = "#d95f02"
                    fweight = "bold"
                elif p_val < 0.10:
                    lbl = f"p={p_val:.3f}\n(trend)"
                    clr = "#884400"
                    fweight = "bold"
                else:
                    lbl = f"ns\np={p_val:.2f}"
                    clr = "#777777"
                    fweight = "normal"
                    
                ax.text(i, y_top * 1.07, lbl, ha="center", va="bottom", fontsize=7.5, fontweight=fweight, color=clr)

        # Pooled Untreated Significance Bracket & Stats
        untreated_all = df_data[df_data["condition"] != "L_dopa"]
        m_u_all = untreated_all[untreated_all["genotype"] == "Monomer"][col].dropna()
        p_u_all = untreated_all[untreated_all["genotype"] == "PFF"][col].dropna()
        t_u, p_u_val = stats.ttest_ind(p_u_all, m_u_all, equal_var=False)
        u_top = max(m_u_all.mean() + m_u_all.sem(), p_u_all.mean() + p_u_all.sem())
        
        y_b = u_top * 1.08
        h_b = y_b * 0.02
        ax.plot([untreated_x - 0.15, untreated_x - 0.15, untreated_x + 0.15, untreated_x + 0.15],
                [y_b - h_b, y_b, y_b, y_b - h_b], color="#222222", linewidth=1.1)
        star_u = "*" if p_u_val < 0.05 else ("#" if p_u_val < 0.1 else "ns")
        ax.text(untreated_x, y_b * 1.02, f"{star_u}\np={p_u_val:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold",
                color="#d95f02" if p_u_val < 0.05 else "#333333")

        # L-Dopa Rescue Bracket & Stats
        ldopa_all = df_data[df_data["condition"] == "L_dopa"]
        m_d_all = ldopa_all[ldopa_all["genotype"] == "Monomer"][col].dropna()
        p_d_all = ldopa_all[ldopa_all["genotype"] == "PFF"][col].dropna()
        t_d, p_d_val = stats.ttest_ind(p_d_all, m_d_all, equal_var=False)
        d_top = max(m_d_all.mean() + m_d_all.sem(), p_d_all.mean() + p_d_all.sem())
        y_db = d_top * 1.08
        h_db = y_db * 0.02
        ax.plot([ldopa_x - 0.15, ldopa_x - 0.15, ldopa_x + 0.15, ldopa_x + 0.15],
                [y_db - h_db, y_db, y_db, y_db - h_db], color="#666666", linewidth=1.1)
        ax.text(ldopa_x, y_db * 1.02, f"ns\np={p_d_val:.2f}", ha="center", va="bottom", fontsize=7.8, color="#666666")

        # Axis ticks and labels
        ax.set_xticks([0, 1, 2, 3, untreated_x, ldopa_x])
        ax.set_xticklabels(["3w PI\n(Base)", "8w PI\n(Mid)", "16w PI\n(Peak)", "24w PI\n(Late)",
                            "All Untreated\n(N=78)", "L-Dopa\n(Rescue)"], fontsize=8.8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10.0, fontweight="bold")
        ax.set_title(title, fontsize=11.0, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0] * 0.95 if y_lims[0] > 0 else 0, y_lims[1] * 1.28)
        
    axes_flat[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0, loc="lower left")
    
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
        
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_name}")


def main():
    global DATA_PKL, OUT_DIR
    parser = argparse.ArgumentParser(description="Unified Executive Presentation Plots")
    parser.add_argument("--data-pkl", type=str, default=str(DEFAULT_DATA_PKL), help="Path to input sessions pickle")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Camera framerate (default: 60.0)")
    args = parser.parse_args()

    DATA_PKL = Path(args.data_pkl)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)
        
    records = []
    
    for s in sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        aid = str(md.get("animal_id", ""))
        cond = md.get("condition", "Unknown")
        is_drug = md.get("is_drug_session", False)
        
        if g not in ["PFF", "Monomer"] or cond not in ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]:
            continue
            
        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue
            
        speed = df["velocity_px_s"].values
        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        norm_jerk = np.mean(np.abs(jerk)) / (np.mean(speed) + 1e-3)
        
        # Bout duration & freezing
        is_active = (speed > 5.0).astype(int)
        active_changes = np.diff(np.concatenate(([0], is_active, [0])))
        starts = np.where(active_changes == 1)[0]
        ends = np.where(active_changes == -1)[0]
        bout_durations = (ends - starts) / fps
        mean_bout_dur = np.mean(bout_durations) if len(bout_durations) > 0 else np.nan
        immobility_pct = np.mean(speed < 5.0) * 100.0
        
        # Syllables & Syntax Entropy
        sylls = df["syllable"].values.astype(int)
        changes = np.concatenate(([True], sylls[1:] != sylls[:-1]))
        collapsed = sylls[changes]
        
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
        else:
            h_cond = np.nan
            
        records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "condition": cond,
            "genotype": g,
            "is_drug_session": is_drug,
            "mean_speed": np.mean(speed),
            "p95_speed": np.percentile(speed, 95),
            "norm_jerk": norm_jerk,
            "mean_bout_dur": mean_bout_dur,
            "immobility_pct": immobility_pct,
            "syntax_entropy_rate": h_cond
        })
        
    df_data = pd.DataFrame(records)
    
    # 1. MASTER 6-PANEL UNIFIED FIGURE
    master_6_panels = [
        ("mean_speed", "1. Locomotor Speed (px/s)", "Speed (px/s)"),
        ("p95_speed", "2. Peak Burst Sprint Speed ($v_{95}$)", "Burst Speed (px/s)"),
        ("norm_jerk", "3. Movement Jerkiness (Norm. Jerk)", "Norm. Jerk $\\langle|\\dddot{x}|\\rangle/\\langle v\\rangle$"),
        ("mean_bout_dur", "4. Active Movement Bout Duration (s)", "Continuous Bout Length (s)"),
        ("immobility_pct", "5. Freezing / Immobility (<5 px/s)", "Stationary Time (%)"),
        ("syntax_entropy_rate", "6. Transition Entropy Rate $H(S_{t+1}|S_t)$", "Syntax Disorder (bits)"),
    ]
    plot_unified_grid(df_data, master_6_panels, 2, 3, (19.5, 10.5), "karen_unified_6panel_master.png",
                      "Unified Longitudinal Phenotyping & L-Dopa Rescue (All 6 Behavioral Dimensions)")

    # 2. PRIMARY 3-PANEL UNIFIED FIGURE
    primary_3_panels = [
        ("mean_speed", "A. Mean Locomotor Speed (px/s)", "Locomotor Speed (px/s)"),
        ("p95_speed", "B. Peak Burst Sprint Speed ($v_{95}$, px/s)", "Burst Speed (px/s)"),
        ("norm_jerk", "C. Movement Jerkiness (Norm. Jerk)", "Normalized Jerk $\\langle|\\dddot{x}|\\rangle/\\langle v\\rangle$")
    ]
    plot_unified_grid(df_data, primary_3_panels, 1, 3, (19.5, 6.2), "karen_unified_longitudinal_and_rescue.png", None)


if __name__ == "__main__":
    main()
