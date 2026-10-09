#!/usr/bin/env python3
"""
generate_longitudinal_cohort_plots.py

Generates publication-ready figures tracking the 21 unique animals
across the 5 experimental conditions / drug challenges:
1. 3w_PI (Baseline, Dec 2025)
2. 8w_PI (Mid-stage, Jan 2026)
3. 16w_PI (Established, Mar 2026)
4. 24w_PI (Late-stage chronic, Apr 2026)
5. L_dopa (L-Dopa + Benserazide Challenge, May 2026)
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}
condition_order = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]
condition_labels = {
    "3w_PI": "3w PI\n(Baseline)",
    "8w_PI": "8w PI\n(Mid)",
    "16w_PI": "16w PI\n(Established)",
    "24w_PI": "24w PI\n(Late)",
    "L_dopa": "L-Dopa\n(+Benserazide)"
}

DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "07_longitudinal_cohorts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)
        
    fps = 80.0
    records = []
    
    for s in sessions:
        meta = getattr(s, "metadata", {})
        ts_str = meta.get("timestamp", "")
        g = meta.get("genotype", "Unknown")
        aid = str(meta.get("animal_id", ""))
        cond = meta.get("condition", "Unknown")
        drug = meta.get("drug", "None")
        is_drug = meta.get("is_drug_session", False)
        weeks_pi = meta.get("weeks_post_injection", np.nan)
        date_str = ts_str.split("T")[0] if "T" in ts_str else ts_str
        
        if g not in ["PFF", "Monomer"]:
            continue
            
        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue
            
        speed = df["velocity_px_s"].values
        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        norm_jerk = np.mean(np.abs(jerk)) / (np.mean(speed) + 1e-3)
        
        sylls = df["syllable"].values.astype(int)
        collapsed = sylls[np.concatenate(([True], sylls[1:] != sylls[:-1]))]
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
            "condition_idx": condition_order.index(cond) if cond in condition_order else -1,
            "drug": drug,
            "is_drug_session": is_drug,
            "weeks_post_injection": weeks_pi,
            "date": date_str,
            "genotype": g,
            "mean_speed_px_s": np.mean(speed),
            "p95_speed_px_s": np.percentile(speed, 95),
            "norm_jerk": norm_jerk,
            "transition_entropy_rate": h_cond,
            "immobility_percentage": np.mean(speed < 5.0) * 100.0
        })
        
    df_long = pd.DataFrame(records)
    df_long = df_long[df_long["condition"].isin(condition_order)]
    df_long.to_csv(OUT_DIR / "longitudinal_data_per_session.csv", index=False)
    
    # Save Condition & Genotype Summary Table
    cond_summary = df_long.groupby(["condition", "genotype"]).agg({
        "mean_speed_px_s": ["mean", "std", "count"],
        "p95_speed_px_s": ["mean", "std"],
        "norm_jerk": ["mean", "std"],
        "transition_entropy_rate": ["mean", "std"],
        "immobility_percentage": ["mean", "std"],
    }).reindex(index=condition_order, level=0).reset_index()
    cond_summary.to_csv(OUT_DIR / "longitudinal_condition_summary.csv", index=False)
    print(f"Saved {OUT_DIR / 'longitudinal_condition_summary.csv'}")
    
    # 1. Generate 4-panel Condition Progression Figure (With L-Dopa demarcation)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)", axes[0, 0]),
        ("p95_speed_px_s", "Peak Sprint Speed (95th % px/s)", axes[0, 1]),
        ("norm_jerk", "Movement Jerkiness (Norm. Jerk)", axes[1, 0]),
        ("transition_entropy_rate", "Syntax Entropy Rate (bits/step)", axes[1, 1]),
    ]
    
    for col, title, ax in metrics:
        # Draw vertical demarcation line before L-Dopa
        ax.axvline(x=3.5, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(3.55, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0, " Drug Rescue\n(L-Dopa)", color="#555555", fontsize=9, style="italic")
        
        for g, color in palette.items():
            sub = df_long[df_long["genotype"] == g].copy()
            # Mean line with error bars
            means = sub.groupby("condition")[col].mean().reindex(condition_order)
            sems = sub.groupby("condition")[col].sem().reindex(condition_order)
            
            x_indices = np.arange(len(condition_order))
            ax.errorbar(x_indices, means.values, yerr=sems.values, color=color, label=f"{g} (Mean ± SEM)", marker="o", markersize=9, linewidth=2.5, capsize=4)
            
            # Individual animal spaghetti faint lines
            for aid in sub["animal_id"].unique():
                a_sub = sub[sub["animal_id"] == aid].copy()
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(condition_order) if c in a_map]
                y_pts = [a_map[condition_order[i]] for i in x_pts]
                if len(x_pts) >= 3:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.0)
                    
        ax.set_xticks(range(len(condition_order)))
        ax.set_xticklabels([condition_labels[c] for c in condition_order], fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=11)
        ax.legend(frameon=True, facecolor="white", loc="best")
        
    plt.tight_layout()
    fig.savefig(OUT_DIR / "longitudinal_cohort_progression.png", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'longitudinal_cohort_progression.png'}")
    
    # 2. Generate L-Dopa Rescue Specific Figure (Pre-Drug 24w Baseline vs L-Dopa Post-Drug)
    fig_rescue, axes_r = plt.subplots(1, 3, figsize=(16, 5))
    rescue_metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)", axes_r[0]),
        ("p95_speed_px_s", "Peak Sprint Speed (px/s)", axes_r[1]),
        ("norm_jerk", "Movement Jerkiness (Norm. Jerk)", axes_r[2]),
    ]
    
    rescue_df = df_long[df_long["condition"].isin(["24w_PI", "L_dopa"])].copy()
    
    for col, title, ax in rescue_metrics:
        sns.barplot(data=rescue_df, x="condition", y=col, hue="genotype", palette=palette, ax=ax, capsize=0.1, err_kws={"linewidth": 1.5})
        sns.stripplot(data=rescue_df, x="condition", y=col, hue="genotype", palette=palette, dodge=True, ax=ax, alpha=0.6, jitter=0.2, edgecolor="black", linewidth=0.5)
        ax.set_title(f"L-Dopa Rescue: {title}", fontsize=11, fontweight="bold")
        ax.set_xticklabels(["24w PI (Pre-Drug)", "L-Dopa + Benserazide"], fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=10)
        # Fix duplicate legend from stripplot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Genotype", frameon=True, facecolor="white")
        
    plt.tight_layout()
    fig_rescue.savefig(OUT_DIR / "ldopa_rescue_comparison.png", dpi=300)
    plt.close(fig_rescue)
    print(f"Saved {OUT_DIR / 'ldopa_rescue_comparison.png'}")


if __name__ == "__main__":
    main()
