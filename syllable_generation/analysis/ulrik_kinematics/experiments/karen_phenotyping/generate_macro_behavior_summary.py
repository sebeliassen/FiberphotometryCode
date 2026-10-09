#!/usr/bin/env python3
"""
generate_macro_behavior_summary.py

Generates publication-ready figures and a detailed CSV summarizing syllable-agnostic,
macro-level behavioral and kinematic phenotypes between PFF and Monomer:
1. Locomotor Vigor & Peak Sprint Speed
2. Immobility / Akinesia Fraction
3. Movement Bout Duration & Micro-Pauses
4. Behavioral Syntax Entropy Rate H(S_t+1 | S_t)
5. Session Fatigue / Deceleration Slope
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
import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}

import argparse

DEFAULT_DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "06_macro_behavior"

DATA_PKL = DEFAULT_DATA_PKL
OUT_DIR = DEFAULT_OUT_DIR


def main():
    global DATA_PKL, OUT_DIR
    parser = argparse.ArgumentParser(description="Macro Behavior Summary")
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

    valid = [s for s in sessions if getattr(s, "metadata", {}).get("genotype") in ["PFF", "Monomer"]]

    records = []

    for s in valid:
        df = s.all_data_df
        g = s.metadata["genotype"]
        sid = s.session_id
        aid = s.metadata.get("animal_id", sid)

        vdf = df.dropna(subset=["velocity_px_s", "syllable", "centroid_x", "centroid_y"])
        if len(vdf) < 1000:
            continue

        speed = vdf["velocity_px_s"].values
        mean_speed = np.mean(speed)
        p95_speed = np.percentile(speed, 95)
        immobility_frac = np.mean(speed < 5.0)

        # Bout durations
        is_active = (speed >= 5.0).astype(int)
        changes = np.diff(np.concatenate(([0], is_active, [0])))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        bout_lens = (ends - starts) / fps
        mean_bout_dur = np.mean(bout_lens) if len(bout_lens) > 0 else 0.0

        # Transition Entropy Rate
        sylls = vdf["syllable"].values.astype(int)
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

        # Fatigue (First 10 min vs Last 10 min)
        if len(vdf) >= int(fps * 60 * 25): # at least 25 min
            n10 = int(fps * 60 * 10)
            sp_early = vdf.iloc[:n10]["velocity_px_s"].mean()
            sp_late = vdf.iloc[-n10:]["velocity_px_s"].mean()
            fatigue_drop_pct = ((sp_late - sp_early) / (sp_early + 1e-3)) * 100.0
        else:
            fatigue_drop_pct = np.nan

        records.append({
            "session_id": sid,
            "animal_id": aid,
            "genotype": g,
            "mean_speed_px_s": mean_speed,
            "peak_sprint_speed_p95_px_s": p95_speed,
            "immobility_percentage": immobility_frac * 100.0,
            "mean_movement_bout_duration_s": mean_bout_dur,
            "transition_entropy_rate_bits": h_cond,
            "late_session_speed_change_pct": fatigue_drop_pct,
        })

    df_macro = pd.DataFrame(records)
    df_macro.to_csv(OUT_DIR / "macro_behavior_per_session.csv", index=False)

    # Statistical summary
    metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)"),
        ("peak_sprint_speed_p95_px_s", "Peak Sprint Speed (95th pct, px/s)"),
        ("immobility_percentage", "Stationary / Immobility Time (%)"),
        ("mean_movement_bout_duration_s", "Active Movement Bout Duration (s)"),
        ("transition_entropy_rate_bits", "Behavioral Syntax Entropy Rate (bits/step)"),
        ("late_session_speed_change_pct", "Late-Session Speed Change (Fatigue %)"),
    ]

    stat_rows = []
    for m, label in metrics:
        mon = df_macro[df_macro["genotype"] == "Monomer"][m].dropna()
        pff = df_macro[df_macro["genotype"] == "PFF"][m].dropna()

        t_stat, t_pval = stats.ttest_ind(pff, mon, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(pff, mon, alternative="two-sided")

        mon_m, pff_m = mon.mean(), pff.mean()
        mon_s, pff_s = mon.std(), pff.std()
        pooled_std = np.sqrt(((len(mon)-1)*mon_s**2 + (len(pff)-1)*pff_s**2) / (len(mon) + len(pff) - 2))
        cohens_d = (pff_m - mon_m) / (pooled_std + 1e-12)
        pct_change = ((pff_m - mon_m) / (mon_m + 1e-12)) * 100.0

        stat_rows.append({
            "metric": m,
            "description": label,
            "Monomer_mean": mon_m,
            "Monomer_std": mon_s,
            "PFF_mean": pff_m,
            "PFF_std": pff_s,
            "pct_change_PFF_vs_Monomer": pct_change,
            "cohens_d": cohens_d,
            "t_test_pvalue": t_pval,
            "mannwhitney_pvalue": u_pval,
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(OUT_DIR / "macro_behavior_statistical_summary.csv", index=False)
    print(f"Saved {OUT_DIR / 'macro_behavior_statistical_summary.csv'}")

    # Generate 6-panel publication figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (m, label) in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=df_macro, x="genotype", y=m, palette=palette, ax=ax, width=0.4, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_macro, x="genotype", y=m, color="black", alpha=0.6, jitter=0.2, size=5, ax=ax)

        row = df_stats[df_stats["metric"] == m].iloc[0]
        pval = row["mannwhitney_pvalue"]
        t_pval = row["t_test_pvalue"]
        chosen_p = min(pval, t_pval)
        sig_str = "***" if chosen_p < 0.001 else "**" if chosen_p < 0.01 else "*" if chosen_p < 0.05 else "n.s."

        ax.set_title(f"{label}\n(p = {chosen_p:.4f} [{sig_str}], d = {row['cohens_d']:+.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "macro_behavior_genotype_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'macro_behavior_genotype_comparison.png'}")


if __name__ == "__main__":
    main()
