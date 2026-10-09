#!/usr/bin/env python3
"""
run_karen_permutation_tests.py

Rigorous, empirical label-shuffling permutation test (10,000 permutations)
to validate that observed genotype differences (PFF vs Monomer) are not artifacts,
random flukes, or data snooping.

Tests:
1. Mean Locomotor Speed
2. Peak Sprint Speed (95th percentile)
3. Immobility / Freezing Fraction
4. Movement Jerkiness (Normalized Jerk)
5. Behavioral Syntax Entropy Rate H(S_t+1 | S_t)
6. Movement Bout Duration
7. Top Differential Syllable Usage (Syllables 74, 68, 83, 31, 40, 14)
8. Top Within-Syllable Execution Speed (Syllables 5, 43, 9, 22)
9. Top Syntax Bigram Transition (14 -> 65)
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
from scipy import stats
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "ulrik_kinematics" / "output" / "karen_genotype_comparison" / "permutation_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading Karen dataset from {DATA_PKL}...")
    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)
        
    valid = [s for s in sessions if getattr(s, "metadata", {}).get("genotype") in ["PFF", "Monomer"]]
    fps = 80.0
    print(f"Total valid sessions: {len(valid)} (PFF: {sum(1 for s in valid if s.metadata['genotype']=='PFF')}, Monomer: {sum(1 for s in valid if s.metadata['genotype']=='Monomer')})")
    
    # 1. Extract per-session features
    session_data = []
    
    for s in valid:
        g = s.metadata["genotype"]
        sid = s.session_id
        df = s.all_data_df
        
        vdf = df.dropna(subset=["velocity_px_s", "syllable", "centroid_x", "centroid_y"])
        if len(vdf) < 1000:
            continue
            
        speed = vdf["velocity_px_s"].values
        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        
        mean_speed = np.mean(speed)
        p95_speed = np.percentile(speed, 95)
        immobility_frac = np.mean(speed < 5.0)
        norm_jerk = np.mean(np.abs(jerk)) / (mean_speed + 1e-3)
        
        # Bout duration
        is_active = (speed >= 5.0).astype(int)
        changes = np.diff(np.concatenate(([0], is_active, [0])))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        bout_lens = (ends - starts) / fps
        mean_bout = np.mean(bout_lens) if len(bout_lens) > 0 else 0.0
        
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
            
        # Top Differential Syllable Usage
        syll_counts = pd.Series(sylls).value_counts()
        total_sylls = len(sylls)
        usage_74 = syll_counts.get(74, 0) / total_sylls
        usage_68 = syll_counts.get(68, 0) / total_sylls
        usage_83 = syll_counts.get(83, 0) / total_sylls
        usage_31 = syll_counts.get(31, 0) / total_sylls
        usage_40 = syll_counts.get(40, 0) / total_sylls
        usage_14 = syll_counts.get(14, 0) / total_sylls
        
        # Top Within-Syllable Speed
        speed_syll_5 = np.mean(speed[sylls == 5]) if np.sum(sylls == 5) > 10 else np.nan
        speed_syll_43 = np.mean(speed[sylls == 43]) if np.sum(sylls == 43) > 10 else np.nan
        speed_syll_9 = np.mean(speed[sylls == 9]) if np.sum(sylls == 9) > 10 else np.nan
        speed_syll_22 = np.mean(speed[sylls == 22]) if np.sum(sylls == 22) > 10 else np.nan
        
        # Syntax 14 -> 65
        trans_14_65 = p_pairs.get((14, 65), 0.0) if len(collapsed) >= 50 else 0.0
        
        session_data.append({
            "session_id": sid,
            "genotype": g,
            "mean_speed": mean_speed,
            "peak_speed_p95": p95_speed,
            "immobility_frac": immobility_frac,
            "norm_jerk": norm_jerk,
            "transition_entropy_rate": h_cond,
            "mean_bout_duration": mean_bout,
            "usage_syll_74": usage_74,
            "usage_syll_68": usage_68,
            "usage_syll_83": usage_83,
            "usage_syll_31": usage_31,
            "usage_syll_40": usage_40,
            "usage_syll_14": usage_14,
            "speed_syll_5": speed_syll_5,
            "speed_syll_43": speed_syll_43,
            "speed_syll_9": speed_syll_9,
            "speed_syll_22": speed_syll_22,
            "syntax_14_to_65": trans_14_65,
        })
        
    df = pd.DataFrame(session_data)
    print(f"Extracted features across {len(df)} sessions.")
    
    # 2. Permutation Testing Framework (N = 10,000 permutations)
    N_PERM = 10000
    np.random.seed(42)
    
    test_metrics = [
        ("mean_speed", "Mean Locomotor Speed (px/s)"),
        ("peak_speed_p95", "Peak Sprint Speed p95 (px/s)"),
        ("immobility_frac", "Immobility Fraction (<5 px/s)"),
        ("norm_jerk", "Movement Jerkiness (Normalized Jerk)"),
        ("transition_entropy_rate", "Syntax Transition Entropy Rate (bits/step)"),
        ("mean_bout_duration", "Active Bout Duration (s)"),
        ("usage_syll_74", "Syllable 74 Usage Fraction"),
        ("usage_syll_68", "Syllable 68 Usage Fraction"),
        ("usage_syll_83", "Syllable 83 Usage Fraction"),
        ("usage_syll_31", "Syllable 31 Usage Fraction"),
        ("usage_syll_40", "Syllable 40 Usage Fraction"),
        ("usage_syll_14", "Syllable 14 Usage Fraction"),
        ("speed_syll_5", "Syllable 5 Execution Speed (px/s)"),
        ("speed_syll_43", "Syllable 43 Execution Speed (px/s)"),
        ("speed_syll_9", "Syllable 9 Execution Speed (px/s)"),
        ("speed_syll_22", "Syllable 22 Execution Speed (px/s)"),
        ("syntax_14_to_65", "Transition 14 -> 65 Probability"),
    ]
    
    perm_results = []
    null_distributions = {}
    
    genotypes_true = df["genotype"].values
    n_pff = (genotypes_true == "PFF").sum()
    n_mon = (genotypes_true == "Monomer").sum()
    
    print(f"\nRunning {N_PERM:,} label-shuffling permutations for each metric...")
    
    for col, label in test_metrics:
        # Valid non-nan subset
        sub = df.dropna(subset=[col])
        vals = sub[col].values
        g_arr = sub["genotype"].values
        
        # True observed values
        obs_mon = vals[g_arr == "Monomer"]
        obs_pff = vals[g_arr == "PFF"]
        obs_delta = np.mean(obs_pff) - np.mean(obs_mon)
        obs_t_stat, obs_t_pval = stats.ttest_ind(obs_pff, obs_mon, equal_var=False)
        obs_mwu_stat, obs_mwu_pval = stats.mannwhitneyu(obs_pff, obs_mon, alternative="two-sided")
        
        # Generate Permutation Null Distribution
        n_sub = len(vals)
        n_pff_sub = (g_arr == "PFF").sum()
        
        perm_deltas = np.zeros(N_PERM)
        for i in range(N_PERM):
            shuffled_idx = np.random.permutation(n_sub)
            pff_perm = vals[shuffled_idx[:n_pff_sub]]
            mon_perm = vals[shuffled_idx[n_pff_sub:]]
            perm_deltas[i] = np.mean(pff_perm) - np.mean(mon_perm)
            
        null_distributions[col] = perm_deltas
        
        # Two-tailed empirical permutation p-value
        emp_pval = (1 + np.sum(np.abs(perm_deltas) >= np.abs(obs_delta))) / (N_PERM + 1)
        
        # Permutation Z-score
        null_mean = np.mean(perm_deltas)
        null_std = np.std(perm_deltas)
        z_score = (obs_delta - null_mean) / (null_std + 1e-12)
        
        perm_results.append({
            "metric": col,
            "description": label,
            "Monomer_mean": np.mean(obs_mon),
            "PFF_mean": np.mean(obs_pff),
            "observed_delta_PFF_minus_Mon": obs_delta,
            "pct_change": (obs_delta / (np.mean(obs_mon) + 1e-12)) * 100.0,
            "parametric_t_pvalue": obs_t_pval,
            "mannwhitney_pvalue": obs_mwu_pval,
            "empirical_permutation_pvalue": emp_pval,
            "permutation_z_score": z_score,
            "null_95pct_CI_lower": np.percentile(perm_deltas, 2.5),
            "null_95pct_CI_upper": np.percentile(perm_deltas, 97.5),
            "significant_at_alpha_05": emp_pval < 0.05,
        })
        
        print(f"  {col:<25} | Obs Delta: {obs_delta:+.4f} | Null 95% CI: [{np.percentile(perm_deltas, 2.5):+.4f}, {np.percentile(perm_deltas, 97.5):+.4f}] | Perm p: {emp_pval:.5f} | MWU p: {obs_mwu_pval:.4f}")
        
    df_perm = pd.DataFrame(perm_results)
    df_perm.to_csv(OUT_DIR / "label_permutation_test_results.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'label_permutation_test_results.csv'}")
    
    # 3. Plot Null Distributions Grid (6 key features)
    plot_keys = [
        ("mean_speed", "Mean Speed (px/s)"),
        ("peak_speed_p95", "Peak Sprint Speed p95 (px/s)"),
        ("norm_jerk", "Movement Jerkiness (Norm. Jerk)"),
        ("transition_entropy_rate", "Syntax Entropy Rate (bits/step)"),
        ("speed_syll_5", "Syllable 5 Execution Speed (px/s)"),
        ("usage_syll_74", "Syllable 74 Usage Fraction"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (k, label) in enumerate(plot_keys):
        ax = axes[i]
        null_dist = null_distributions[k]
        row = df_perm[df_perm["metric"] == k].iloc[0]
        obs_val = row["observed_delta_PFF_minus_Mon"]
        p_val = row["empirical_permutation_pvalue"]
        
        # Plot Null Histogram
        ax.hist(null_dist, bins=50, color="#4a7bb0", alpha=0.65, edgecolor="white", density=True, label="Permutation Null (10,000 Shuffles)")
        
        # 95% CI
        ci_l = row["null_95pct_CI_lower"]
        ci_u = row["null_95pct_CI_upper"]
        ax.axvline(ci_l, color="gray", linestyle="--", linewidth=1.5, label="Null 95% CI")
        ax.axvline(ci_u, color="gray", linestyle="--", linewidth=1.5)
        
        # Observed Statistic Line
        ax.axvline(obs_val, color="#d95f02", linewidth=2.5, linestyle="-", label=f"Observed PFF-Mon Delta\n(p_perm = {p_val:.4f})")
        
        sig_str = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "n.s."
        ax.set_title(f"{label}\n[Observed = {obs_val:+.3f} | {sig_str}]", fontsize=11, fontweight="bold")
        ax.set_xlabel("Permuted Group Mean Difference (PFF - Monomer)")
        ax.set_ylabel("Probability Density")
        ax.legend(frameon=True, facecolor="white", fontsize=9, loc="upper right" if obs_val < 0 else "upper left")
        
    plt.tight_layout()
    fig.savefig(OUT_DIR / "label_permutation_null_distributions.png", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'label_permutation_null_distributions.png'}")


if __name__ == "__main__":
    main()
