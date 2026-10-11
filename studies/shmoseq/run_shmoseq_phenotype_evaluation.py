#!/usr/bin/env python3
"""
run_shmoseq_phenotype_evaluation.py
=============================================================================
Focused Initial Phenotyping on Discovered shMoSeq High-Level States.

Methodological Guidelines:
- Mice are the independent units of inference (N=15 mice: 8 WT, 7 KI).
- Preserves paired within-animal design (Saline vs. Amphetamine).
- Analyzes:
  1. State Occupancy P(Z = k).
  2. State Entry Rate (entries per minute spent outside that state).
  3. Episode Duration (median seconds, censoring boundary episodes).
- Estimates paired drug response: Delta_drug = Amph - Saline.
- Tests genotype interaction: Delta_WT vs Delta_KI (Welch t-test + exact permutation).
- Caches all mouse-level metrics for rapid downstream analysis.
=============================================================================
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "models", "selected_exploratory_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output")
CACHE_FILE = os.path.join(OUTPUT_DIR, "data", "shmoseq_phenotype_metrics_cache.pkl")
MOUSE_TABLE_FILE = os.path.join(OUTPUT_DIR, "tables", "mouse_level_phenotype_table.csv")
STATS_TABLE_FILE = os.path.join(OUTPUT_DIR, "tables", "phenotype_statistical_summary.csv")
FIG_FILE = os.path.join(OUTPUT_DIR, "figures", "fig4_shmoseq_genotype_drug_dissection.png")

def extract_episodes(state_seq):
    """Contiguous runs of states with boundary censoring."""
    episodes = []
    T = len(state_seq)
    if T == 0:
        return episodes
    cur_state = state_seq[0]
    cur_start = 0
    for t in range(1, T):
        if state_seq[t] != cur_state:
            episodes.append({
                "state": int(cur_state),
                "length": t - cur_start,
                "is_boundary": (cur_start == 0)
            })
            cur_state = state_seq[t]
            cur_start = t
    episodes.append({
        "state": int(cur_state),
        "length": T - cur_start,
        "is_boundary": True
    })
    return episodes

def exact_mouse_permutation_test(vals_wt, vals_ki, num_perms=5000):
    """Two-tailed permutation test between WT and KI independent mice."""
    # Filter NaNs
    w = np.array(vals_wt)[~np.isnan(vals_wt)]
    k = np.array(vals_ki)[~np.isnan(vals_ki)]
    if len(w) < 3 or len(k) < 3:
        return np.nan
    obs_diff = np.mean(w) - np.mean(k)
    pooled = np.concatenate([w, k])
    n_w = len(w)

    count_extreme = 0
    np.random.seed(42)
    for _ in range(num_perms):
        perm = np.random.permutation(pooled)
        diff = np.mean(perm[:n_w]) - np.mean(perm[n_w:])
        if abs(diff) >= abs(obs_diff):
            count_extreme += 1
    return float(count_extreme / num_perms)

def main():
    print("=" * 80)
    print("RUNNING SHMOSEQ INITIAL PHENOTYPIC EVALUATION (PAIRED DESIGN)")
    print("=" * 80)

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(MOUSE_TABLE_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_FILE), exist_ok=True)

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    states_dict = bundle["states_dict"]
    metadata_dict = bundle["metadata_dict"]
    fps = bundle.get("fps", 40.0125)
    N_STATES = bundle["hypparams"]["n_states"]

    # -------------------------------------------------------------------------
    # 1. Compute Mouse-Level Metrics (Occupancy, Entry Rate, Duration)
    # -------------------------------------------------------------------------
    mouse_records = []

    for session_key, state_seq in states_dict.items():
        meta = metadata_dict[session_key]
        aid = str(meta["animal_id"])
        genotype = meta["genotype"]
        drug = meta["drug"]
        cohort = meta["cohort"]
        T = len(state_seq)
        total_minutes = (T / fps) / 60.0

        episodes = extract_episodes(state_seq)

        # State-level metrics for this session
        counts = np.bincount(state_seq, minlength=N_STATES)
        occupancies = counts / float(T)

        for k in range(N_STATES):
            # Time spent outside state k
            frames_outside = T - counts[k]
            minutes_outside = (frames_outside / fps) / 60.0

            # Entries into state k (uncensored entries from another state)
            k_eps = [ep for ep in episodes if ep["state"] == k]
            num_entries = len([ep for ep in k_eps if not ep["is_boundary"]])

            entry_rate = (num_entries / minutes_outside) if minutes_outside > 0.1 else 0.0

            # Duration of uncensored episodes
            uncensored_lengths = [ep["length"] for ep in k_eps if not ep["is_boundary"]]
            if len(uncensored_lengths) > 0:
                med_dur_sec = float(np.median(uncensored_lengths) / fps)
                mean_dur_sec = float(np.mean(uncensored_lengths) / fps)
            else:
                med_dur_sec = np.nan
                mean_dur_sec = np.nan

            mouse_records.append({
                "session_key": session_key,
                "animal_id": aid,
                "genotype": genotype,
                "drug": drug,
                "cohort": cohort,
                "state_id": k,
                "occupancy_percent": float(occupancies[k] * 100.0),
                "entry_rate_per_min": float(entry_rate),
                "median_duration_sec": med_dur_sec,
                "mean_duration_sec": mean_dur_sec,
                "num_episodes": len(k_eps),
                "total_frames": int(counts[k]),
                "session_minutes": float(total_minutes),
            })

    df_mouse = pd.DataFrame(mouse_records)
    df_mouse.to_csv(MOUSE_TABLE_FILE, index=False)
    print(f"Saved mouse-level metrics table ({len(df_mouse)} rows) to:\n  {MOUSE_TABLE_FILE}")

    # -------------------------------------------------------------------------
    # 2. Paired Drug Differences & Genotype Interaction Tests
    # -------------------------------------------------------------------------
    print("\nEstimating paired drug differences and genotype interactions...")
    stats_rows = []

    # Pivot to paired structure per animal
    pivoted = df_mouse.pivot(index=["animal_id", "genotype", "state_id"], columns="drug",
                             values=["occupancy_percent", "entry_rate_per_min", "median_duration_sec"]).reset_index()

    for metric in ["occupancy_percent", "entry_rate_per_min", "median_duration_sec"]:
        pivoted[f"{metric}_delta"] = pivoted[(metric, "Amphetamine")] - pivoted[(metric, "Saline")]

    for k in range(N_STATES):
        sub = pivoted[pivoted["state_id"] == k]

        wt_sub = sub[sub["genotype"] == "WT"]
        ki_sub = sub[sub["genotype"] == "KI"]

        for metric, label in [
            ("occupancy_percent", "Occupancy (%)"),
            ("entry_rate_per_min", "Entry Rate (min^-1)"),
            ("median_duration_sec", "Duration (s)"),
        ]:
            wt_sal = wt_sub[(metric, "Saline")].dropna()
            wt_amph = wt_sub[(metric, "Amphetamine")].dropna()
            wt_delta = wt_sub[f"{metric}_delta"].dropna()

            ki_sal = ki_sub[(metric, "Saline")].dropna()
            ki_amph = ki_sub[(metric, "Amphetamine")].dropna()
            ki_delta = ki_sub[f"{metric}_delta"].dropna()

            # Baseline Saline Genotype Diff
            t_base, p_base = stats.ttest_ind(wt_sal, ki_sal, equal_var=False) if len(wt_sal)>1 and len(ki_sal)>1 else (np.nan, np.nan)
            perm_p_base = exact_mouse_permutation_test(wt_sal, ki_sal)

            # Paired Drug Effect inside WT
            t_wt_drug, p_wt_drug = stats.ttest_1samp(wt_delta, 0.0) if len(wt_delta)>1 else (np.nan, np.nan)
            # Paired Drug Effect inside KI
            t_ki_drug, p_ki_drug = stats.ttest_1samp(ki_delta, 0.0) if len(ki_delta)>1 else (np.nan, np.nan)

            # Genotype x Drug Interaction (Difference of deltas)
            t_int, p_int = stats.ttest_ind(wt_delta, ki_delta, equal_var=False) if len(wt_delta)>1 and len(ki_delta)>1 else (np.nan, np.nan)
            perm_p_int = exact_mouse_permutation_test(wt_delta, ki_delta)

            stats_rows.append({
                "state_id": k,
                "metric": metric,
                "WT_Saline_mean": float(np.mean(wt_sal)),
                "WT_Saline_sem": float(stats.sem(wt_sal)) if len(wt_sal)>1 else 0.0,
                "KI_Saline_mean": float(np.mean(ki_sal)),
                "KI_Saline_sem": float(stats.sem(ki_sal)) if len(ki_sal)>1 else 0.0,
                "Baseline_Diff_WT_minus_KI": float(np.mean(wt_sal) - np.mean(ki_sal)),
                "Baseline_Perm_p": perm_p_base,
                "WT_Delta_Amph_minus_Sal": float(np.mean(wt_delta)),
                "WT_Delta_p": float(p_wt_drug),
                "KI_Delta_Amph_minus_Sal": float(np.mean(ki_delta)),
                "KI_Delta_p": float(p_ki_drug),
                "Genotype_Interaction_Diff": float(np.mean(wt_delta) - np.mean(ki_delta)),
                "Interaction_Perm_p": perm_p_int,
                "WT_N": len(wt_delta),
                "KI_N": len(ki_delta),
            })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(STATS_TABLE_FILE, index=False)
    print(f"Saved statistical summary to: {STATS_TABLE_FILE}")

    # Print Key Findings to Console
    print("\n" + "=" * 80)
    print("KEY PHENOTYPIC FINDINGS ACROSS SHMOSEQ STATES (EXPLORATORY REPORT):")
    print("=" * 80)

    # 1. Baseline Saline Genotype Differences
    print("\nA. Baseline Saline Differences (WT Saline vs. KI Saline):")
    occ_stats = df_stats[df_stats["metric"] == "occupancy_percent"]
    for _, r in occ_stats.iterrows():
        sid = int(r["state_id"])
        diff = r["Baseline_Diff_WT_minus_KI"]
        p_val = r["Baseline_Perm_p"]
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else "ns"))
        print(f"  State {sid}: WT = {r['WT_Saline_mean']:5.2f}% | KI = {r['KI_Saline_mean']:5.2f}% | Diff = {diff:+5.2f}% (perm p = {p_val:.4f}) {sig}")

    # 2. Drug-Induced Shifts
    print("\nB. Paired Amphetamine Response & Genotype Interaction:")
    for _, r in occ_stats.iterrows():
        sid = int(r["state_id"])
        int_diff = r["Genotype_Interaction_Diff"]
        int_p = r["Interaction_Perm_p"]
        sig = "***" if int_p < 0.01 else ("**" if int_p < 0.05 else ("*" if int_p < 0.10 else "ns"))
        print(f"  State {sid}: Δ_WT = {r['WT_Delta_Amph_minus_Sal']:+5.2f}% (p={r['WT_Delta_p']:.3f}) | Δ_KI = {r['KI_Delta_Amph_minus_Sal']:+5.2f}% (p={r['KI_Delta_p']:.3f}) | Int = {int_diff:+5.2f}% (p={int_p:.4f}) {sig}")

    # -------------------------------------------------------------------------
    # 3. Comprehensive Publication Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), dpi=300)

    cohorts_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    colors = {"WT_Saline": "#1f77b4", "WT_Amphetamine": "#aec7e8",
              "KI_Saline": "#d62728", "KI_Amphetamine": "#ff9896"}

    # Panel 1: State Occupancy
    ax = axes[0]
    width = 0.18
    x = np.arange(N_STATES)

    for i, coh in enumerate(cohorts_order):
        coh_data = df_mouse[df_mouse["cohort"] == coh]
        means = [coh_data[coh_data["state_id"] == k]["occupancy_percent"].mean() for k in range(N_STATES)]
        sems = [stats.sem(coh_data[coh_data["state_id"] == k]["occupancy_percent"]) for k in range(N_STATES)]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, label=coh,
               color=colors[coh], edgecolor="black", alpha=0.9, capsize=3)

    ax.set_ylabel("Occupancy (% of session)", fontweight="bold")
    ax.set_title("A. High-Level State Occupancy by Genotype & Drug Condition", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"State {k}" for k in range(N_STATES)], fontweight="bold")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    # Panel 2: Entry Rate (entries per min outside)
    ax = axes[1]
    for i, coh in enumerate(cohorts_order):
        coh_data = df_mouse[df_mouse["cohort"] == coh]
        means = [coh_data[coh_data["state_id"] == k]["entry_rate_per_min"].mean() for k in range(N_STATES)]
        sems = [stats.sem(coh_data[coh_data["state_id"] == k]["entry_rate_per_min"]) for k in range(N_STATES)]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, label=coh,
               color=colors[coh], edgecolor="black", alpha=0.9, capsize=3)

    ax.set_ylabel("Entry Rate (entries/min outside)", fontweight="bold")
    ax.set_title("B. State Initiation / Entry Hazard Decomposition", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"State {k}" for k in range(N_STATES)], fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    # Panel 3: Episode Duration
    ax = axes[2]
    for i, coh in enumerate(cohorts_order):
        coh_data = df_mouse[df_mouse["cohort"] == coh]
        means = []
        sems = []
        for k in range(N_STATES):
            vals = coh_data[coh_data["state_id"] == k]["median_duration_sec"].dropna()
            means.append(np.mean(vals) if len(vals) > 0 else 0.0)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, label=coh,
               color=colors[coh], edgecolor="black", alpha=0.9, capsize=3)

    ax.set_ylabel("Episode Duration (seconds)", fontweight="bold")
    ax.set_title("C. State Maintenance / Episode Persistence (Uncensored)", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"State {k}" for k in range(N_STATES)], fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_FILE, bbox_inches="tight")
    plt.close()
    print(f"\nSaved publication figure to:\n  {FIG_FILE}")

    # Cache everything
    cache_bundle = {
        "df_mouse": df_mouse,
        "df_stats": df_stats,
        "pivoted": pivoted,
        "fps": fps,
        "n_states": N_STATES,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_bundle, f)
    print(f"Cached all phenotype metrics to:\n  {CACHE_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
