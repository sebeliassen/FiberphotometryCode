#!/usr/bin/env python3
"""
cluster_duration_baseline.py
=============================================================================
Decomposes baseline syllable cluster run episodes per cluster to resolve
whether the pooled DAT-Het KI elongation (4.68 -> 9.44 bouts/run) is driven by:
1. Within-cluster prolongation of runs (increased intrinsic cluster stickiness),
2. Redistribution of visits toward naturally more persistent clusters,
3. Or specific cluster subtypes.
=============================================================================
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"
OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_NAMES = {
    0: "Cluster 0 (Minor Sniff)",
    1: "Cluster 1 (Dart / Fast Locomotion)",
    2: "Cluster 2 (Head Cast / Lateral Turn)",
    3: "Cluster 3 (Rearing / Upright Posture)",
    4: "Cluster 4 (Pause / Micro-Grooming)",
    5: "Cluster 5 (Core Active / Locomotor-Exploratory Repertoire)"
}

def main():
    print("=" * 80)
    print("SYLLABLE CLUSTER RUN DURATION & BOUT DECOMPOSITION (BASELINE AUDIT)")
    print("=" * 80)

    with open(DATA_PKL, "rb") as f:
        bundle = pickle.load(f)

    sessions = bundle["sessions"]
    cluster_map = np.array(bundle["clusters_0idx"])

    runs_data = []

    for sess in sessions:
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        s_idx = sess["session_idx"]
        df = sess["bouts_df"]

        durs = df["duration_sec"].values
        sylls = df["syllable"].values
        c_seq = cluster_map[sylls]

        change_idx = np.where(c_seq[1:] != c_seq[:-1])[0] + 1
        starts = np.r_[0, change_idx]
        ends = np.r_[change_idx, len(c_seq)]

        for st, en in zip(starts, ends):
            c_id = int(c_seq[st])
            bouts = int(en - st)
            sec = float(np.sum(durs[st:en]))
            runs_data.append({
                "animal_id": aid,
                "genotype": gt,
                "drug": drug,
                "session_idx": s_idx,
                "cluster_id": c_id,
                "cluster_name": CLUSTER_NAMES.get(c_id, f"Cluster {c_id}"),
                "bouts": bouts,
                "duration_sec": sec
            })

    df_runs = pd.DataFrame(runs_data)
    df_runs.to_csv(OUTPUT_DIR / "cluster_run_episodes_all.csv", index=False)

    df_sal = df_runs[df_runs["drug"] == "Saline"]

    summary_rows = []
    for c_id in sorted(CLUSTER_NAMES.keys()):
        c_name = CLUSTER_NAMES[c_id]
        sub = df_sal[df_sal["cluster_id"] == c_id]

        wt_sub = sub[sub["genotype"] == "WT"]
        ki_sub = sub[sub["genotype"] == "KI"]

        n_wt = len(wt_sub)
        n_ki = len(ki_sub)

        wt_bout_med = float(np.median(wt_sub["bouts"])) if n_wt > 0 else 0.0
        wt_bout_mean = float(np.mean(wt_sub["bouts"])) if n_wt > 0 else 0.0
        wt_bout_p90 = float(np.percentile(wt_sub["bouts"], 90)) if n_wt > 0 else 0.0
        wt_sec_mean = float(np.mean(wt_sub["duration_sec"])) if n_wt > 0 else 0.0
        wt_tot_sec = float(np.sum(wt_sub["duration_sec"])) if n_wt > 0 else 0.0

        ki_bout_med = float(np.median(ki_sub["bouts"])) if n_ki > 0 else 0.0
        ki_bout_mean = float(np.mean(ki_sub["bouts"])) if n_ki > 0 else 0.0
        ki_bout_p90 = float(np.percentile(ki_sub["bouts"], 90)) if n_ki > 0 else 0.0
        ki_sec_mean = float(np.mean(ki_sub["duration_sec"])) if n_ki > 0 else 0.0
        ki_tot_sec = float(np.sum(ki_sub["duration_sec"])) if n_ki > 0 else 0.0

        summary_rows.append({
            "cluster_id": c_id,
            "cluster_name": c_name,
            "wt_runs": n_wt,
            "ki_runs": n_ki,
            "wt_bout_med": wt_bout_med,
            "ki_bout_med": ki_bout_med,
            "wt_bout_mean": round(wt_bout_mean, 2),
            "ki_bout_mean": round(ki_bout_mean, 2),
            "wt_bout_p90": round(wt_bout_p90, 1),
            "ki_bout_p90": round(ki_bout_p90, 1),
            "wt_sec_mean": round(wt_sec_mean, 2),
            "ki_sec_mean": round(ki_sec_mean, 2),
            "wt_tot_time_min": round(wt_tot_sec / 60.0, 1),
            "ki_tot_time_min": round(ki_tot_sec / 60.0, 1),
            "bout_ratio_ki_wt": round(ki_bout_mean / (wt_bout_mean + 1e-8), 2)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "cluster_run_durations_by_genotype.csv", index=False)

    print("\n--- PER-CLUSTER DURATION BREAKDOWN (SALINE COHORT) ---")
    header_fmt = "{:<55s} | {:>7s} | {:>7s} | {:>11s} | {:>11s} | {:>10s}"
    print(header_fmt.format("Cluster Name", "WT Runs", "KI Runs", "WT Mean Bts", "KI Mean Bts", "KI/WT Ratio"))
    print("-" * 115)
    for _, r in df_summary.iterrows():
        print(header_fmt.format(
            r["cluster_name"], str(r["wt_runs"]), str(r["ki_runs"]), f"{r['wt_bout_mean']:.2f}", f"{r['ki_bout_mean']:.2f}", f"{r['bout_ratio_ki_wt']:.2f}x"
        ))

    print(f"\n✓ Baseline audit complete. Saved to {OUTPUT_DIR / 'cluster_run_durations_by_genotype.csv'}")

if __name__ == "__main__":
    main()
