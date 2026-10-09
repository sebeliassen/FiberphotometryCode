#!/usr/bin/env python3
"""
inspect_multi_layer_durations.py
=============================================================================
Quantitative Empirical Analysis of Durations, Persistence & Stickiness (kappa)
Across Hierarchical Behavioral Layers:
- Layer 0: Frame-level Syllable Bouts (Sub-second motor primitives)
- Layer 1: Multi-Syllable Chains (Triplets and Subroutines)
- Layer 2: Behavioral Episodes (Cluster-homogeneous runs & loop repetitions)
- Layer 3: Macro-States / Regimes (HMM states & 20-bout context blocks)
=============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"

with open(DATA_PKL, "rb") as f:
    bundle = pickle.load(f)

sessions = bundle["sessions"]
cluster_map = bundle["clusters_0idx"]

# ---------------------------------------------------------
# LAYER 0: Syllable Bout Level
# ---------------------------------------------------------
all_syll_durs = []
all_syll_gts = []
all_syll_tokens = []

# LAYER 1: Multi-Syllable Chains (Triplets)
chain_durs = []
chain_types = []
chain_gts = []

# LAYER 2: Behavioral Episodes (Cluster-homogeneous runs)
cluster_run_durs = []
cluster_run_bouts = []
cluster_run_gts = []

# Transition matrices to compute self-transition persistence
n_sylls = 100
syll_trans_counts = np.zeros((n_sylls, n_sylls))
n_clusters = 5
cluster_trans_counts = np.zeros((n_clusters, n_clusters))

for sess in sessions:
    gt = sess["genotype"]
    drug = sess["drug"]
    if drug != "Saline":
        continue
    df = sess["bouts_df"]
    durs = df["duration_sec"].values
    sylls = df["syllable"].values

    all_syll_durs.extend(durs)
    all_syll_gts.extend([gt] * len(durs))
    all_syll_tokens.extend(sylls)

    # Layer 0 transitions
    for i in range(len(sylls) - 1):
        syll_trans_counts[sylls[i], sylls[i+1]] += 1

    # Layer 1: Triplets
    for i in range(len(sylls) - 2):
        s1, s2, s3 = sylls[i], sylls[i+1], sylls[i+2]
        tot_dur = durs[i] + durs[i+1] + durs[i+2]
        if (s1, s2, s3) == (1, 11, 1):
            chain_durs.append(tot_dur)
            chain_types.append("Locomotor [1, 11, 1]")
            chain_gts.append(gt)
        elif (s1, s2, s3) == (6, 17, 2):
            chain_durs.append(tot_dur)
            chain_types.append("Orienting [6, 17, 2]")
            chain_gts.append(gt)
        elif (s1, s2, s3) == (14, 7, 0):
            chain_durs.append(tot_dur)
            chain_types.append("Rearing-Exit [14, 7, 0]")
            chain_gts.append(gt)

    # Layer 2: Cluster Episodes
    c_seq = cluster_map[sylls]
    for i in range(len(c_seq) - 1):
        if c_seq[i] < n_clusters and c_seq[i+1] < n_clusters:
            cluster_trans_counts[c_seq[i], c_seq[i+1]] += 1

    change_idx = np.where(c_seq[1:] != c_seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(c_seq)]
    for st, en in zip(starts, ends):
        cluster_run_durs.append(float(np.sum(durs[st:en])))
        cluster_run_bouts.append(int(en - st))
        cluster_run_gts.append(gt)

# Convert to arrays/dfs
all_syll_durs = np.array(all_syll_durs)
all_syll_gts = np.array(all_syll_gts)
df_chains = pd.DataFrame({"type": chain_types, "duration": chain_durs, "gt": chain_gts})
df_ep = pd.DataFrame({"duration": cluster_run_durs, "bouts": cluster_run_bouts, "gt": cluster_run_gts})

print("=" * 80)
print("QUANTITATIVE DURATION & STICKINESS (KAPPA) PROFILING ACROSS BEHAVIORAL LAYERS")
print("=" * 80)

print("\n--- LAYER 0: SUB-SECOND SYLLABLE PRIMITIVES (KPMS BOUTS) ---")
print(f"Total Analyzed Bouts (Saline): {len(all_syll_durs):,}")
print(f"Overall Duration: Mean = {np.mean(all_syll_durs):.3f}s | Median = {np.median(all_syll_durs):.3f}s | IQR = [{np.percentile(all_syll_durs, 25):.3f}s - {np.percentile(all_syll_durs, 75):.3f}s]")
print(f"90th Percentile = {np.percentile(all_syll_durs, 90):.3f}s | 99th Percentile = {np.percentile(all_syll_durs, 99):.3f}s")
for gt in ["WT", "KI"]:
    mask = all_syll_gts == gt
    d_sub = all_syll_durs[mask]
    print(f"  • {gt}: Mean = {np.mean(d_sub):.3f}s | Median = {np.median(d_sub):.3f}s | IQR = [{np.percentile(d_sub, 25):.3f}s - {np.percentile(d_sub, 75):.3f}s]")

# Syllable self-transition (in RLE, by definition bout self-transition is 0, but frame self-transition was governed by kappa)
print("Note: In RLE bout representation, consecutive identical syllables are compressed into single bouts.")

print("\n--- LAYER 1: MULTI-SYLLABLE CHAINS (3-BOUT MOTOR PROGRESSIONS) ---")
print(f"Total Target Chains: {len(df_chains):,}")
for ctype in sorted(df_chains["type"].unique()):
    sub = df_chains[df_chains["type"] == ctype]
    print(f"  • {ctype} (N={len(sub)}):")
    print(f"      Overall: Mean = {sub['duration'].mean():.3f}s | Median = {sub['duration'].median():.3f}s | IQR = [{sub['duration'].quantile(0.25):.3f}s - {sub['duration'].quantile(0.75):.3f}s]")
    for gt in ["WT", "KI"]:
        g_sub = sub[sub["gt"] == gt]
        print(f"      {gt}: Median = {g_sub['duration'].median():.3f}s | Mean = {g_sub['duration'].mean():.3f}s (N={len(g_sub)})")

print("\n--- LAYER 2: BEHAVIORAL EPISODES (CLUSTER-HOMOGENEOUS RUNS) ---")
print(f"Total Clustered Episodes: {len(df_ep):,}")
print(f"Bouts per Episode: Mean = {df_ep['bouts'].mean():.2f} bouts | Median = {df_ep['bouts'].median():.1f} | 90th = {df_ep['bouts'].quantile(0.90):.0f} bouts")
print(f"Episode Duration: Mean = {df_ep['duration'].mean():.2f}s | Median = {df_ep['duration'].median():.2f}s | IQR = [{df_ep['duration'].quantile(0.25):.2f}s - {df_ep['duration'].quantile(0.75):.2f}s] | 90th = {df_ep['duration'].quantile(0.90):.2f}s")
for gt in ["WT", "KI"]:
    g_sub = df_ep[df_ep["gt"] == gt]
    print(f"  • {gt}: Median = {g_sub['duration'].median():.2f}s | Mean = {g_sub['duration'].mean():.2f}s | Bouts Mean = {g_sub['bouts'].mean():.2f}")

# Cluster persistence (stickiness)
cluster_trans_prob = cluster_trans_counts / (cluster_trans_counts.sum(axis=1, keepdims=True) + 1e-8)
print("\nCluster Self-Persistence (Stickiness / Diagonal Probability):")
cluster_names = ["0: Exploration/Sniff", "1: Locomotion", "2: Head Cast/Turn", "3: Rearing", "4: Pause/Stationary"]
for c in range(5):
    p_self = cluster_trans_prob[c, c]
    print(f"  Cluster {cluster_names[c]}: Self-Transition = {p_self*100:.1f}%")

print("\n--- LAYER 3: MACRO-STATES / REGIMES (FROM HMM & 20-BOUT BLOCKS) ---")
print("From Jonathan Run 12 Benchmark (Chapter 10 & NMF Review Package):")
print("  • Block Window Size: 20 consecutive bouts (~12.0 seconds)")
print("  • Macro-State Dwell Times (Physical Seconds):")
print("      Q1 (Active Locomotor Regime):   WT = 36.6 min (61.5%) vs KI = 20.8 min (34.9%)")
print("      Q2 (High-Investigation Regime): WT = 9.8 min (16.5%)  vs KI = 40.5 min (68.1%)")
print("      Q4 (Quiescence / Pause State):  WT = 15.9 min (26.8%) vs KI = 1.8 min (3.0%)  [p=0.0300]")
print("  • Macro-State Stickiness (Exit Hazard per 20-bout block):")
print("      Q1 Exit Hazard: WT = 8.5%  vs KI = 12.1%")
print("      Q4 Exit Hazard: WT = 38.4% vs KI = 67.8% (KI cannot remain sticky in rest!)")

print("=" * 80)
