#!/usr/bin/env python3
"""
run_detailed_audit_and_timelines.py
=============================================================================
Deep dive into shMoSeq temporal dynamics and cross-model reorganization:
  1. Reconcile exact frame counts and minutes (unpadded vs. padded batching).
  2. All-session state timelines (15 mice x 2 conditions, 0-60 min).
  3. Two-way cross-model decomposition (K8 <-> K12):
     - Where did K8 go? (row-normalized)
     - Where did K12 come from? (column-normalized)
     - Formal classification: Preserved, Clean Split, Merge, Reorganization.
  4. Sensitivity analysis of Z6 phenotype in K12:
     - K12 Z6 alone vs. [Z6, Z10].
  5. Empirical macro-state destination routing conditional on leaving.
  6. Corrected pair-specific syllable exit support for transition JS distance.

Outputs:
  - Figures in syllable_generation/analysis/shmoseq/output/figures/timelines/
  - Tables in syllable_generation/analysis/shmoseq/output/tables/timelines/
=============================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
DATA_FILE = os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl")

POOLED_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "selected_exploratory_model.pkl")
POOLED_K12_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "pooled_k12_seed0.pkl")

FIG_DIR = os.path.join(SHMOSEQ_DIR, "output", "figures", "timelines")
TABLE_DIR = os.path.join(SHMOSEQ_DIR, "output", "tables", "timelines")

def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    print("=" * 80, flush=True)
    print("RUNNING DETAILED AUDIT, STATE TIMELINES, & TWO-WAY REORGANIZATION", flush=True)
    print("=" * 80, flush=True)
    
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    fps = bundle.get("fps", 40.0125)
    
    with open(POOLED_K8_FILE, "rb") as f:
        m8 = pickle.load(f)
    with open(POOLED_K12_FILE, "rb") as f:
        m12 = pickle.load(f)
        
    k8_dict = m8["states_dict"]
    k12_dict = m12["states_dict"]
    
    # -----------------------------------------------------------------------
    # 1. FRAME COUNT & TIME RECONCILIATION
    # -----------------------------------------------------------------------
    print("\n--- 1. Reconciling Frame Counts & Timestamps ---", flush=True)
    total_raw_frames = sum(len(s) for s in syllables_dict.values())
    total_raw_minutes = total_raw_frames / fps / 60.0
    print(f"  Total unpadded session frames across all 30 sessions: {total_raw_frames:,} frames", flush=True)
    print(f"  Frame rate: {fps:.4f} Hz (native KPMS downsampled 2x from 80.025 Hz camera clock)", flush=True)
    print(f"  True total duration: {total_raw_minutes:.2f} minutes ({total_raw_minutes/60:.2f} hours)", flush=True)
    print(f"  Explanation of previous 1,643.9 min: in sm.batch, padding to 142,990 frames created a tensor of 4,289,700 frames with a 0.9586 valid mask; 1,712.8 min is the true unpadded sequence duration.", flush=True)

    # -----------------------------------------------------------------------
    # 2. ALL-SESSION STATE TIMELINES FIGURE (30 SESSIONS)
    # -----------------------------------------------------------------------
    print("\n--- 2. Generating All-Session State Timelines (0-60 min) ---", flush=True)
    
    # Organize animals
    animals = {}
    for key, meta in metadata_dict.items():
        aid = meta["animal_id"]
        gt = meta["genotype"]
        drug = meta["drug"]
        if aid not in animals:
            animals[aid] = {"genotype": gt, "sessions": {}}
        animals[aid]["sessions"][drug] = key
        
    wt_mice = sorted([a for a, v in animals.items() if v["genotype"] == "WT"])
    ki_mice = sorted([a for a, v in animals.items() if v["genotype"] == "KI"])
    ordered_mice = wt_mice + ki_mice
    
    # Colormap for K8 states
    cmap_k8 = plt.cm.tab10(np.linspace(0, 1, 8))
    
    fig_time, axs_time = plt.subplots(len(ordered_mice), 2, figsize=(16, 12), dpi=200, sharex=True)
    
    session_timeline_stats = []
    
    for row_idx, aid in enumerate(ordered_mice):
        gt = animals[aid]["genotype"]
        for col_idx, drug in enumerate(["Saline", "Amphetamine"]):
            ax = axs_time[row_idx, col_idx]
            sess_key = animals[aid]["sessions"][drug]
            seq = k8_dict[sess_key]
            
            # RLE for visualization
            st, dur, starts, ends = run_length_encoding(seq)
            
            for b_i in range(len(st)):
                k = st[b_i]
                t_start_min = starts[b_i] / fps / 60.0
                t_end_min = ends[b_i] / fps / 60.0
                ax.axvspan(t_start_min, t_end_min, color=cmap_k8[k], alpha=0.85, edgecolor="none")
                
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            
            # Label
            if col_idx == 0:
                ax.set_ylabel(f"{gt} #{aid}", fontsize=9, fontweight="bold", rotation=0, labelpad=35, va="center")
            if row_idx == 0:
                ax.set_title(f"{drug} Sessions", fontsize=12, fontweight="bold")
            if row_idx == len(ordered_mice) - 1:
                ax.set_xlabel("Time Post-Injection (minutes)", fontsize=10, fontweight="bold")
                
            ax.set_xlim(0, 60)
            
            # Record episode stats
            session_timeline_stats.append({
                "animal_id": aid,
                "genotype": gt,
                "drug": drug,
                "session_key": sess_key,
                "total_episodes": len(st),
                "unique_states": len(np.unique(st)),
                "max_episode_min": float(np.max(dur) / fps / 60.0),
                "dominant_state": int(stats.mode(seq, keepdims=True)[0][0]),
                "dominant_state_pct": float(np.mean(seq == stats.mode(seq, keepdims=True)[0][0]) * 100.0),
            })
            
    # Add legend at bottom
    handles = [matplotlib.patches.Patch(color=cmap_k8[k], label=f"Z_{k}") for k in range(8)]
    fig_time.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=True)
    
    plt.tight_layout()
    fig_time_path = os.path.join(FIG_DIR, "fig_session_state_timelines_k8.png")
    plt.savefig(fig_time_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_time_path}", flush=True)
    
    df_timeline_stats = pd.DataFrame(session_timeline_stats)
    df_timeline_stats.to_csv(os.path.join(TABLE_DIR, "session_timeline_episode_summary.csv"), index=False)
    
    # Print extreme sessions
    print("\n  [Key Timeline Dynamics Observations]:", flush=True)
    mono_sessions = df_timeline_stats[df_timeline_stats["dominant_state_pct"] >= 80.0]
    print(f"  Sessions with >=80% time in a single state ({len(mono_sessions)}/30 sessions):", flush=True)
    for _, r in mono_sessions.iterrows():
        print(f"    {r['session_key']:20s} ({r['genotype']}): State Z_{r['dominant_state']} occupies {r['dominant_state_pct']:.1f}% of entire session (Max episode = {r['max_episode_min']:.1f} min)", flush=True)

    # -----------------------------------------------------------------------
    # 3. TWO-WAY CROSS-MODEL DECOMPOSITION (K8 <-> K12)
    # -----------------------------------------------------------------------
    print("\n--- 3. Evaluating Two-Way K8 <-> K12 Reorganization & Splits ---", flush=True)
    
    # Compute full confusion matrix across all 30 sessions
    N_30 = np.zeros((8, 12), dtype=np.float64)
    for sess_key in k8_dict.keys():
        z8 = k8_dict[sess_key]
        z12 = k12_dict[sess_key]
        for a, b in zip(z8, z12):
            N_30[a, b] += 1
            
    # Where did K8 go? (row-normalized: P(K12=j | K8=i))
    P_k12_given_k8 = N_30 / np.maximum(N_30.sum(axis=1, keepdims=True), 1.0)
    
    # Where did K12 come from? (column-normalized: P(K8=i | K12=j))
    P_k8_given_k12 = N_30 / np.maximum(N_30.sum(axis=0, keepdims=True), 1.0)
    
    # Formal Pattern Classification
    reorg_records = []
    for i in range(8):
        # Targets that receive >= 10% of K8 state i
        major_targets = np.where(P_k12_given_k8[i] >= 0.10)[0]
        row_shares = {int(j): float(P_k12_given_k8[i, j]) for j in major_targets}
        
        # Check source purity of each target
        target_purities = {int(j): float(P_k8_given_k12[i, j]) for j in major_targets}
        
        # Determine classification
        if len(major_targets) == 1:
            j = major_targets[0]
            if P_k12_given_k8[i, j] >= 0.70 and P_k8_given_k12[i, j] >= 0.70:
                pattern = "Preserved State"
            elif P_k8_given_k12[i, j] < 0.70:
                pattern = "Merge into K12"
            else:
                pattern = "Dominant Mapping"
        elif len(major_targets) >= 2:
            # Check if targets come primarily from state i (Clean Split) or absorb other states (Reorganization)
            clean_split = all(P_k8_given_k12[i, j] >= 0.60 for j in major_targets)
            if clean_split:
                pattern = "Clean Split"
            else:
                pattern = "Reorganization / Mixing"
                
        reorg_records.append({
            "k8_state": f"Z_{i}",
            "k8_frames": int(N_30[i].sum()),
            "k8_occupancy_pct": float(N_30[i].sum() / N_30.sum() * 100),
            "k12_destinations": str(row_shares),
            "k12_source_purities": str(target_purities),
            "classification": pattern,
        })
        print(f"  K8 Z_{i} ({N_30[i].sum()/N_30.sum()*100:5.1f}%): Destination Shares = {row_shares} | Source Purities = {target_purities} -> [{pattern}]", flush=True)
        
    df_reorg = pd.DataFrame(reorg_records)
    csv_reorg_path = os.path.join(TABLE_DIR, "k8_to_k12_pattern_classification.csv")
    df_reorg.to_csv(csv_reorg_path, index=False)
    print(f"Saved: {csv_reorg_path}", flush=True)

    # -----------------------------------------------------------------------
    # 4. SENSITIVITY ANALYSIS: Z6 PHENOTYPE IN K12 (ALONE VS REGROUPED)
    # -----------------------------------------------------------------------
    print("\n--- 4. Z6 Phenotype Sensitivity Analysis: K12 Z6 Alone vs. [Z6, Z10] ---", flush=True)
    
    wt_sal_z6_k8 = []
    ki_sal_z6_k8 = []
    
    wt_sal_z6_alone_k12 = []
    ki_sal_z6_alone_k12 = []
    
    wt_sal_z6_regroup_k12 = []
    ki_sal_z6_regroup_k12 = []
    
    for aid in wt_mice:
        s_key = animals[aid]["sessions"]["Saline"]
        z8 = k8_dict[s_key]
        z12 = k12_dict[s_key]
        wt_sal_z6_k8.append(np.mean(z8 == 6) * 100.0)
        wt_sal_z6_alone_k12.append(np.mean(z12 == 6) * 100.0)
        wt_sal_z6_regroup_k12.append(np.mean(np.isin(z12, [6, 10])) * 100.0)
        
    for aid in ki_mice:
        s_key = animals[aid]["sessions"]["Saline"]
        z8 = k8_dict[s_key]
        z12 = k12_dict[s_key]
        ki_sal_z6_k8.append(np.mean(z8 == 6) * 100.0)
        ki_sal_z6_alone_k12.append(np.mean(z12 == 6) * 100.0)
        ki_sal_z6_regroup_k12.append(np.mean(np.isin(z12, [6, 10])) * 100.0)
        
    u_k8, p_k8 = stats.mannwhitneyu(wt_sal_z6_k8, ki_sal_z6_k8)
    u_alone, p_alone = stats.mannwhitneyu(wt_sal_z6_alone_k12, ki_sal_z6_alone_k12)
    u_regroup, p_regroup = stats.mannwhitneyu(wt_sal_z6_regroup_k12, ki_sal_z6_regroup_k12)
    
    print(f"  K8 Z6 Saline:              WT = {np.mean(wt_sal_z6_k8):5.2f} ± {stats.sem(wt_sal_z6_k8):4.2f}% vs KI = {np.mean(ki_sal_z6_k8):5.2f} ± {stats.sem(ki_sal_z6_k8):4.2f}% | MW p = {p_k8:.4f}", flush=True)
    print(f"  K12 Z6 Alone (Saline):      WT = {np.mean(wt_sal_z6_alone_k12):5.2f} ± {stats.sem(wt_sal_z6_alone_k12):4.2f}% vs KI = {np.mean(ki_sal_z6_alone_k12):5.2f} ± {stats.sem(ki_sal_z6_alone_k12):4.2f}% | MW p = {p_alone:.4f}", flush=True)
    print(f"  K12 [Z6, Z10] Regrouped:    WT = {np.mean(wt_sal_z6_regroup_k12):5.2f} ± {stats.sem(wt_sal_z6_regroup_k12):4.2f}% vs KI = {np.mean(ki_sal_z6_regroup_k12):5.2f} ± {stats.sem(ki_sal_z6_regroup_k12):4.2f}% | MW p = {p_regroup:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 5. EMPIRICAL MACRO-STATE DESTINATION ROUTING (CONDITIONAL ON LEAVING)
    # -----------------------------------------------------------------------
    print("\n--- 5. Empirical Macro-State Destination Routing ---", flush=True)
    
    def compute_macro_exit_routing(states_sub):
        exit_counts = np.zeros((8, 8), dtype=np.float64)
        for seq in states_sub:
            st, _, _, _ = run_length_encoding(seq)
            for t in range(len(st) - 1):
                i = st[t]
                j = st[t+1]
                if i != j:
                    exit_counts[i, j] += 1
        exit_sums = exit_counts.sum(axis=1, keepdims=True)
        exit_prob = exit_counts / np.maximum(exit_sums, 1.0)
        return exit_prob, exit_counts
        
    wt_sal_seqs = [k8_dict[animals[a]["sessions"]["Saline"]] for a in wt_mice]
    ki_sal_seqs = [k8_dict[animals[a]["sessions"]["Saline"]] for a in ki_mice]
    
    wt_exit_p, wt_exit_counts = compute_macro_exit_routing(wt_sal_seqs)
    ki_exit_p, ki_exit_counts = compute_macro_exit_routing(ki_sal_seqs)
    
    fig_exit, axs_exit = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    im_w = axs_exit[0].imshow(wt_exit_p, cmap="Blues", vmin=0, vmax=1.0)
    axs_exit[0].set_title(f"A. WT Saline Exit Routing (Total Exits = {int(wt_exit_counts.sum())})", fontweight="bold", fontsize=11)
    axs_exit[0].set_xlabel("Next State Entered ($Z_{t+1}$)", fontweight="bold")
    axs_exit[0].set_ylabel("Leaving State ($Z_t$)", fontweight="bold")
    axs_exit[0].set_xticks(range(8)); axs_exit[0].set_yticks(range(8))
    plt.colorbar(im_w, ax=axs_exit[0], fraction=0.046, pad=0.04)
    
    im_k = axs_exit[1].imshow(ki_exit_p, cmap="Reds", vmin=0, vmax=1.0)
    axs_exit[1].set_title(f"B. DAT-KI Saline Exit Routing (Total Exits = {int(ki_exit_counts.sum())})", fontweight="bold", fontsize=11)
    axs_exit[1].set_xlabel("Next State Entered ($Z_{t+1}$)", fontweight="bold")
    axs_exit[1].set_ylabel("Leaving State ($Z_t$)", fontweight="bold")
    axs_exit[1].set_xticks(range(8)); axs_exit[1].set_yticks(range(8))
    plt.colorbar(im_k, ax=axs_exit[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig_exit_path = os.path.join(FIG_DIR, "fig_macro_exit_routing_wt_vs_ki.png")
    plt.savefig(fig_exit_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_exit_path}", flush=True)

    # -----------------------------------------------------------------------
    # 6. CORRECTED PAIR-SPECIFIC EMPIRICAL EXIT SUPPORT FOR TRANSITION JS
    # -----------------------------------------------------------------------
    print("\n--- 6. Corrected Pair-Specific Exit Support for Transition JS ---", flush=True)
    
    # Extract empirical syllable exits per state
    # exits[k, s, s'] = count of transitions from syllable s to s' (s != s') while in state k
    empirical_exits = np.zeros((8, 100, 100), dtype=np.float64)
    syll_counts_by_state = np.zeros((8, 100), dtype=np.float64)
    
    for sess_key in k8_dict.keys():
        z_seq = k8_dict[sess_key]
        s_seq = syllables_dict[sess_key]
        for t in range(len(s_seq) - 1):
            k = z_seq[t]
            s_cur = s_seq[t]
            s_nxt = s_seq[t+1]
            syll_counts_by_state[k, s_cur] += 1
            if s_cur != s_nxt:
                empirical_exits[k, s_cur, s_nxt] += 1
                
    # Normalize empirical exit distributions
    exit_sums = empirical_exits.sum(axis=2, keepdims=True)
    empirical_exit_dist = empirical_exits / np.maximum(exit_sums, 1.0)
    
    # For each state pair (a, b), find syllables with >= 10 observed exits in BOTH states
    pair_support_records = []
    D_trans_empirical = np.zeros((8, 8), dtype=np.float64)
    
    for a in range(8):
        for b in range(a, 8):
            exits_a = empirical_exits[a].sum(axis=1) # (100,)
            exits_b = empirical_exits[b].sum(axis=1) # (100,)
            
            # Common supported source syllables: >= 10 observed exits in both states
            shared_sylls = np.where((exits_a >= 10) & (exits_b >= 10))[0]
            
            if len(shared_sylls) > 0:
                weights = (exits_a[shared_sylls] + exits_b[shared_sylls])
                weights = weights / weights.sum()
                
                d_pair = sum(weights[idx] * jensenshannon(empirical_exit_dist[a, s], empirical_exit_dist[b, s], base=2)
                             for idx, s in enumerate(shared_sylls))
            else:
                d_pair = np.nan
                
            D_trans_empirical[a, b] = d_pair
            D_trans_empirical[b, a] = d_pair
            
            pair_support_records.append({
                "state_a": a,
                "state_b": b,
                "num_shared_sylls_ge10exits": len(shared_sylls),
                "shared_syll_list": str(shared_sylls.tolist()),
                "total_exits_a": int(exits_a.sum()),
                "total_exits_b": int(exits_b.sum()),
                "empirical_exit_js_distance": float(d_pair) if not np.isnan(d_pair) else None,
            })
            
    df_pair_support = pd.DataFrame(pair_support_records)
    csv_pair_path = os.path.join(TABLE_DIR, "audit_pair_specific_exit_support.csv")
    df_pair_support.to_csv(csv_pair_path, index=False)
    print(f"Saved: {csv_pair_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
