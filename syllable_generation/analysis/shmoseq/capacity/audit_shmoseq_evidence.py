#!/usr/bin/env python3
"""
audit_shmoseq_evidence.py
=============================================================================
Comprehensive Evidence Audit of shMoSeq Analysis:
  1. Run configurations, hyperparameter tracking, and training memberships.
  2. Corrected animal-level vs. session-level state support (including K12 State 8).
  3. Recomputed K8 vs. K12 assignment overlap on all 30 sessions + condition stratified + animal-balanced.
  4. Explicit mouse-level phenotypes (Saline baseline, paired Amphetamine response) for K8 and K12 child states.
  5. JS distance code audit and statistical formulation details.

Outputs:
  - Tables in: syllable_generation/analysis/shmoseq/output/tables/audit/
  - Figures in: syllable_generation/analysis/shmoseq/output/figures/audit/
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
WT_SAL_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "wt_saline_k8_seed0.pkl")
MATCHED_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "matched_mixed_k8_seed0.pkl")
POOLED_K12_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "pooled_k12_seed0.pkl")

TABLE_AUDIT_DIR = os.path.join(SHMOSEQ_DIR, "output", "tables", "audit")
FIG_AUDIT_DIR = os.path.join(SHMOSEQ_DIR, "output", "figures", "audit")

def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 1e-7 else 0.0

def main():
    os.makedirs(TABLE_AUDIT_DIR, exist_ok=True)
    os.makedirs(FIG_AUDIT_DIR, exist_ok=True)
    
    print("=" * 80, flush=True)
    print("SHMOSEQ EVIDENCE AUDIT EXECUTION", flush=True)
    print("=" * 80, flush=True)
    
    with open(DATA_FILE, "rb") as f:
        data_bundle = pickle.load(f)
    syllables_dict = data_bundle["syllables_dict"]
    metadata_dict = data_bundle["metadata_dict"]
    fps = data_bundle.get("fps", 40.0125)
    
    # Load all 4 models
    models = {}
    for name, path in [
        ("Pooled_K8", POOLED_K8_FILE),
        ("WT_Saline_K8", WT_SAL_K8_FILE),
        ("Matched_Mixed_K8", MATCHED_K8_FILE),
        ("Pooled_K12", POOLED_K12_FILE),
    ]:
        with open(path, "rb") as f:
            m = pickle.load(f)
            m["n_states"] = m.get("n_states", m.get("hypparams", {}).get("n_states", 8))
            models[name] = m
            
    # -----------------------------------------------------------------------
    # 1. RUN CONFIGURATION & TRAINING MEMBERSHIP AUDIT
    # -----------------------------------------------------------------------
    print("\n--- 1. Auditing Run Configurations & Training Membership ---", flush=True)
    membership_records = []
    
    for fit_name, m in models.items():
        hypp = m.get("hypparams", {})
        seed = hypp.get("random_seed", 0)
        n_states = m["n_states"]
        kappa = hypp.get("trans_kappa", 5e5)
        num_iters = len(m.get("log_probs", []))
        decoding = "sm.hhmm_efficient.smoothed_states (posterior smoothing argmax)"
        
        for sess_key in m["states_dict"].keys():
            meta = metadata_dict[sess_key]
            seq = m["states_dict"][sess_key]
            n_frames = len(seq)
            valid_dur_s = n_frames / fps
            valid_dur_min = valid_dur_s / 60.0
            
            membership_records.append({
                "fit_name": fit_name,
                "n_states": n_states,
                "trans_kappa": kappa,
                "random_seed": seed,
                "gibbs_iterations": num_iters,
                "decoding_method": decoding,
                "session_key": sess_key,
                "animal_id": meta["animal_id"],
                "genotype": meta["genotype"],
                "drug": meta["drug"],
                "cohort": meta["cohort"],
                "valid_frames": n_frames,
                "valid_duration_s": valid_dur_s,
                "valid_duration_min": valid_dur_min,
            })
            
    df_membership = pd.DataFrame(membership_records)
    csv_membership_path = os.path.join(TABLE_AUDIT_DIR, "audit_training_membership_and_configs.csv")
    df_membership.to_csv(csv_membership_path, index=False)
    print(f"Saved: {csv_membership_path}", flush=True)
    
    # Check Matched Mixed vs WT Saline properties
    wt_sal_subset = df_membership[df_membership["fit_name"] == "WT_Saline_K8"]
    matched_subset = df_membership[df_membership["fit_name"] == "Matched_Mixed_K8"]
    
    wt_sal_animals = wt_sal_subset["animal_id"].unique()
    matched_animals = matched_subset["animal_id"].unique()
    
    print(f"  WT Saline K8 fit: {len(wt_sal_subset)} sessions, {len(wt_sal_animals)} unique animals: {sorted(list(wt_sal_animals))}", flush=True)
    print(f"  Matched Mixed K8 fit: {len(matched_subset)} sessions, {len(matched_animals)} unique animals: {sorted(list(matched_animals))}", flush=True)
    print(f"  Distinct animals confirmed in Matched Mixed? {len(matched_animals) == 8}", flush=True)
    print(f"  WT Saline valid duration: total = {wt_sal_subset['valid_duration_min'].sum():.1f} min (mean {wt_sal_subset['valid_duration_min'].mean():.1f} min/sess)", flush=True)
    print(f"  Matched Mixed valid duration: total = {matched_subset['valid_duration_min'].sum():.1f} min (mean {matched_subset['valid_duration_min'].mean():.1f} min/sess)", flush=True)

    # -----------------------------------------------------------------------
    # 2. STATE SUPPORT AUDIT (UNIQUE ANIMALS VS SESSIONS & K12 STATE 8)
    # -----------------------------------------------------------------------
    print("\n--- 2. Auditing Corrected State Support Across Animals & Conditions ---", flush=True)
    
    animals_dict = {}
    for key, meta in metadata_dict.items():
        aid = meta["animal_id"]
        gt = meta["genotype"]
        drug = meta["drug"]
        if aid not in animals_dict:
            animals_dict[aid] = {"genotype": gt, "sessions": {}}
        animals_dict[aid]["sessions"][drug] = key
        
    support_records = []
    
    for fit_name in ["Pooled_K8", "Pooled_K12"]:
        m = models[fit_name]
        K = m["n_states"]
        states_dict = m["states_dict"]
        
        for k in range(K):
            # Compute per-animal occupancy across conditions
            mouse_occs_overall = []
            mouse_occs_by_cohort = {"WT_Saline": [], "WT_Amphetamine": [], "KI_Saline": [], "KI_Amphetamine": []}
            
            total_frames_k = 0
            total_episodes_k = 0
            censored_durations_k = []
            
            for aid, a_info in animals_dict.items():
                for drug, sess_key in a_info["sessions"].items():
                    seq = states_dict[sess_key]
                    cohort = f"{a_info['genotype']}_{drug}"
                    
                    k_frames = np.sum(seq == k)
                    total_frames_k += k_frames
                    occ_pct = (k_frames / len(seq)) * 100.0
                    
                    mouse_occs_overall.append(occ_pct)
                    mouse_occs_by_cohort[cohort].append(occ_pct)
                    
                    st, dur, starts, ends = run_length_encoding(seq)
                    k_ep_idx = np.where(st == k)[0]
                    total_episodes_k += len(k_ep_idx)
                    for ep in k_ep_idx:
                        if starts[ep] > 0 and ends[ep] < len(seq):
                            censored_durations_k.append(dur[ep] / fps)
                            
            mouse_occs_overall = np.array(mouse_occs_overall)
            
            # Number of unique animals expressing state at thresholds
            # Animal is considered expressing if at least one of its sessions meets threshold
            animals_gt0 = 0
            animals_gt1 = 0
            animals_gt2 = 0
            animals_gt5 = 0
            
            for aid, a_info in animals_dict.items():
                aid_occs = [np.mean(states_dict[s_key] == k) * 100.0 for s_key in a_info["sessions"].values()]
                max_aid_occ = max(aid_occs) if len(aid_occs) > 0 else 0.0
                if max_aid_occ > 0.0: animals_gt0 += 1
                if max_aid_occ >= 1.0: animals_gt1 += 1
                if max_aid_occ >= 2.0: animals_gt2 += 1
                if max_aid_occ >= 5.0: animals_gt5 += 1
                
            support_records.append({
                "fit_name": fit_name,
                "state": k,
                "total_valid_time_s": total_frames_k / fps,
                "total_valid_time_min": total_frames_k / fps / 60.0,
                "total_episodes": total_episodes_k,
                "median_censored_duration_s": np.median(censored_durations_k) if len(censored_durations_k) > 0 else 0.0,
                "num_animals_gt0pct": animals_gt0,
                "num_animals_gt1pct": animals_gt1,
                "num_animals_gt2pct": animals_gt2,
                "num_animals_gt5pct": animals_gt5,
                "overall_mean_occ_pct": float(np.mean(mouse_occs_overall)),
                "overall_median_occ_pct": float(np.median(mouse_occs_overall)),
                "overall_min_occ_pct": float(np.min(mouse_occs_overall)),
                "overall_25pct_occ": float(np.percentile(mouse_occs_overall, 25)),
                "overall_75pct_occ": float(np.percentile(mouse_occs_overall, 75)),
                "overall_max_occ_pct": float(np.max(mouse_occs_overall)),
                "wt_sal_mean_occ_pct": float(np.mean(mouse_occs_by_cohort["WT_Saline"])),
                "wt_sal_median_occ_pct": float(np.median(mouse_occs_by_cohort["WT_Saline"])),
                "wt_amph_mean_occ_pct": float(np.mean(mouse_occs_by_cohort["WT_Amphetamine"])),
                "ki_sal_mean_occ_pct": float(np.mean(mouse_occs_by_cohort["KI_Saline"])),
                "ki_sal_median_occ_pct": float(np.median(mouse_occs_by_cohort["KI_Saline"])),
                "ki_amph_mean_occ_pct": float(np.mean(mouse_occs_by_cohort["KI_Amphetamine"])),
            })
            
    df_support = pd.DataFrame(support_records)
    csv_support_path = os.path.join(TABLE_AUDIT_DIR, "audit_state_support_detailed.csv")
    df_support.to_csv(csv_support_path, index=False)
    print(f"Saved: {csv_support_path}", flush=True)
    
    # Identify K12 State 8 contributor
    print("\n  [K12 State 8 Audit]:", flush=True)
    k12_states_dict = models["Pooled_K12"]["states_dict"]
    k12_s8_sessions = {}
    for sess_key, seq in k12_states_dict.items():
        occ = np.mean(seq == 8) * 100.0
        if occ > 0.1:
            k12_s8_sessions[sess_key] = occ
            
    for s_key, occ in k12_s8_sessions.items():
        meta = metadata_dict[s_key]
        aid = meta["animal_id"]
        other_drug = "Saline" if meta["drug"] == "Amphetamine" else "Amphetamine"
        other_key = animals_dict[aid]["sessions"].get(other_drug)
        other_seq = k12_states_dict[other_key]
        other_occ = np.mean(other_seq == 8) * 100.0
        print(f"    State 8 contributor: Session '{s_key}' (Animal {aid}, {meta['cohort']}): {occ:.2f}% occupancy ({np.sum(k12_states_dict[s_key] == 8)} frames)", flush=True)
        print(f"    Other session for same animal: '{other_key}' ({other_drug}): {other_occ:.2f}% occupancy ({np.sum(other_seq == 8)} frames)", flush=True)

    # -----------------------------------------------------------------------
    # 3. RECOMPUTE OVERLAP ON ALL 30 SESSIONS & STRATIFIED BY CONDITION
    # -----------------------------------------------------------------------
    print("\n--- 3. Recomputing K8 vs. K12 Assignment Overlap Across All 30 Sessions ---", flush=True)
    
    k8_dict = models["Pooled_K8"]["states_dict"]
    k12_dict = models["Pooled_K12"]["states_dict"]
    
    # 1. Full 30 sessions pooled
    counts_30 = np.zeros((8, 12), dtype=np.float64)
    counts_by_cohort = {"WT_Saline": np.zeros((8, 12)), "WT_Amphetamine": np.zeros((8, 12)),
                        "KI_Saline": np.zeros((8, 12)), "KI_Amphetamine": np.zeros((8, 12))}
    
    # Animal-balanced overlap
    animal_balanced_row = np.zeros((8, 12), dtype=np.float64)
    animal_balanced_col = np.zeros((8, 12), dtype=np.float64)
    
    for aid, a_info in animals_dict.items():
        aid_counts = np.zeros((8, 12), dtype=np.float64)
        for drug, s_key in a_info["sessions"].items():
            z8 = k8_dict[s_key]
            z12 = k12_dict[s_key]
            cohort = f"{a_info['genotype']}_{drug}"
            
            for a, b in zip(z8, z12):
                counts_30[a, b] += 1
                counts_by_cohort[cohort][a, b] += 1
                aid_counts[a, b] += 1
                
        # Row norm per animal
        aid_row = aid_counts / np.maximum(aid_counts.sum(axis=1, keepdims=True), 1.0)
        animal_balanced_row += aid_row / len(animals_dict)
        
        # Col norm per animal
        aid_col = aid_counts / np.maximum(aid_counts.sum(axis=0, keepdims=True), 1.0)
        animal_balanced_col += aid_col / len(animals_dict)
        
    row_norm_30 = counts_30 / np.maximum(counts_30.sum(axis=1, keepdims=True), 1.0)
    col_norm_30 = counts_30 / np.maximum(counts_30.sum(axis=0, keepdims=True), 1.0)
    
    # Export overlap table
    overlap_rows = []
    for i in range(8):
        for j in range(12):
            overlap_rows.append({
                "k8_state": i,
                "k12_state": j,
                "pooled_frames": int(counts_30[i, j]),
                "p_k12_given_k8_pooled": float(row_norm_30[i, j]),
                "p_k8_given_k12_pooled": float(col_norm_30[i, j]),
                "p_k12_given_k8_animal_balanced": float(animal_balanced_row[i, j]),
                "p_k8_given_k12_animal_balanced": float(animal_balanced_col[i, j]),
                "p_k12_given_k8_wt_sal": float(counts_by_cohort["WT_Saline"][i, j] / max(counts_by_cohort["WT_Saline"][i, :].sum(), 1.0)),
                "p_k12_given_k8_ki_sal": float(counts_by_cohort["KI_Saline"][i, j] / max(counts_by_cohort["KI_Saline"][i, :].sum(), 1.0)),
                "p_k12_given_k8_wt_amph": float(counts_by_cohort["WT_Amphetamine"][i, j] / max(counts_by_cohort["WT_Amphetamine"][i, :].sum(), 1.0)),
                "p_k12_given_k8_ki_amph": float(counts_by_cohort["KI_Amphetamine"][i, j] / max(counts_by_cohort["KI_Amphetamine"][i, :].sum(), 1.0)),
            })
    df_overlap = pd.DataFrame(overlap_rows)
    csv_overlap_path = os.path.join(TABLE_AUDIT_DIR, "audit_k8_k12_overlap_matrices.csv")
    df_overlap.to_csv(csv_overlap_path, index=False)
    print(f"Saved: {csv_overlap_path}", flush=True)
    
    # Generate 6-panel comprehensive overlap figure
    fig_ov, axs_ov = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    
    # 1. Pooled 30 Sessions Row Normalized
    im1 = axs_ov[0, 0].imshow(row_norm_30, cmap="Purples", vmin=0, vmax=1.0)
    axs_ov[0, 0].set_title("A. All 30 Sessions (Pooled Row-Norm: P(K12 | K8))", fontweight="bold", fontsize=11)
    axs_ov[0, 0].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[0, 0].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[0, 0].set_xticks(range(12)); axs_ov[0, 0].set_yticks(range(8))
    plt.colorbar(im1, ax=axs_ov[0, 0], fraction=0.046, pad=0.04)
    
    # 2. Pooled 30 Sessions Column Normalized
    im2 = axs_ov[0, 1].imshow(col_norm_30, cmap="Purples", vmin=0, vmax=1.0)
    axs_ov[0, 1].set_title("B. All 30 Sessions (Pooled Col-Norm: P(K8 | K12))", fontweight="bold", fontsize=11)
    axs_ov[0, 1].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[0, 1].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[0, 1].set_xticks(range(12)); axs_ov[0, 1].set_yticks(range(8))
    plt.colorbar(im2, ax=axs_ov[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Animal-Balanced Row Normalized
    im3 = axs_ov[0, 2].imshow(animal_balanced_row, cmap="Purples", vmin=0, vmax=1.0)
    axs_ov[0, 2].set_title("C. Animal-Balanced Row-Norm: Mean_m P_m(K12 | K8)", fontweight="bold", fontsize=11)
    axs_ov[0, 2].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[0, 2].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[0, 2].set_xticks(range(12)); axs_ov[0, 2].set_yticks(range(8))
    plt.colorbar(im3, ax=axs_ov[0, 2], fraction=0.046, pad=0.04)
    
    # 4. WT Saline
    row_wt_sal = counts_by_cohort["WT_Saline"] / np.maximum(counts_by_cohort["WT_Saline"].sum(axis=1, keepdims=True), 1.0)
    im4 = axs_ov[1, 0].imshow(row_wt_sal, cmap="Blues", vmin=0, vmax=1.0)
    axs_ov[1, 0].set_title("D. WT Saline Only (N=8 sessions)", fontweight="bold", fontsize=11)
    axs_ov[1, 0].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[1, 0].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[1, 0].set_xticks(range(12)); axs_ov[1, 0].set_yticks(range(8))
    plt.colorbar(im4, ax=axs_ov[1, 0], fraction=0.046, pad=0.04)
    
    # 5. KI Saline
    row_ki_sal = counts_by_cohort["KI_Saline"] / np.maximum(counts_by_cohort["KI_Saline"].sum(axis=1, keepdims=True), 1.0)
    im5 = axs_ov[1, 1].imshow(row_ki_sal, cmap="Reds", vmin=0, vmax=1.0)
    axs_ov[1, 1].set_title("E. DAT-KI Saline Only (N=7 sessions)", fontweight="bold", fontsize=11)
    axs_ov[1, 1].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[1, 1].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[1, 1].set_xticks(range(12)); axs_ov[1, 1].set_yticks(range(8))
    plt.colorbar(im5, ax=axs_ov[1, 1], fraction=0.046, pad=0.04)
    
    # 6. Combined Amphetamine (WT & KI)
    counts_amph = counts_by_cohort["WT_Amphetamine"] + counts_by_cohort["KI_Amphetamine"]
    row_amph = counts_amph / np.maximum(counts_amph.sum(axis=1, keepdims=True), 1.0)
    im6 = axs_ov[1, 2].imshow(row_amph, cmap="Oranges", vmin=0, vmax=1.0)
    axs_ov[1, 2].set_title("F. Amphetamine Sessions (WT & KI, N=15)", fontweight="bold", fontsize=11)
    axs_ov[1, 2].set_xlabel("K12 State ($Z^{K12}$)", fontweight="bold")
    axs_ov[1, 2].set_ylabel("K8 State ($Z^{K8}$)", fontweight="bold")
    axs_ov[1, 2].set_xticks(range(12)); axs_ov[1, 2].set_yticks(range(8))
    plt.colorbar(im6, ax=axs_ov[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig_ov_path = os.path.join(FIG_AUDIT_DIR, "fig_audit_overlap_comprehensive.png")
    plt.savefig(fig_ov_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_ov_path}", flush=True)

    # -----------------------------------------------------------------------
    # 4. ACTUAL MOUSE-LEVEL PHENOTYPE VALUES FOR K8 VS K12
    # -----------------------------------------------------------------------
    print("\n--- 4. Exporting Actual Mouse-Level Phenotypes (K8 vs. K12) ---", flush=True)
    
    # Child state selection rule:
    # Based on the POOLED 30-session row-normalized overlap matrix row_norm_30:
    # A K12 state is a child of K8 state i if P(K12=j | K8=i) >= 0.15 (15% frame threshold)
    k8_to_k12_map = {}
    for i in range(8):
        children = np.where(row_norm_30[i] >= 0.15)[0].tolist()
        k8_to_k12_map[i] = children
        print(f"  K8 State Z_{i} (pooled frames = {int(counts_30[i].sum())}) -> Objective K12 Children (>=15%): {children} (shares: {[round(row_norm_30[i, c], 3) for c in children]})", flush=True)
        
    z6_k12_children = k8_to_k12_map[6] # WT-dominant in K8
    z1_k12_children = k8_to_k12_map[1] # KI-dominant in K8
    
    mouse_phenotype_rows = []
    
    for aid, a_info in animals_dict.items():
        gt = a_info["genotype"]
        sal_key = a_info["sessions"]["Saline"]
        amph_key = a_info["sessions"]["Amphetamine"]
        
        # K8 sequences
        z8_sal = k8_dict[sal_key]
        z8_amph = k8_dict[amph_key]
        
        # K12 sequences
        z12_sal = k12_dict[sal_key]
        z12_amph = k12_dict[amph_key]
        
        # K8 Z6 and Z1 occupancies
        k8_z6_sal = np.mean(z8_sal == 6) * 100.0
        k8_z6_amph = np.mean(z8_amph == 6) * 100.0
        k8_z6_delta = k8_z6_amph - k8_z6_sal
        
        k8_z1_sal = np.mean(z8_sal == 1) * 100.0
        k8_z1_amph = np.mean(z8_amph == 1) * 100.0
        k8_z1_delta = k8_z1_amph - k8_z1_sal
        
        # K12 child states
        k12_z6_sal = np.mean(np.isin(z12_sal, z6_k12_children)) * 100.0
        k12_z6_amph = np.mean(np.isin(z12_amph, z6_k12_children)) * 100.0
        k12_z6_delta = k12_z6_amph - k12_z6_sal
        
        k12_z1_sal = np.mean(np.isin(z12_sal, z1_k12_children)) * 100.0
        k12_z1_amph = np.mean(np.isin(z12_amph, z1_k12_children)) * 100.0
        k12_z1_delta = k12_z1_amph - k12_z1_sal
        
        mouse_phenotype_rows.append({
            "animal_id": aid,
            "genotype": gt,
            "k8_z6_sal": k8_z6_sal,
            "k8_z6_amph": k8_z6_amph,
            "k8_z6_delta": k8_z6_delta,
            "k8_z1_sal": k8_z1_sal,
            "k8_z1_amph": k8_z1_amph,
            "k8_z1_delta": k8_z1_delta,
            "k12_z6_sal": k12_z6_sal,
            "k12_z6_amph": k12_z6_amph,
            "k12_z6_delta": k12_z6_delta,
            "k12_z1_sal": k12_z1_sal,
            "k12_z1_amph": k12_z1_amph,
            "k12_z1_delta": k12_z1_delta,
        })
        
    df_pheno = pd.DataFrame(mouse_phenotype_rows)
    csv_pheno_path = os.path.join(TABLE_AUDIT_DIR, "audit_mouse_phenotypes_k8_vs_k12.csv")
    df_pheno.to_csv(csv_pheno_path, index=False)
    print(f"Saved: {csv_pheno_path}", flush=True)
    
    # Statistical tests
    wt_p = df_pheno[df_pheno["genotype"] == "WT"]
    ki_p = df_pheno[df_pheno["genotype"] == "KI"]
    
    print("\n  [Statistical Comparisons - Exact Metrics]:", flush=True)
    
    # K8 Z6 Saline
    u_k8_z6, p_mw_k8_z6 = stats.mannwhitneyu(wt_p["k8_z6_sal"], ki_p["k8_z6_sal"])
    t_k8_z6, p_tt_k8_z6 = stats.ttest_ind(wt_p["k8_z6_sal"], ki_p["k8_z6_sal"], equal_var=False)
    d_k8_z6 = cohens_d(wt_p["k8_z6_sal"], ki_p["k8_z6_sal"])
    print(f"    K8 Z6 Saline: WT = {wt_p['k8_z6_sal'].mean():.2f} ± {wt_p['k8_z6_sal'].sem():.2f}% vs KI = {ki_p['k8_z6_sal'].mean():.2f} ± {ki_p['k8_z6_sal'].sem():.2f}% | Diff = {wt_p['k8_z6_sal'].mean() - ki_p['k8_z6_sal'].mean():.2f}% (d={d_k8_z6:.2f}, MW p={p_mw_k8_z6:.4f}, Welch p={p_tt_k8_z6:.4f})", flush=True)
    
    # K12 Z6-children Saline
    u_k12_z6, p_mw_k12_z6 = stats.mannwhitneyu(wt_p["k12_z6_sal"], ki_p["k12_z6_sal"])
    t_k12_z6, p_tt_k12_z6 = stats.ttest_ind(wt_p["k12_z6_sal"], ki_p["k12_z6_sal"], equal_var=False)
    d_k12_z6 = cohens_d(wt_p["k12_z6_sal"], ki_p["k12_z6_sal"])
    print(f"    K12 Z6-children Saline: WT = {wt_p['k12_z6_sal'].mean():.2f} ± {wt_p['k12_z6_sal'].sem():.2f}% vs KI = {ki_p['k12_z6_sal'].mean():.2f} ± {ki_p['k12_z6_sal'].sem():.2f}% | Diff = {wt_p['k12_z6_sal'].mean() - ki_p['k12_z6_sal'].mean():.2f}% (d={d_k12_z6:.2f}, MW p={p_mw_k12_z6:.4f}, Welch p={p_tt_k12_z6:.4f})", flush=True)
    
    # K8 Z1 Saline
    u_k8_z1, p_mw_k8_z1 = stats.mannwhitneyu(wt_p["k8_z1_sal"], ki_p["k8_z1_sal"])
    t_k8_z1, p_tt_k8_z1 = stats.ttest_ind(wt_p["k8_z1_sal"], ki_p["k8_z1_sal"], equal_var=False)
    d_k8_z1 = cohens_d(ki_p["k8_z1_sal"], wt_p["k8_z1_sal"])
    print(f"    K8 Z1 Saline: WT = {wt_p['k8_z1_sal'].mean():.2f} ± {wt_p['k8_z1_sal'].sem():.2f}% vs KI = {ki_p['k8_z1_sal'].mean():.2f} ± {ki_p['k8_z1_sal'].sem():.2f}% | Diff = {ki_p['k8_z1_sal'].mean() - wt_p['k8_z1_sal'].mean():.2f}% (d={d_k8_z1:.2f}, MW p={p_mw_k8_z1:.4f}, Welch p={p_tt_k8_z1:.4f})", flush=True)
    
    # K12 Z1-children Saline
    u_k12_z1, p_mw_k12_z1 = stats.mannwhitneyu(wt_p["k12_z1_sal"], ki_p["k12_z1_sal"])
    t_k12_z1, p_tt_k12_z1 = stats.ttest_ind(wt_p["k12_z1_sal"], ki_p["k12_z1_sal"], equal_var=False)
    d_k12_z1 = cohens_d(ki_p["k12_z1_sal"], wt_p["k12_z1_sal"])
    print(f"    K12 Z1-children Saline: WT = {wt_p['k12_z1_sal'].mean():.2f} ± {wt_p['k12_z1_sal'].sem():.2f}% vs KI = {ki_p['k12_z1_sal'].mean():.2f} ± {ki_p['k12_z1_sal'].sem():.2f}% | Diff = {ki_p['k12_z1_sal'].mean() - wt_p['k12_z1_sal'].mean():.2f}% (d={d_k12_z1:.2f}, MW p={p_mw_k12_z1:.4f}, Welch p={p_tt_k12_z1:.4f})", flush=True)
    
    # K8 Z1 Paired Amphetamine Delta
    u_k8_d1, p_mw_k8_d1 = stats.mannwhitneyu(wt_p["k8_z1_delta"], ki_p["k8_z1_delta"])
    t_k8_d1, p_tt_k8_d1 = stats.ttest_ind(wt_p["k8_z1_delta"], ki_p["k8_z1_delta"], equal_var=False)
    print(f"    K8 Z1 Paired Shift (Δ): WT = {wt_p['k8_z1_delta'].mean():.2f} ± {wt_p['k8_z1_delta'].sem():.2f}% vs KI = {ki_p['k8_z1_delta'].mean():.2f} ± {ki_p['k8_z1_delta'].sem():.2f}% | Interaction MW p={p_mw_k8_d1:.4f}, Welch p={p_tt_k8_d1:.4f}", flush=True)
    
    # K12 Z1-children Paired Amphetamine Delta
    u_k12_d1, p_mw_k12_d1 = stats.mannwhitneyu(wt_p["k12_z1_delta"], ki_p["k12_z1_delta"])
    t_k12_d1, p_tt_k12_d1 = stats.ttest_ind(wt_p["k12_z1_delta"], ki_p["k12_z1_delta"], equal_var=False)
    print(f"    K12 Z1-children Paired Shift (Δ): WT = {wt_p['k12_z1_delta'].mean():.2f} ± {wt_p['k12_z1_delta'].sem():.2f}% vs KI = {ki_p['k12_z1_delta'].mean():.2f} ± {ki_p['k12_z1_delta'].sem():.2f}% | Interaction MW p={p_mw_k12_d1:.4f}, Welch p={p_tt_k12_d1:.4f}", flush=True)
    
    # Generate 4-panel Mouse-Level Phenotype Figure
    fig_ph, axs_ph = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    
    # Panel A: K8 Z6 vs K12 Z6-children (Saline Baseline)
    axs_ph[0, 0].scatter(np.repeat(0, len(wt_p)), wt_p["k8_z6_sal"], color="#2C7BB6", alpha=0.7, s=40, label="WT mice")
    axs_ph[0, 0].scatter(np.repeat(1, len(ki_p)), ki_p["k8_z6_sal"], color="#D7191C", alpha=0.7, s=40, label="DAT-KI mice")
    axs_ph[0, 0].errorbar(0, wt_p["k8_z6_sal"].mean(), yerr=wt_p["k8_z6_sal"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[0, 0].errorbar(1, ki_p["k8_z6_sal"].mean(), yerr=ki_p["k8_z6_sal"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[0, 0].scatter(np.repeat(2.5, len(wt_p)), wt_p["k12_z6_sal"], color="#2C7BB6", alpha=0.7, s=40)
    axs_ph[0, 0].scatter(np.repeat(3.5, len(ki_p)), ki_p["k12_z6_sal"], color="#D7191C", alpha=0.7, s=40)
    axs_ph[0, 0].errorbar(2.5, wt_p["k12_z6_sal"].mean(), yerr=wt_p["k12_z6_sal"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[0, 0].errorbar(3.5, ki_p["k12_z6_sal"].mean(), yerr=ki_p["k12_z6_sal"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[0, 0].set_xticks([0, 1, 2.5, 3.5])
    axs_ph[0, 0].set_xticklabels(["WT\n(K8 Z6)", "KI\n(K8 Z6)", f"WT\n(K12 {z6_k12_children})", f"KI\n(K12 {z6_k12_children})"], fontweight="bold")
    axs_ph[0, 0].set_ylabel("Saline Baseline Occupancy (%)", fontweight="bold")
    axs_ph[0, 0].set_title(f"A. Z6 Baseline: K8 (p={p_mw_k8_z6:.3f}) vs K12 Children (p={p_mw_k12_z6:.3f})", fontweight="bold", fontsize=11)
    axs_ph[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axs_ph[0, 0].legend()
    
    # Panel B: K8 Z1 vs K12 Z1-children (Saline Baseline)
    axs_ph[0, 1].scatter(np.repeat(0, len(wt_p)), wt_p["k8_z1_sal"], color="#2C7BB6", alpha=0.7, s=40)
    axs_ph[0, 1].scatter(np.repeat(1, len(ki_p)), ki_p["k8_z1_sal"], color="#D7191C", alpha=0.7, s=40)
    axs_ph[0, 1].errorbar(0, wt_p["k8_z1_sal"].mean(), yerr=wt_p["k8_z1_sal"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[0, 1].errorbar(1, ki_p["k8_z1_sal"].mean(), yerr=ki_p["k8_z1_sal"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[0, 1].scatter(np.repeat(2.5, len(wt_p)), wt_p["k12_z1_sal"], color="#2C7BB6", alpha=0.7, s=40)
    axs_ph[0, 1].scatter(np.repeat(3.5, len(ki_p)), ki_p["k12_z1_sal"], color="#D7191C", alpha=0.7, s=40)
    axs_ph[0, 1].errorbar(2.5, wt_p["k12_z1_sal"].mean(), yerr=wt_p["k12_z1_sal"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[0, 1].errorbar(3.5, ki_p["k12_z1_sal"].mean(), yerr=ki_p["k12_z1_sal"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[0, 1].set_xticks([0, 1, 2.5, 3.5])
    axs_ph[0, 1].set_xticklabels(["WT\n(K8 Z1)", "KI\n(K8 Z1)", f"WT\n(K12 {z1_k12_children})", f"KI\n(K12 {z1_k12_children})"], fontweight="bold")
    axs_ph[0, 1].set_ylabel("Saline Baseline Occupancy (%)", fontweight="bold")
    axs_ph[0, 1].set_title(f"B. Z1 Baseline: K8 (p={p_mw_k8_z1:.3f}) vs K12 Children (p={p_mw_k12_z1:.3f})", fontweight="bold", fontsize=11)
    axs_ph[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    
    # Panel C: K8 Z1 Paired Amphetamine Response per Mouse
    for _, r in wt_p.iterrows():
        axs_ph[1, 0].plot([0, 1], [r["k8_z1_sal"], r["k8_z1_amph"]], color="#2C7BB6", alpha=0.4, lw=1.5)
    for _, r in ki_p.iterrows():
        axs_ph[1, 0].plot([2, 3], [r["k8_z1_sal"], r["k8_z1_amph"]], color="#D7191C", alpha=0.4, lw=1.5)
        
    axs_ph[1, 0].errorbar([0, 1], [wt_p["k8_z1_sal"].mean(), wt_p["k8_z1_amph"].mean()],
                          yerr=[wt_p["k8_z1_sal"].sem(), wt_p["k8_z1_amph"].sem()], fmt="o-", color="#2C7BB6", capsize=6, lw=2.5, label="WT (Mean ± SEM)")
    axs_ph[1, 0].errorbar([2, 3], [ki_p["k8_z1_sal"].mean(), ki_p["k8_z1_amph"].mean()],
                          yerr=[ki_p["k8_z1_sal"].sem(), ki_p["k8_z1_amph"].sem()], fmt="s-", color="#D7191C", capsize=6, lw=2.5, label="DAT-KI (Mean ± SEM)")
    axs_ph[1, 0].set_xticks([0, 1, 2, 3])
    axs_ph[1, 0].set_xticklabels(["WT Sal", "WT Amph", "KI Sal", "KI Amph"], fontweight="bold")
    axs_ph[1, 0].set_ylabel("K8 State Z1 Occupancy (%)", fontweight="bold")
    axs_ph[1, 0].set_title("C. K8 State Z1: Paired Drug Shift per Mouse (Δ = Amph - Sal)", fontweight="bold", fontsize=11)
    axs_ph[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axs_ph[1, 0].legend()
    
    # Panel D: Paired Delta (Δ = Amph - Sal) for Z1 in K8 vs K12
    axs_ph[1, 1].scatter(np.repeat(0, len(wt_p)), wt_p["k8_z1_delta"], color="#2C7BB6", alpha=0.7, s=40)
    axs_ph[1, 1].scatter(np.repeat(1, len(ki_p)), ki_p["k8_z1_delta"], color="#D7191C", alpha=0.7, s=40)
    axs_ph[1, 1].errorbar(0, wt_p["k8_z1_delta"].mean(), yerr=wt_p["k8_z1_delta"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[1, 1].errorbar(1, ki_p["k8_z1_delta"].mean(), yerr=ki_p["k8_z1_delta"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[1, 1].scatter(np.repeat(2.5, len(wt_p)), wt_p["k12_z1_delta"], color="#2C7BB6", alpha=0.7, s=40)
    axs_ph[1, 1].scatter(np.repeat(3.5, len(ki_p)), ki_p["k12_z1_delta"], color="#D7191C", alpha=0.7, s=40)
    axs_ph[1, 1].errorbar(2.5, wt_p["k12_z1_delta"].mean(), yerr=wt_p["k12_z1_delta"].sem(), fmt="o", color="#2C7BB6", capsize=6, markersize=8)
    axs_ph[1, 1].errorbar(3.5, ki_p["k12_z1_delta"].mean(), yerr=ki_p["k12_z1_delta"].sem(), fmt="s", color="#D7191C", capsize=6, markersize=8)
    
    axs_ph[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axs_ph[1, 1].set_xticks([0, 1, 2.5, 3.5])
    axs_ph[1, 1].set_xticklabels(["WT\n(K8 ΔZ1)", "KI\n(K8 ΔZ1)", "WT\n(K12 ΔZ1-ch)", "KI\n(K12 ΔZ1-ch)"], fontweight="bold")
    axs_ph[1, 1].set_ylabel("Paired Change Δ Occupancy (%)", fontweight="bold")
    axs_ph[1, 1].set_title(f"D. Paired Drug Response: K8 (p={p_mw_k8_d1:.3f}) vs K12 (p={p_mw_k12_d1:.3f})", fontweight="bold", fontsize=11)
    axs_ph[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig_ph_path = os.path.join(FIG_AUDIT_DIR, "fig_audit_phenotypes_k8_vs_k12.png")
    plt.savefig(fig_ph_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_ph_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
