#!/usr/bin/env python3
"""
compute_state_similarity_and_overlap.py
=============================================================================
Computes objective statistical state similarities, cross-model assignment
overlaps, per-mouse support metrics, and K=12 phenotype robustness:
  1. Syllable Repertoire JS Distance (frame-weighted & bout-onset-weighted)
  2. Modeled Syllable Transition JS Distance (continuation vs destination)
  3. Cross-Model Assignment Overlap (C_ab confusion matrices)
  4. Per-Mouse State Support (tallying contributing mice >=2% and bout metrics)
  5. Phenotype Preservation in K=12 vs K=8

Outputs:
  - 4 publication-quality multi-panel figures in output/figures/capacity/
  - Detailed CSV tables in output/tables/capacity/
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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
DATA_FILE = os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl")

# Models
POOLED_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "selected_exploratory_model.pkl")
WT_SAL_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "wt_saline_k8_seed0.pkl")
MATCHED_K8_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "matched_mixed_k8_seed0.pkl")
POOLED_K12_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "capacity", "pooled_k12_seed0.pkl")

FIG_DIR = os.path.join(SHMOSEQ_DIR, "output", "figures", "capacity")
TABLE_DIR = os.path.join(SHMOSEQ_DIR, "output", "tables", "capacity")

def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends

def compute_repertoire_distributions(states_dict, syllables_dict, n_states, n_syllables=100):
    """
    Computes frame-weighted and bout-onset-weighted syllable distributions.
    Bout onsets are true syllable changes (S_t != S_{t-1}); macro-state boundary
    within an ongoing syllable is NOT counted as a new syllable onset.
    """
    frame_counts = np.zeros((n_states, n_syllables), dtype=np.float64)
    onset_counts = np.zeros((n_states, n_syllables), dtype=np.float64)
    
    for key, z_seq in states_dict.items():
        s_seq = syllables_dict[key]
        assert len(z_seq) == len(s_seq)
        
        # 1. Frame counts
        for k in range(n_states):
            mask_k = (z_seq == k)
            if np.any(mask_k):
                frame_counts[k] += np.bincount(s_seq[mask_k], minlength=n_syllables)
                
        # 2. Syllable bout onsets
        syll_types, syll_durs, syll_starts, syll_ends = run_length_encoding(s_seq)
        for b_i in range(len(syll_types)):
            s_type = syll_types[b_i]
            onset_frame = syll_starts[b_i]
            k_state = z_seq[onset_frame]
            onset_counts[k_state, s_type] += 1
            
    # Normalize with safe division
    frame_sums = frame_counts.sum(axis=1, keepdims=True)
    frame_sums[frame_sums == 0] = 1.0
    p_frame = frame_counts / frame_sums
    
    onset_sums = onset_counts.sum(axis=1, keepdims=True)
    onset_sums[onset_sums == 0] = 1.0
    p_onset = onset_counts / onset_sums
    
    return p_frame, p_onset, frame_counts, onset_counts

def compute_pairwise_js_distance(p_matrix):
    """Pairwise JS distance (sqrt of JS divergence, base 2) between all state rows."""
    K = p_matrix.shape[0]
    D = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            d = jensenshannon(p_matrix[i], p_matrix[j], base=2)
            D[i, j] = d
            D[j, i] = d
    return D

def compute_transition_js_distance(params, n_states, global_syll_counts, n_syllables=100):
    """
    Computes syllable transition JS distance using sm.hhmm_efficient.get_syllable_trans_probs.
    Decouples self-transition continuation probability from exit destination routing.
    """
    # trans_probs: (n_states, n_syllables, n_syllables)
    trans_probs = np.array(sm.hhmm_efficient.get_syllable_trans_probs(params["emission_base"], params["emission_biases"]))
    
    # Global syllable weights (normalized across top syllables)
    w_s = global_syll_counts / global_syll_counts.sum()
    valid_source_sylls = np.where(global_syll_counts >= 100)[0] # adequate empirical support
    w_sub = w_s[valid_source_sylls]
    w_sub = w_sub / w_sub.sum()
    
    # Extract continuation (diagonal) and destination routing (off-diagonal normalized)
    continuation_prob = np.zeros((n_states, n_syllables), dtype=np.float64)
    exit_routing = np.zeros((n_states, n_syllables, n_syllables), dtype=np.float64)
    
    for k in range(n_states):
        for s in range(n_syllables):
            continuation_prob[k, s] = trans_probs[k, s, s]
            off_row = trans_probs[k, s, :].copy()
            off_row[s] = 0.0
            sum_off = off_row.sum()
            if sum_off > 0:
                exit_routing[k, s, :] = off_row / sum_off
                
    # Compute pairwise weighted JS distance on exit routing
    D_trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(i, n_states):
            d_weighted = 0.0
            for idx, s in enumerate(valid_source_sylls):
                d_s = jensenshannon(exit_routing[i, s, :], exit_routing[j, s, :], base=2)
                d_weighted += w_sub[idx] * d_s
            D_trans[i, j] = d_weighted
            D_trans[j, i] = d_weighted
            
    return D_trans, continuation_prob

def decode_sessions_with_model(sessions_dict, model_bundle):
    """Decodes a dictionary of sessions using a fitted model's parameters."""
    seqs, mask, batch_metadata = sm.batch(sessions_dict)
    data = {"syllables": jnp.array(seqs), "mask": jnp.array(mask)}
    smoothed_probs = sm.hhmm_efficient.smoothed_states(data, model_bundle["params"], parallel=False)
    decoded_states = jnp.argmax(smoothed_probs, axis=-1)
    states_dict = sm.unbatch(np.array(decoded_states), *batch_metadata)
    return states_dict

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    print("=" * 80)
    print("COMPUTING STATE SIMILARITY, CROSS-MODEL OVERLAP, & CAPACITY METRICS")
    print("=" * 80)
    
    # 1. Load Data
    with open(DATA_FILE, "rb") as f:
        data_bundle = pickle.load(f)
    syllables_dict = data_bundle["syllables_dict"]
    metadata_dict = data_bundle["metadata_dict"]
    fps = data_bundle.get("fps", 40.0125)
    
    all_sylls_flat = np.hstack(list(syllables_dict.values()))
    global_syll_counts = np.bincount(all_sylls_flat, minlength=100)
    
    # 2. Load Models
    models = {}
    for name, path in [
        ("Pooled_K8", POOLED_K8_FILE),
        ("WT_Saline_K8", WT_SAL_K8_FILE),
        ("Matched_Mixed_K8", MATCHED_K8_FILE),
        ("Pooled_K12", POOLED_K12_FILE),
    ]:
        if not os.path.exists(path):
            print(f"Error: Required model file not found: {path}")
            sys.exit(1)
        with open(path, "rb") as f:
            m_loaded = pickle.load(f)
            m_loaded["n_states"] = m_loaded.get("n_states", m_loaded.get("hypparams", {}).get("n_states", 8))
            models[name] = m_loaded
        print(f"Loaded {name}: K={models[name]['n_states']}, sessions={len(models[name]['states_dict'])}", flush=True)
        
    # -----------------------------------------------------------------------
    # FIGURE 5: Capacity Comparison & Condition Diversity
    # -----------------------------------------------------------------------
    print("\n--- Generating Figure 5: Capacity Comparison & State Occupancies ---")
    fig5, axs5 = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    
    fit_names = ["Pooled_K8", "Matched_Mixed_K8", "WT_Saline_K8", "Pooled_K12"]
    titles = [
        "A. Full Pooled (N=30, K=8)",
        "B. Matched Mixed Control (N=8, K=8)",
        "C. WT Saline Only (N=8, K=8)",
        "D. Capacity Expansion (N=30, K=12)"
    ]
    
    capacity_summary_rows = []
    for idx, (m_name, title) in enumerate(zip(fit_names, titles)):
        ax = axs5[idx // 2, idx % 2]
        m = models[m_name]
        K = m["n_states"]
        props = m["state_proportions"] * 100
        
        # Effective states (e.g. >2% occupancy)
        active_states = np.where(props >= 2.0)[0]
        # Shannon perplexity: exp(entropy)
        ent = stats.entropy(m["state_proportions"] + 1e-12)
        perplexity = np.exp(ent)
        
        bars = ax.bar(np.arange(K), props, color=plt.cm.tab20(np.linspace(0, 1, K)), edgecolor="black", alpha=0.85)
        ax.axhline(2.0, color="gray", linestyle="--", alpha=0.7, label="2% Active Threshold")
        ax.set_title(f"{title}\nActive States (>2%): {len(active_states)}/{K} | Perplexity: {perplexity:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("State ID ($Z_k$)", fontweight="bold")
        ax.set_ylabel("Occupancy (%)", fontweight="bold")
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"$Z_{{{k}}}$" for k in range(K)])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
        for bar, p in zip(bars, props):
            if p >= 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
                
        capacity_summary_rows.append({
            "model": m_name,
            "K_cap": K,
            "num_sessions": len(m["states_dict"]),
            "active_states_gt2pct": len(active_states),
            "perplexity": perplexity,
            "max_occupancy_pct": np.max(props),
            "min_occupancy_pct": np.min(props),
        })
        
    plt.tight_layout()
    fig5_path = os.path.join(FIG_DIR, "fig5_capacity_and_diversity_comparison.png")
    plt.savefig(fig5_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to:\n  {fig5_path}")
    
    pd.DataFrame(capacity_summary_rows).to_csv(os.path.join(TABLE_DIR, "capacity_model_comparison_table.csv"), index=False)
    
    # -----------------------------------------------------------------------
    # FIGURE 6: Statistical State Similarity Heatmaps (JS Distances)
    # -----------------------------------------------------------------------
    print("\n--- Generating Figure 6: Syllable Repertoire & Transition Distances ---")
    fig6, axs6 = plt.subplots(2, 2, figsize=(13, 11), dpi=200)
    
    # For Pooled K=8
    p_frame_k8, p_onset_k8, _, _ = compute_repertoire_distributions(models["Pooled_K8"]["states_dict"], syllables_dict, n_states=8)
    D_js_frame_k8 = compute_pairwise_js_distance(p_frame_k8)
    D_js_onset_k8 = compute_pairwise_js_distance(p_onset_k8)
    D_trans_k8, _ = compute_transition_js_distance(models["Pooled_K8"]["params"], n_states=8, global_syll_counts=global_syll_counts)
    
    # For Pooled K=12
    p_frame_k12, p_onset_k12, _, _ = compute_repertoire_distributions(models["Pooled_K12"]["states_dict"], syllables_dict, n_states=12)
    D_js_frame_k12 = compute_pairwise_js_distance(p_frame_k12)
    
    # Panel A: Pooled K=8 Frame-weighted Repertoire Distance
    im_a = axs6[0, 0].imshow(D_js_frame_k8, cmap="viridis", vmin=0, vmax=1.0)
    axs6[0, 0].set_title("A. Pooled K=8: Syllable Repertoire Distance (Frame-Weighted)", fontsize=11, fontweight="bold")
    axs6[0, 0].set_xticks(range(8))
    axs6[0, 0].set_yticks(range(8))
    axs6[0, 0].set_xticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    axs6[0, 0].set_yticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    plt.colorbar(im_a, ax=axs6[0, 0], fraction=0.046, pad=0.04, label="JS Distance ($D_{JS}$)")
    
    # Panel B: Pooled K=8 Bout-Onset-Weighted Repertoire Distance
    im_b = axs6[0, 1].imshow(D_js_onset_k8, cmap="viridis", vmin=0, vmax=1.0)
    axs6[0, 1].set_title("B. Pooled K=8: Syllable Repertoire Distance (Bout-Onset-Weighted)", fontsize=11, fontweight="bold")
    axs6[0, 1].set_xticks(range(8))
    axs6[0, 1].set_yticks(range(8))
    axs6[0, 1].set_xticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    axs6[0, 1].set_yticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    plt.colorbar(im_b, ax=axs6[0, 1], fraction=0.046, pad=0.04, label="JS Distance ($D_{JS}$)")
    
    # Panel C: Modeled Syllable Transition Routing Distance
    im_c = axs6[1, 0].imshow(D_trans_k8, cmap="plasma", vmin=0, vmax=1.0)
    axs6[1, 0].set_title("C. Pooled K=8: Syllable Transition Routing Distance ($D_{trans}$)", fontsize=11, fontweight="bold")
    axs6[1, 0].set_xticks(range(8))
    axs6[1, 0].set_yticks(range(8))
    axs6[1, 0].set_xticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    axs6[1, 0].set_yticklabels([f"$Z_{{{k}}}$" for k in range(8)])
    plt.colorbar(im_c, ax=axs6[1, 0], fraction=0.046, pad=0.04, label="Transition JS Distance")
    
    # Panel D: Pooled K=12 Frame-weighted Repertoire Distance
    im_d = axs6[1, 1].imshow(D_js_frame_k12, cmap="viridis", vmin=0, vmax=1.0)
    axs6[1, 1].set_title("D. Pooled K=12: Syllable Repertoire Distance (12x12 Matrix)", fontsize=11, fontweight="bold")
    axs6[1, 1].set_xticks(range(12))
    axs6[1, 1].set_yticks(range(12))
    axs6[1, 1].set_xticklabels([f"$Z_{{{k}}}$" for k in range(12)], fontsize=8)
    axs6[1, 1].set_yticklabels([f"$Z_{{{k}}}$" for k in range(12)], fontsize=8)
    plt.colorbar(im_d, ax=axs6[1, 1], fraction=0.046, pad=0.04, label="JS Distance ($D_{JS}$)")
    
    plt.tight_layout()
    fig6_path = os.path.join(FIG_DIR, "fig6_statistical_state_similarity_heatmaps.png")
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to:\n  {fig6_path}")
    
    # -----------------------------------------------------------------------
    # FIGURE 7: Cross-Model Frame Assignment Overlap (C_ab)
    # -----------------------------------------------------------------------
    print("\n--- Generating Figure 7: Cross-Model Assignment Overlap ---")
    # Decode the EXACT SAME 8 WT Saline sessions with all 4 models
    wt_sal_keys = [k for k, v in metadata_dict.items() if v["cohort"] == "WT_Saline"]
    wt_sal_syllables = {k: syllables_dict[k] for k in wt_sal_keys}
    
    print(f"Decoding {len(wt_sal_keys)} WT Saline sessions across models...")
    dec_wt_k8 = decode_sessions_with_model(wt_sal_syllables, models["WT_Saline_K8"])
    dec_matched_k8 = decode_sessions_with_model(wt_sal_syllables, models["Matched_Mixed_K8"])
    dec_pooled_k8 = decode_sessions_with_model(wt_sal_syllables, models["Pooled_K8"])
    dec_pooled_k12 = decode_sessions_with_model(wt_sal_syllables, models["Pooled_K12"])
    
    # Flatten shared sequences
    z_wt_flat = np.hstack([dec_wt_k8[k] for k in wt_sal_keys])
    z_matched_flat = np.hstack([dec_matched_k8[k] for k in wt_sal_keys])
    z_pooled8_flat = np.hstack([dec_pooled_k8[k] for k in wt_sal_keys])
    z_pooled12_flat = np.hstack([dec_pooled_k12[k] for k in wt_sal_keys])
    
    def compute_overlap_matrix(z_A, z_B, K_A, K_B):
        counts = np.zeros((K_A, K_B), dtype=np.float64)
        for a, b in zip(z_A, z_B):
            counts[a, b] += 1
        row_norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        col_norm = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1.0)
        return row_norm, col_norm
        
    C_pooled_vs_wt_row, C_pooled_vs_wt_col = compute_overlap_matrix(z_pooled8_flat, z_wt_flat, 8, 8)
    C_matched_vs_wt_row, C_matched_vs_wt_col = compute_overlap_matrix(z_matched_flat, z_wt_flat, 8, 8)
    C_k8_vs_k12_row, C_k8_vs_k12_col = compute_overlap_matrix(z_pooled8_flat, z_pooled12_flat, 8, 12)
    
    fig7, axs7 = plt.subplots(2, 2, figsize=(13, 11), dpi=200)
    
    # Panel A: Pooled K=8 vs WT Saline Only (Row Normalized)
    im7_a = axs7[0, 0].imshow(C_pooled_vs_wt_row, cmap="Blues", vmin=0, vmax=1.0)
    axs7[0, 0].set_title("A. Pooled K=8 → WT Saline K=8 (P(WT | Pooled))", fontsize=11, fontweight="bold")
    axs7[0, 0].set_xlabel("WT Saline K=8 State", fontweight="bold")
    axs7[0, 0].set_ylabel("Pooled K=8 State", fontweight="bold")
    axs7[0, 0].set_xticks(range(8))
    axs7[0, 0].set_yticks(range(8))
    plt.colorbar(im7_a, ax=axs7[0, 0], fraction=0.046, pad=0.04)
    
    # Panel B: Matched Mixed K=8 vs WT Saline Only (Row Normalized)
    im7_b = axs7[0, 1].imshow(C_matched_vs_wt_row, cmap="Blues", vmin=0, vmax=1.0)
    axs7[0, 1].set_title("B. Matched Mixed K=8 → WT Saline K=8 (P(WT | Matched))", fontsize=11, fontweight="bold")
    axs7[0, 1].set_xlabel("WT Saline K=8 State", fontweight="bold")
    axs7[0, 1].set_ylabel("Matched Mixed K=8 State", fontweight="bold")
    axs7[0, 1].set_xticks(range(8))
    axs7[0, 1].set_yticks(range(8))
    plt.colorbar(im7_b, ax=axs7[0, 1], fraction=0.046, pad=0.04)
    
    # Panel C: Pooled K=8 vs Pooled K=12 (Row Normalized - Detecting Splits)
    im7_c = axs7[1, 0].imshow(C_k8_vs_k12_row, cmap="Purples", vmin=0, vmax=1.0)
    axs7[1, 0].set_title("C. Pooled K=8 → Pooled K=12 (Detecting State Splits)", fontsize=11, fontweight="bold")
    axs7[1, 0].set_xlabel("Pooled K=12 State ($Z^{K12}$)", fontweight="bold")
    axs7[1, 0].set_ylabel("Pooled K=8 State ($Z^{K8}$)", fontweight="bold")
    axs7[1, 0].set_xticks(range(12))
    axs7[1, 0].set_yticks(range(8))
    plt.colorbar(im7_c, ax=axs7[1, 0], fraction=0.046, pad=0.04)
    
    # Panel D: Pooled K=12 vs Pooled K=8 (Column Normalized - Reverse Mapping)
    im7_d = axs7[1, 1].imshow(C_k8_vs_k12_col.T, cmap="Purples", vmin=0, vmax=1.0)
    axs7[1, 1].set_title("D. Pooled K=12 → Pooled K=8 (Detecting State Mergers / Ancestry)", fontsize=11, fontweight="bold")
    axs7[1, 1].set_xlabel("Pooled K=8 State ($Z^{K8}$)", fontweight="bold")
    axs7[1, 1].set_ylabel("Pooled K=12 State ($Z^{K12}$)", fontweight="bold")
    axs7[1, 1].set_xticks(range(8))
    axs7[1, 1].set_yticks(range(12))
    plt.colorbar(im7_d, ax=axs7[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig7_path = os.path.join(FIG_DIR, "fig7_cross_model_assignment_overlap.png")
    plt.savefig(fig7_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7 to:\n  {fig7_path}")
    
    # -----------------------------------------------------------------------
    # FIGURE 8: Mouse-Level Support & Phenotype Robustness in K=12
    # -----------------------------------------------------------------------
    print("\n--- Generating Figure 8: Mouse Support & K=12 Phenotype Robustness ---")
    # Tally mouse-level occupancy in Pooled K=8 and Pooled K=12
    # Organize animals
    animals = {}
    for key, meta in metadata_dict.items():
        aid = meta["animal_id"]
        gt = meta["genotype"]
        drug = meta["drug"]
        if aid not in animals:
            animals[aid] = {"genotype": gt, "sessions": {}}
        animals[aid]["sessions"][drug] = key
        
    wt_mice = [a for a, v in animals.items() if v["genotype"] == "WT"]
    ki_mice = [a for a, v in animals.items() if v["genotype"] == "KI"]
    
    def compute_mouse_support_table(states_dict, n_states, name="model"):
        rows = []
        for k in range(n_states):
            mouse_occs = []
            for aid, a_info in animals.items():
                for drug, sess_key in a_info["sessions"].items():
                    if sess_key in states_dict:
                        seq = states_dict[sess_key]
                        occ = np.mean(seq == k) * 100.0
                        mouse_occs.append(occ)
            mouse_occs = np.array(mouse_occs)
            n_contrib = np.sum(mouse_occs >= 2.0)
            rows.append({
                "model": name,
                "state": k,
                "mean_occupancy_pct": float(np.mean(mouse_occs)),
                "median_occupancy_pct": float(np.median(mouse_occs)),
                "num_mice_contrib_gt2pct": int(n_contrib),
                "total_eval_sessions": len(mouse_occs),
                "frac_mice_contrib": float(n_contrib / len(mouse_occs)),
            })
        return pd.DataFrame(rows)
        
    support_k8 = compute_mouse_support_table(models["Pooled_K8"]["states_dict"], 8, "Pooled_K8")
    support_k12 = compute_mouse_support_table(models["Pooled_K12"]["states_dict"], 12, "Pooled_K12")
    support_df = pd.concat([support_k8, support_k12], ignore_index=True)
    support_df.to_csv(os.path.join(TABLE_DIR, "mouse_level_state_support_table.csv"), index=False)
    
    # Phenotype test on K=12
    # In K=8, Z6 was WT-specific, Z1 was DAT-KI-dominant.
    # Find which K=12 states correspond to K=8 Z6 and Z1 using overlap matrix
    z6_children_k12 = np.where(C_k8_vs_k12_row[6] > 0.15)[0]
    z1_children_k12 = np.where(C_k8_vs_k12_row[1] > 0.15)[0]
    print(f"K=8 Z_6 maps to K=12 children: {z6_children_k12.tolist()}")
    print(f"K=8 Z_1 maps to K=12 children: {z1_children_k12.tolist()}")
    
    # Calculate pooled occupancy of Z6-children and Z1-children per mouse
    k12_states_dict = models["Pooled_K12"]["states_dict"]
    
    wt_sal_z6_sum = []
    ki_sal_z6_sum = []
    wt_sal_z1_sum = []
    ki_sal_z1_sum = []
    
    for aid in wt_mice:
        s_key = animals[aid]["sessions"]["Saline"]
        seq = k12_states_dict[s_key]
        wt_sal_z6_sum.append(np.mean(np.isin(seq, z6_children_k12)) * 100.0)
        wt_sal_z1_sum.append(np.mean(np.isin(seq, z1_children_k12)) * 100.0)
        
    for aid in ki_mice:
        s_key = animals[aid]["sessions"]["Saline"]
        seq = k12_states_dict[s_key]
        ki_sal_z6_sum.append(np.mean(np.isin(seq, z6_children_k12)) * 100.0)
        ki_sal_z1_sum.append(np.mean(np.isin(seq, z1_children_k12)) * 100.0)
        
    u_z6, p_z6_k12 = stats.mannwhitneyu(wt_sal_z6_sum, ki_sal_z6_sum)
    u_z1, p_z1_k12 = stats.mannwhitneyu(wt_sal_z1_sum, ki_sal_z1_sum)
    
    print(f"Phenotype in K=12:")
    print(f"  Z6-children (WT Saline): WT={np.mean(wt_sal_z6_sum):.1f}% vs KI={np.mean(ki_sal_z6_sum):.1f}% (p={p_z6_k12:.4f})")
    print(f"  Z1-children (KI Saline): WT={np.mean(wt_sal_z1_sum):.1f}% vs KI={np.mean(ki_sal_z1_sum):.1f}% (p={p_z1_k12:.4f})")
    
    fig8, axs8 = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    
    # Panel A: Mouse Support in Pooled K=8 vs K=12
    axs8[0].bar(np.arange(8) - 0.2, support_k8["num_mice_contrib_gt2pct"], width=0.4, label="Pooled K=8", color="#1f77b4", edgecolor="black", alpha=0.85)
    axs8[0].bar(np.arange(12) + 0.2, support_k12["num_mice_contrib_gt2pct"], width=0.4, label="Pooled K=12", color="#ff7f0e", edgecolor="black", alpha=0.85)
    axs8[0].set_title("A. Number of Contributing Mice per State (Occ >= 2%)", fontsize=11, fontweight="bold")
    axs8[0].set_xlabel("State ID ($Z_k$)", fontweight="bold")
    axs8[0].set_ylabel("Contributing Mice (out of 30 sessions)", fontweight="bold")
    axs8[0].set_xticks(range(12))
    axs8[0].legend()
    axs8[0].grid(axis="y", linestyle="--", alpha=0.4)
    
    # Panel B: Phenotype Robustness in K=12
    x_pos = [0, 1, 2.5, 3.5]
    vals = [np.mean(wt_sal_z6_sum), np.mean(ki_sal_z6_sum), np.mean(wt_sal_z1_sum), np.mean(ki_sal_z1_sum)]
    sems = [stats.sem(wt_sal_z6_sum), stats.sem(ki_sal_z6_sum), stats.sem(wt_sal_z1_sum), stats.sem(ki_sal_z1_sum)]
    colors = ["#2C7BB6", "#D7191C", "#2C7BB6", "#D7191C"]
    
    bars8 = axs8[1].bar(x_pos, vals, yerr=sems, color=colors, edgecolor="black", capsize=5, width=0.7, alpha=0.85)
    axs8[1].set_xticks(x_pos)
    axs8[1].set_xticklabels(["WT\nZ6-children", "KI\nZ6-children", "WT\nZ1-children", "KI\nZ1-children"], fontweight="bold")
    axs8[1].set_ylabel("Occupancy (%) in K=12 Model", fontweight="bold")
    axs8[1].set_title(f"B. Phenotype Preservation in K=12 (Saline)\nZ6-children p={p_z6_k12:.3f} | Z1-children p={p_z1_k12:.3f}", fontsize=11, fontweight="bold")
    axs8[1].grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig8_path = os.path.join(FIG_DIR, "fig8_per_mouse_support_and_k12_phenotype.png")
    plt.savefig(fig8_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 8 to:\n  {fig8_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
