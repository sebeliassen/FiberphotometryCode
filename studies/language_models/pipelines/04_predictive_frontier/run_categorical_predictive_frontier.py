#!/usr/bin/env python3
"""
Categorical Predictive Frontier Sweep for K in {4, 8, 16}
=========================================================
Neutral Strategy Labels (Cluster 0..K-1)
Full empirical tables and figures comparing:
- K = 4
- K = 8
- K = 16

Evaluates:
- Baseline Autoregressive State Persistence
- Unconditional Null Baselines (Accuracy, Entropy, N_eff)
- Epsilon-Radius Sweep (Density, Oracle Accuracy A*, Bayes Error R*, Conditional Entropy H, Effective Futures N_eff)
- Absolute State (Y_state) vs Strategic Transition (Y_transition)
"""

import os
import sys
import time

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))

import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
TABLE_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

LOG_FILE = os.path.join(LM_DIR, "output", "categorical_predictive_frontier_multi_k.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED CATEGORICAL FRONTIER MULTI-K SWEEP (K in 4, 8, 16)\n")

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_ACTIVE_SYLLS = 99

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
log(f"Loading continuous bout cache: {cache_path}")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

raw_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(raw_path, "rb") as f:
    raw_sessions = pickle.load(f)

session_meta = []
for i, s in enumerate(raw_sessions):
    m = getattr(s, "metadata", {})
    gt = m.get("genotype", "WT")
    drug = m.get("drug", "Saline")
    session_meta.append({
        "sess_idx": i,
        "cohort": f"{gt}_{drug}",
        "drug": drug,
        "genotype": gt,
        "inj_frame": m.get("injection_frame", 0),
        "entry_frame": m.get("entry_frame", 0)
    })

log("Preprocessing session timecourses with direct discrete bout onsets...")
sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    sp = np.zeros((N, 6), dtype=np.float32)
    pose = np.zeros((N, 30), dtype=np.float32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_raw = b["syllable"]
        s_id = s_raw - 1
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0

        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            sp[sf:ef+1, :4] = vec[:4]
            sp[sf:ef+1, 4] = vec[10]
            sp[sf:ef+1, 5] = vec[12]
            if len(vec) >= 30:
                pose[sf:ef+1] = vec[:30]
            else:
                pose[sf:ef+1, :len(vec)] = vec

    b_starts = np.array([b["start_frame"] for b in bouts])
    for t_idx, b_st in enumerate(b_starts):
        prev_st = b_starts[t_idx-1] if t_idx > 0 else 0
        next_bout_ptrs[prev_st:b_st] = b_st
    if len(b_starts) > 0:
        next_bout_ptrs[b_starts[-1]:] = N

    cum_bout = np.vstack([np.zeros((1, NUM_ACTIVE_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
    cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp**2, axis=0)])

    sessions_data.append({
        "sess_idx": sess_idx, "N": N, "fs": fs, "pose": pose, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq, "meta": session_meta[sess_idx]
    })

raw_windows = []
for s in sessions_data:
    N = s["N"]
    if N < W_FRAMES + H_FRAMES + 200: continue
    inj_f = s["meta"]["inj_frame"]

    t_anchors = np.arange(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES)
    t_tgt_start = s["next_bout_ptrs"][t_anchors]
    valid = (t_tgt_start + H_FRAMES < N)
    t_anchors = t_anchors[valid]
    t_tgt_start = t_tgt_start[valid]

    past_start = t_anchors - W_FRAMES
    future_end = t_tgt_start + H_FRAMES
    no_straddle = (future_end < inj_f) | (past_start > inj_f)
    t_anchors = t_anchors[no_straddle]
    t_tgt_start = t_tgt_start[no_straddle]
    M = len(t_anchors)
    if M == 0: continue

    # Past Spatial context (14-D)
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    feat_space_past = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # Past Repertoire distribution (99-D)
    b_cnt_past = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep_past = (b_cnt_past + 1e-4) / (np.sum(b_cnt_past, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # Future Target distribution over H=120s (99-D)
    f_cnt_future = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target_future = (f_cnt_future + 1e-4) / (np.sum(f_cnt_future, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # Future Spatial & Kinematic context (14-D)
    sp_sum_f = s["cum_spatial"][t_tgt_start + H_FRAMES] - s["cum_spatial"][t_tgt_start]
    sp_mean_f = sp_sum_f / float(H_FRAMES)
    sp_sq_f = s["cum_spatial_sq"][t_tgt_start + H_FRAMES] - s["cum_spatial_sq"][t_tgt_start]
    sp_var_f = np.maximum(sp_sq_f / float(H_FRAMES) - sp_mean_f**2, 1e-6)
    feat_space_future = np.hstack([p1, sp_mean_f, np.sqrt(sp_var_f)]).astype(np.float32)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "t_sec": t_anchors[i] / FPS,
            "space_past": feat_space_past[i],
            "rep_past": feat_rep_past[i],
            "space_future": feat_space_future[i],
            "target_future": target_future[i],
            "cohort": s["meta"]["cohort"],
            "drug": s["meta"]["drug"]
        })

N_win = len(raw_windows)
log(f"Extracted {N_win:,} clean windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target_future"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space_past"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep_past"] for w in raw_windows], dtype=np.float32)
Y_space_raw = np.array([w["space_future"] for w in raw_windows], dtype=np.float32)

# Hybrid past representation: Space + Repertoire (Hellinger)
scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])
X_rep_hellinger = np.sqrt(X_rep_raw)
X_rep_norm = X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
X_hybrid = np.hstack([X_space_norm, X_rep_norm]) / np.sqrt(2.0)

# Multi-modal future window feature vector for clustering Y
scaler_space_f = StandardScaler().fit(Y_space_raw)
Y_space_std = scaler_space_f.transform(Y_space_raw).astype(np.float32)
Y_rep_hellinger = np.sqrt(Y_targets)
future_entropy = -np.sum(Y_targets * np.log2(np.clip(Y_targets, 1e-12, 1.0)), axis=1, keepdims=True)
scaler_ent = StandardScaler().fit(future_entropy)
future_entropy_std = scaler_ent.transform(future_entropy).astype(np.float32)

Y_cluster_features = np.hstack([
    Y_rep_hellinger,
    0.5 * Y_space_std / np.sqrt(Y_space_std.shape[1]),
    0.5 * future_entropy_std
]).astype(np.float32)

past_features = np.hstack([
    X_rep_hellinger,
    0.5 * X_space_std / np.sqrt(X_space_std.shape[1]),
    0.5 * future_entropy_std
]).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running GPU-accelerated categorical engine on: {device}")

X_tensor = torch.tensor(X_hybrid, dtype=torch.float32, device=device)
EPSILON_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
unique_sessions = np.unique(sess_ids)

K_VALUES = [4, 8, 16]
all_k_sweep_results = []
all_k_models = {}

for K in K_VALUES:
    log("\n=================================================================")
    log(f"EVALUATING DISCRETE RESOLUTION: K = {K} MACRO-STRATEGIES")
    log("=================================================================")

    # 1. Measure Cross-Session Stability (ARI between 2 disjoint session halves)
    half1_sess = unique_sessions[:len(unique_sessions)//2]
    half2_sess = unique_sessions[len(unique_sessions)//2:]
    mask1 = np.isin(sess_ids, half1_sess)
    mask2 = np.isin(sess_ids, half2_sess)

    km1 = KMeans(n_clusters=K, random_state=42, n_init=5).fit(Y_cluster_features[mask1])
    km2 = KMeans(n_clusters=K, random_state=42, n_init=5).fit(Y_cluster_features[mask2])
    preds1 = km1.predict(Y_cluster_features)
    preds2 = km2.predict(Y_cluster_features)
    ari = adjusted_rand_score(preds1, preds2)

    # Subsampled silhouette
    sub_idx = np.random.choice(N_win, size=min(4000, N_win), replace=False)
    km_full = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Y_cluster_features)
    sil = silhouette_score(Y_cluster_features[sub_idx], km_full.labels_[sub_idx])

    Y_state_labels = km_full.labels_
    X_state_labels = km_full.predict(past_features)

    # Transition labels: 0 = Persist, 1..K-1 = Switch to modular target
    transition_labels = np.zeros(N_win, dtype=np.int32)
    for i in range(N_win):
        if Y_state_labels[i] == X_state_labels[i]:
            transition_labels[i] = 0
        else:
            transition_labels[i] = (Y_state_labels[i] - X_state_labels[i]) % K

    persistence_pct = float(np.mean(Y_state_labels == X_state_labels) * 100.0)

    # Unconditional Nulls
    state_counts = np.bincount(Y_state_labels, minlength=K)
    state_probs = state_counts / float(N_win)
    null_acc_state = float(np.max(state_probs)) * 100.0
    null_ent_state = float(-np.sum(state_probs * np.log2(state_probs + 1e-12)))
    null_neff_state = float(2.0 ** null_ent_state)

    trans_counts = np.bincount(transition_labels, minlength=K)
    trans_probs = trans_counts / float(N_win)
    null_acc_trans = float(np.max(trans_probs)) * 100.0
    null_ent_trans = float(-np.sum(trans_probs * np.log2(trans_probs + 1e-12)))

    log(f"  [K={K:2d}] Cross-Session ARI: {ari:.4f} | Silhouette: {sil:.4f}")
    log(f"  [K={K:2d}] Autoregressive State Persistence Baseline: {persistence_pct:.2f}%")
    log(f"  [K={K:2d}] Unconditional Null State Acc: {null_acc_state:.2f}% | Null Entropy: {null_ent_state:.3f} b | Null N_eff: {null_neff_state:.2f}")

    all_k_models[K] = {
        "Y_state": Y_state_labels,
        "X_state": X_state_labels,
        "Y_trans": transition_labels,
        "persistence_pct": persistence_pct,
        "null_acc_state": null_acc_state,
        "null_ent_state": null_ent_state,
        "null_neff_state": null_neff_state,
        "null_acc_trans": null_acc_trans,
        "null_ent_trans": null_ent_trans,
        "class_prevalence": state_probs * 100.0
    }

    # Sweep epsilon
    for eps_r in EPSILON_SWEEP:
        acc_state_arr = np.full(N_win, np.nan, dtype=np.float32)
        ent_state_arr = np.full(N_win, np.nan, dtype=np.float32)
        neff_state_arr = np.full(N_win, np.nan, dtype=np.float32)

        acc_trans_arr = np.full(N_win, np.nan, dtype=np.float32)
        ent_trans_arr = np.full(N_win, np.nan, dtype=np.float32)

        density_arr = np.zeros(N_win, dtype=np.int32)
        mean_dist_arr = np.full(N_win, np.nan, dtype=np.float32)

        for s_idx in unique_sessions:
            anchor_mask = (sess_ids == s_idx)
            cand_mask = (sess_ids != s_idx)

            anchor_indices = np.where(anchor_mask)[0]
            cand_indices = np.where(cand_mask)[0]

            X_anchors = X_tensor[anchor_indices]
            X_cands = X_tensor[cand_indices]

            Y_state_cands = Y_state_labels[cand_indices]
            Y_trans_cands = transition_labels[cand_indices]

            batch_size = 512
            for b_start in range(0, len(anchor_indices), batch_size):
                b_end = min(b_start + batch_size, len(anchor_indices))
                b_X = X_anchors[b_start:b_end]
                b_global_idx = anchor_indices[b_start:b_end]

                dists = torch.cdist(b_X, X_cands, p=2.0)
                within_eps = (dists < eps_r)
                counts = within_eps.sum(dim=1).cpu().numpy()
                density_arr[b_global_idx] = counts

                valid_b = np.where(counts >= 2)[0]
                for v_idx in valid_b:
                    mask_j = within_eps[v_idx].cpu().numpy()
                    matched_state = Y_state_cands[mask_j]
                    matched_trans = Y_trans_cands[mask_j]
                    matched_d = dists[v_idx, mask_j].cpu().numpy()

                    p_state = np.bincount(matched_state, minlength=K) / float(len(matched_state))
                    a_state = np.max(p_state)
                    h_state = -np.sum(p_state * np.log2(p_state + 1e-12))

                    p_trans = np.bincount(matched_trans, minlength=K) / float(len(matched_trans))
                    a_trans = np.max(p_trans)
                    h_trans = -np.sum(p_trans * np.log2(p_trans + 1e-12))

                    acc_state_arr[b_global_idx[v_idx]] = a_state
                    ent_state_arr[b_global_idx[v_idx]] = h_state
                    neff_state_arr[b_global_idx[v_idx]] = 2.0 ** h_state

                    acc_trans_arr[b_global_idx[v_idx]] = a_trans
                    ent_trans_arr[b_global_idx[v_idx]] = h_trans
                    mean_dist_arr[b_global_idx[v_idx]] = np.mean(matched_d)

        valid_mask = ~np.isnan(acc_state_arr)
        n_cov = np.sum(valid_mask)
        cov_pct = (n_cov / N_win) * 100.0

        if n_cov > 0:
            mean_acc_state = float(np.mean(acc_state_arr[valid_mask])) * 100.0
            mean_ent_state = float(np.mean(ent_state_arr[valid_mask]))
            mean_neff_state = float(np.mean(neff_state_arr[valid_mask]))
            mean_acc_trans = float(np.mean(acc_trans_arr[valid_mask])) * 100.0
            mean_ent_trans = float(np.mean(ent_trans_arr[valid_mask]))
            mean_dens = float(np.mean(density_arr[valid_mask]))
            mean_d = float(np.mean(mean_dist_arr[valid_mask]))
        else:
            mean_acc_state, mean_ent_state, mean_neff_state = np.nan, np.nan, np.nan
            mean_acc_trans, mean_ent_trans, mean_dens, mean_d = np.nan, np.nan, np.nan, np.nan

        log(f"    [eps={eps_r:4.2f}] Cov: {cov_pct:5.1f}% | A*(Y_state): {mean_acc_state:5.2f}% (N_eff={mean_neff_state:4.2f}) | A*(Y_trans): {mean_acc_trans:5.2f}%")

        all_k_sweep_results.append({
            "K_Strategies": K,
            "Epsilon_Radius": eps_r,
            "Coverage_pct": cov_pct,
            "Covered_Windows": n_cov,
            "Mean_Density_n": mean_dens,
            "Mean_Past_Distance": mean_d,
            "Oracle_Acc_State_pct": mean_acc_state,
            "Bayes_Error_State_pct": 100.0 - mean_acc_state,
            "Conditional_Entropy_State_bits": mean_ent_state,
            "N_effective_Futures": mean_neff_state,
            "Oracle_Acc_Transition_pct": mean_acc_trans,
            "Bayes_Error_Transition_pct": 100.0 - mean_acc_trans,
            "Conditional_Entropy_Trans_bits": mean_ent_trans,
            "State_Persistence_Baseline_pct": persistence_pct,
            "Null_State_Acc_pct": null_acc_state,
            "Null_Entropy_State_bits": null_ent_state,
            "Null_Neff_State": null_neff_state
        })

df_all_k = pd.DataFrame(all_k_sweep_results)
multi_k_csv = os.path.join(TABLE_DIR, "categorical_predictive_frontier_multi_k_results.csv")
df_all_k.to_csv(multi_k_csv, index=False)
log(f"\nSaved Multi-K sweep results table to: {multi_k_csv}")

# -------------------------------------------------------------------------
# STEP 3: RENDER MULTI-PANEL FIGURE FOR K in {4, 8, 16}
# -------------------------------------------------------------------------
log("Rendering Multi-K Categorical Predictive Frontier Figure...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(16, 11), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.32)

k_colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}
k_markers = {4: "o", 8: "s", 16: "^"}

# Panel A: Empirical Oracle Accuracy A*(Y_state) vs Epsilon for K in {4, 8, 16}
ax1 = fig.add_subplot(gs[0, 0])
for K in K_VALUES:
    sub = df_all_k[df_all_k["K_Strategies"] == K]
    c = k_colors[K]
    m = k_markers[K]
    null_val = all_k_models[K]["null_acc_state"]
    persist_val = all_k_models[K]["persistence_pct"]

    ax1.plot(sub["Epsilon_Radius"], sub["Oracle_Acc_State_pct"],
             marker=m, linewidth=2.2, color=c, label=f'K={K} A*(Y_state) [Null: {null_val:.1f}%, Persist: {persist_val:.1f}%]')

ax1.set_title("A. Empirical Bayes Oracle Accuracy A*(Y_state) vs. Past History Similarity ($\\epsilon$)\n[Comparing Macro-Strategy Resolutions $K \\in \\{4, 8, 16\\}$]", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax1.set_ylabel("Empirical Oracle Accuracy $A^*(X)$ (%)", fontsize=10.5)
ax1.legend(loc='lower right', frameon=True, fontsize=8.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Effective Plausible Futures N_eff(X) vs Epsilon for K in {4, 8, 16}
ax2 = fig.add_subplot(gs[0, 1])
for K in K_VALUES:
    sub = df_all_k[df_all_k["K_Strategies"] == K]
    c = k_colors[K]
    m = k_markers[K]
    null_neff = all_k_models[K]["null_neff_state"]

    ax2.plot(sub["Epsilon_Radius"], sub["N_effective_Futures"],
             marker=m, linewidth=2.2, color=c, label=f'K={K} $N_{{\\text{{eff}}}}(X)$ [Unconditional Null: {null_neff:.2f}]')

ax2.set_title("B. Future Strategy Branching Factor $N_{\\text{eff}}(X) = 2^{H(Y|X)}$\n[How Many Distinct Future Strategies Remain Plausible Given Past $X$]", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax2.set_ylabel("Effective Number of Plausible Futures ($N_{\\text{eff}}$)", fontsize=10.5)
ax2.legend(loc='lower right', frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel C: Strategic Transition Oracle Accuracy A*(Y_trans) vs Epsilon
ax3 = fig.add_subplot(gs[1, 0])
for K in K_VALUES:
    sub = df_all_k[df_all_k["K_Strategies"] == K]
    c = k_colors[K]
    m = k_markers[K]
    ax3.plot(sub["Epsilon_Radius"], sub["Oracle_Acc_Transition_pct"],
             marker=m, linewidth=2.2, color=c, label=f'K={K} A*(Y_transition)')

ax3.set_title("C. Strategic Switch Oracle Accuracy A*(Y_transition)\n[Evaluating Future Choice Beyond Autoregressive State Persistence]", fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax3.set_ylabel("Transition Oracle Accuracy $A^*(X)$ (%)", fontsize=10.5)
ax3.legend(loc='lower right', frameon=True, fontsize=8.5)
ax3.grid(True, linestyle='--', alpha=0.5)

# Panel D: Class Prevalence Profiles Across K in {4, 8, 16}
ax4 = fig.add_subplot(gs[1, 1])
x_offsets = [-0.25, 0.0, 0.25]
w_bar = 0.22

for idx, K in enumerate(K_VALUES):
    prev = all_k_models[K]["class_prevalence"]
    x_coords = np.arange(len(prev))
    ax4.bar(x_coords + x_offsets[idx], prev, width=w_bar, color=k_colors[K], alpha=0.75, label=f'K={K} Strategy Prevalence')

ax4.set_title("D. Discrete Class Distribution Across Resolutions\n[Neutral Labels: Cluster 0 .. Cluster K-1]", fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel("Discrete Macro-Strategy Index (Cluster $0 \\dots K-1$)", fontsize=10.5)
ax4.set_ylabel("Class Prevalence (% of Windows)", fontsize=10.5)
ax4.set_xticks(np.arange(16))
ax4.set_xticklabels([f"C{i}" for i in range(16)], fontsize=8.5)
ax4.legend(loc='upper right', frameon=True, fontsize=8.5)
ax4.grid(axis='y', linestyle='--', alpha=0.5)

fig_path = os.path.join(FIG_DIR, "fig15_categorical_predictive_frontier.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"\nSaved Multi-K Categorical Predictive Frontier Figure to: {fig_path}")

log("CATEGORICAL MULTI-K SWEEP COMPLETE!")
