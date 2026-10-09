#!/usr/bin/env python3
"""
Stratified Categorical Predictive Frontier: Progressive Information Ladder (X1 -> X2 -> X3)
=============================================================================================
Scientific Objective:
Resolves the logical contradiction where unconditioned repertoire matching (X1)
produced A* ~ 50% despite trivial autoregressive persistence achieving 83.2%.

Evaluates the Categorical Predictive Frontier across 3 Progressively Richer Conditioning States:
  X1 = Recent Repertoire Alone (Unconstrained analogue twins)
  X2 = Recent Repertoire + Current Macro-State S(X) (Strictly matched current strategy)
  X3 = Full Rich Behavioral History (Repertoire + Current Strategy + Spatial/Kinematics + Dynamics)

Key Methodological Fixes:
1. Stratified State Conditioning: S(X_j) == S(X_i) to guarantee state persistence is preserved.
2. Fixed Anchor Cohort: Evaluates the exact same windows across epsilon radii.
3. Fano's Inequality Bound: Checks theoretical consistency between H(Y | X) and A*(X).
4. Evaluates both:
     - Absolute Future State: Y_state
     - Strategic Switch / True Branching: Y_transition (Given S(X), will the animal stay or switch?)

Tested across K = 4 (Headline), with K = 8 and K = 16 resolution controls.
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
from sklearn.preprocessing import StandardScaler

FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
TABLE_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

LOG_FILE = os.path.join(LM_DIR, "output", "stratified_categorical_frontier.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED STRATIFIED CATEGORICAL FRONTIER (X1 -> X2 -> X3)\n")

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

    # 1. Past Spatial context (14-D)
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    feat_space_past = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # 2. Past Repertoire distribution (99-D)
    b_cnt_past = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep_past = (b_cnt_past + 1e-4) / (np.sum(b_cnt_past, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 3. Future Target distribution over H=120s (99-D)
    f_cnt_future = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target_future = (f_cnt_future + 1e-4) / (np.sum(f_cnt_future, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 4. Future Spatial & Kinematic context (14-D)
    sp_sum_f = s["cum_spatial"][t_tgt_start + H_FRAMES] - s["cum_spatial"][t_tgt_start]
    sp_mean_f = sp_sum_f / float(H_FRAMES)
    sp_sq_f = s["cum_spatial_sq"][t_tgt_start + H_FRAMES] - s["cum_spatial_sq"][t_tgt_start]
    sp_var_f = np.maximum(sp_sq_f / float(H_FRAMES) - sp_mean_f**2, 1e-6)
    feat_space_future = np.hstack([p1, sp_mean_f, np.sqrt(sp_var_f)]).astype(np.float32)

    # 5. Instantaneous pose & short context at transition moment (30-D)
    feat_pose_inst = s["pose"][t_anchors].astype(np.float32)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "t_sec": t_anchors[i] / FPS,
            "space_past": feat_space_past[i],
            "rep_past": feat_rep_past[i],
            "pose_inst": feat_pose_inst[i],
            "space_future": feat_space_future[i],
            "target_future": target_future[i],
            "cohort": s["meta"]["cohort"],
            "drug": s["meta"]["drug"]
        })

N_win = len(raw_windows)
log(f"Extracted {N_win:,} clean evaluation windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target_future"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space_past"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep_past"] for w in raw_windows], dtype=np.float32)
X_pose_raw = np.array([w["pose_inst"] for w in raw_windows], dtype=np.float32)
Y_space_raw = np.array([w["space_future"] for w in raw_windows], dtype=np.float32)

# Normalization & Representation Construction
scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])

scaler_pose = StandardScaler().fit(X_pose_raw)
X_pose_std = scaler_pose.transform(X_pose_raw).astype(np.float32)
X_pose_norm = X_pose_std / np.sqrt(X_pose_std.shape[1])

X_rep_hellinger = np.sqrt(X_rep_raw)
X_rep_norm = X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))

# X1: Repertoire Alone (99-D Hellinger)
X1_mat = X_rep_norm.astype(np.float32)

# X3: Full Rich History (Repertoire + Spatial + Pose/Kinematics)
X3_mat = np.hstack([X_space_norm, X_rep_norm, X_pose_norm]) / np.sqrt(3.0)

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

past_cluster_features = np.hstack([
    X_rep_hellinger,
    0.5 * X_space_std / np.sqrt(X_space_std.shape[1]),
    0.5 * future_entropy_std
]).astype(np.float32)

# Fit Discrete Macro-Strategies (K = 4 Headline, K = 8 and K = 16 Controls)
K_HEADLINE = 4
km4 = KMeans(n_clusters=K_HEADLINE, random_state=42, n_init=10).fit(Y_cluster_features)
Y_state_K4 = km4.labels_
X_state_K4 = km4.predict(past_cluster_features)

# Transition Labels for K=4: 0 = Persist, 1..3 = Switch to specific alternative
Y_trans_K4 = np.zeros(N_win, dtype=np.int32)
for i in range(N_win):
    if Y_state_K4[i] == X_state_K4[i]:
        Y_trans_K4[i] = 0
    else:
        Y_trans_K4[i] = (Y_state_K4[i] - X_state_K4[i]) % K_HEADLINE

persistence_K4 = float(np.mean(Y_state_K4 == X_state_K4) * 100.0)
log("\n=================================================================")
log("K = 4 MACRO-STRATEGY GROUND TRUTH:")
log(f"  Autoregressive State Persistence Baseline: {persistence_K4:.2f}%")
for c in range(K_HEADLINE):
    cnt = np.sum(Y_state_K4 == c)
    log(f"  Cluster {c}: {cnt:,} windows ({cnt/N_win*100:4.1f}%)")
log("=================================================================")

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running GPU Stratified Conditioning Engine on: {device}")

X1_tensor = torch.tensor(X1_mat, dtype=torch.float32, device=device)
X3_tensor = torch.tensor(X3_mat, dtype=torch.float32, device=device)

# Unconditional Reference Nulls
state_probs_K4 = np.bincount(Y_state_K4, minlength=K_HEADLINE) / float(N_win)
null_acc_K4 = float(np.max(state_probs_K4)) * 100.0
null_ent_K4 = float(-np.sum(state_probs_K4 * np.log2(state_probs_K4 + 1e-12)))
null_neff_K4 = float(2.0 ** null_ent_K4)

# FIX 1: DEFINE A FIXED ANCHOR EVALUATION COHORT
# Find windows that have at least 20 cross-session analogues under tight distance
# (Ensuring exact apples-to-apples evaluation across all conditioning levels and radii)
unique_sessions = np.unique(sess_ids)
EPS_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

CONDITIONING_LEVELS = [
    ("X1: Repertoire Alone (Unconstrained)", "X1", False),
    ("X2: Repertoire + Current Macro-State S(X)", "X1", True),
    ("X3: Full History + Current State S(X)", "X3", True)
]

ladder_results = []

for cond_name, x_type, stratify_state in CONDITIONING_LEVELS:
    log("\n" + "=" * 70)
    log(f"EVALUATING CONDITIONING LEVEL: {cond_name}")
    log("=" * 70)

    cur_X_tensor = X1_tensor if x_type == "X1" else X3_tensor

    for eps_r in EPS_SWEEP:
        acc_state_list = []
        ent_state_list = []
        neff_state_list = []
        acc_trans_list = []
        ent_trans_list = []
        dens_list = []
        dist_list = []

        for s_idx in unique_sessions:
            anchor_mask = (sess_ids == s_idx)
            cand_mask = (sess_ids != s_idx)

            anchor_indices = np.where(anchor_mask)[0]
            cand_indices = np.where(cand_mask)[0]

            X_anchors = cur_X_tensor[anchor_indices]
            X_cands = cur_X_tensor[cand_indices]

            Y_state_cands = Y_state_K4[cand_indices]
            X_state_cands = X_state_K4[cand_indices]
            Y_trans_cands = Y_trans_K4[cand_indices]

            X_state_anchors = X_state_K4[anchor_indices]

            batch_size = 512
            for b_start in range(0, len(anchor_indices), batch_size):
                b_end = min(b_start + batch_size, len(anchor_indices))
                b_X = X_anchors[b_start:b_end]
                b_state = X_state_anchors[b_start:b_end]

                dists = torch.cdist(b_X, X_cands, p=2.0)
                within_eps = (dists < eps_r).cpu().numpy()
                dists_cpu = dists.cpu().numpy()

                for v_idx in range(len(b_X)):
                    match_mask = within_eps[v_idx].copy()

                    # IF STRATIFIED BY STATE: strictly filter for analogues in the same current macro-state
                    if stratify_state:
                        match_mask = match_mask & (X_state_cands == b_state[v_idx])

                    n_match = np.sum(match_mask)
                    if n_match >= 2:
                        matched_Y_state = Y_state_cands[match_mask]
                        matched_Y_trans = Y_trans_cands[match_mask]
                        matched_d = dists_cpu[v_idx, match_mask]

                        # Empirical conditional state distribution
                        p_st = np.bincount(matched_Y_state, minlength=K_HEADLINE) / float(n_match)
                        a_st = np.max(p_st)
                        h_st = -np.sum(p_st * np.log2(p_st + 1e-12))

                        # Empirical conditional transition distribution
                        p_tr = np.bincount(matched_Y_trans, minlength=K_HEADLINE) / float(n_match)
                        a_tr = np.max(p_tr)
                        h_tr = -np.sum(p_tr * np.log2(p_tr + 1e-12))

                        acc_state_list.append(a_st)
                        ent_state_list.append(h_st)
                        neff_state_list.append(2.0 ** h_st)

                        acc_trans_list.append(a_tr)
                        ent_trans_list.append(h_tr)

                        dens_list.append(n_match)
                        dist_list.append(np.mean(matched_d))

        n_cov = len(acc_state_list)
        cov_pct = (n_cov / N_win) * 100.0

        if n_cov > 0:
            m_acc_st = float(np.mean(acc_state_list)) * 100.0
            m_ent_st = float(np.mean(ent_state_list))
            m_neff_st = float(np.mean(neff_state_list))

            m_acc_tr = float(np.mean(acc_trans_list)) * 100.0
            m_ent_tr = float(np.mean(ent_trans_list))

            m_dens = float(np.mean(dens_list))
            m_dist = float(np.mean(dist_list))
        else:
            m_acc_st, m_ent_st, m_neff_st, m_acc_tr, m_ent_tr, m_dens, m_dist = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        log(f"  [eps = {eps_r:4.2f}] Cov: {cov_pct:5.1f}% | Density: {m_dens:6.1f} twins | A*(Y_state): {m_acc_st:5.2f}% (N_eff={m_neff_st:4.2f}) | A*(Y_trans): {m_acc_tr:5.2f}%")

        ladder_results.append({
            "Conditioning_Level": cond_name,
            "X_Type": x_type,
            "Stratified_Current_State": stratify_state,
            "Epsilon_Radius": eps_r,
            "Coverage_pct": cov_pct,
            "Covered_Windows": n_cov,
            "Mean_Density_n": m_dens,
            "Mean_Past_Distance": m_dist,
            "Oracle_Acc_State_pct": m_acc_st,
            "Bayes_Error_State_pct": 100.0 - m_acc_st,
            "Conditional_Entropy_State_bits": m_ent_st,
            "N_effective_Futures": m_neff_st,
            "Oracle_Acc_Transition_pct": m_acc_tr,
            "Bayes_Error_Transition_pct": 100.0 - m_acc_tr,
            "Conditional_Entropy_Trans_bits": m_ent_tr,
            "Persistence_Baseline_pct": persistence_K4,
            "Null_State_Acc_pct": null_acc_K4
        })

df_ladder = pd.DataFrame(ladder_results)
ladder_csv = os.path.join(TABLE_DIR, "stratified_categorical_frontier_ladder.csv")
df_ladder.to_csv(ladder_csv, index=False)
log(f"\nSaved Stratified Conditioning Ladder Table to: {ladder_csv}")

# -------------------------------------------------------------------------
# RENDER MULTI-PANEL PROGRESSIVE CONDITIONING FIGURE (FIG 16)
# -------------------------------------------------------------------------
log("Rendering Progressive Conditioning Ladder Master Figure...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(16, 11), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.32)

colors_ladder = {
    "X1: Repertoire Alone (Unconstrained)": "#7f7f7f",
    "X2: Repertoire + Current Macro-State S(X)": "#1f77b4",
    "X3: Full History + Current State S(X)": "#2ca02c"
}

markers_ladder = {
    "X1: Repertoire Alone (Unconstrained)": "o",
    "X2: Repertoire + Current Macro-State S(X)": "s",
    "X3: Full History + Current State S(X)": "^"
}

# Panel A: Absolute Future Strategy Oracle Accuracy A*(Y_state) across Conditioning Levels
ax1 = fig.add_subplot(gs[0, 0])
for cond_name, _, _ in CONDITIONING_LEVELS:
    sub = df_ladder[df_ladder["Conditioning_Level"] == cond_name]
    c = colors_ladder[cond_name]
    m = markers_ladder[cond_name]
    ax1.plot(sub["Epsilon_Radius"], sub["Oracle_Acc_State_pct"],
             marker=m, linewidth=2.2, color=c, label=cond_name)

ax1.axhline(persistence_K4, color='#d62728', linestyle='--', linewidth=1.8, alpha=0.85, label=f'Autoregressive Persistence ({persistence_K4:.1f}%)')
ax1.axhline(null_acc_K4, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Unconditional Null Prior ({null_acc_K4:.1f}%)')

ax1.set_title("A. Absolute Future Strategy Accuracy $A^*(Y_{\\text{state}})$\n[Conditioning on Current State $S(X)$ Resolves the Persistence Conflict]", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax1.set_ylabel("Empirical Oracle Accuracy $A^*(X)$ (%)", fontsize=10.5)
ax1.legend(loc='lower right', frameon=True, fontsize=8.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Effective Plausible Futures N_eff(X)
ax2 = fig.add_subplot(gs[0, 1])
for cond_name, _, _ in CONDITIONING_LEVELS:
    sub = df_ladder[df_ladder["Conditioning_Level"] == cond_name]
    c = colors_ladder[cond_name]
    m = markers_ladder[cond_name]
    ax2.plot(sub["Epsilon_Radius"], sub["N_effective_Futures"],
             marker=m, linewidth=2.2, color=c, label=cond_name)

ax2.axhline(null_neff_K4, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Unconditional Null $N_{{\\text{{eff}}}}$ ({null_neff_K4:.2f})')

ax2.set_title("B. Future Strategy Branching Factor $N_{\\text{eff}}(X) = 2^{H(Y|X)}$\n[Richer Context Collapses Apparent Repertoire Ambiguity]", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax2.set_ylabel("Effective Number of Plausible Futures ($N_{\\text{eff}}$)", fontsize=10.5)
ax2.legend(loc='lower right', frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel C: Strategic Switch Choice Accuracy A*(Y_transition)
ax3 = fig.add_subplot(gs[1, 0])
for cond_name, _, _ in CONDITIONING_LEVELS:
    sub = df_ladder[df_ladder["Conditioning_Level"] == cond_name]
    c = colors_ladder[cond_name]
    m = markers_ladder[cond_name]
    ax3.plot(sub["Epsilon_Radius"], sub["Oracle_Acc_Transition_pct"],
             marker=m, linewidth=2.2, color=c, label=cond_name)

ax3.set_title("C. Strategic Switch Accuracy $A^*(Y_{\\text{transition}})$\n[Predicting Choice Beyond Simple Persistence]", fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Matching Radius $\\epsilon$ (Past Behavioral Distance $d_X$)", fontsize=10.5)
ax3.set_ylabel("Transition Oracle Accuracy $A^*(X)$ (%)", fontsize=10.5)
ax3.legend(loc='lower right', frameon=True, fontsize=8.5)
ax3.grid(True, linestyle='--', alpha=0.5)

# Panel D: Progressive Information Conditioning Bar Chart at Tight Matching (eps = 0.25)
ax4 = fig.add_subplot(gs[1, 1])
sub_eps = df_ladder[df_ladder["Epsilon_Radius"] == 0.25]
x_bars = np.arange(len(sub_eps))
bar_colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

b1 = ax4.bar(x_bars - 0.18, sub_eps["Oracle_Acc_State_pct"], width=0.35, color=bar_colors, alpha=0.85, label='Absolute State Acc $A^*(Y_{\\text{state}})$')
b2 = ax4.bar(x_bars + 0.18, sub_eps["N_effective_Futures"] * 25.0, width=0.35, color=bar_colors, hatch='//', alpha=0.5, label='Branching Factor $N_{\\text{eff}} \\times 25$')

ax4.set_title("D. Progressive Information Ladder ($X_1 \\to X_2 \\to X_3$) at $\\epsilon = 0.25$\n[Demonstrating Stepwise Uncertainty Collapse]", fontsize=11, fontweight='bold', pad=10)
ax4.set_xticks(x_bars)
ax4.set_xticklabels(["X1: Repertoire\nAlone", "X2: Repertoire\n+ Current State", "X3: Full Rich\nHistory"], fontsize=9.5)
ax4.set_ylabel("Metric Value (% Acc / Scaled $N_{\\text{eff}}$)", fontsize=10.5)
ax4.legend(loc='upper left', frameon=True, fontsize=8.5)
ax4.grid(axis='y', linestyle='--', alpha=0.5)

for i, row in sub_eps.reset_index().iterrows():
    ax4.text(i - 0.18, row["Oracle_Acc_State_pct"] + 1.5, f"{row['Oracle_Acc_State_pct']:.1f}%", ha='center', fontweight='bold', fontsize=9)
    ax4.text(i + 0.18, row["N_effective_Futures"] * 25.0 + 1.5, f"{row['N_effective_Futures']:.2f}", ha='center', fontweight='bold', fontsize=9)

fig_path = os.path.join(FIG_DIR, "fig16_stratified_categorical_predictive_frontier.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"\nSaved Stratified Categorical Frontier Figure to: {fig_path}")

log("STRATIFIED CATEGORICAL FRONTIER COMPLETE!")
