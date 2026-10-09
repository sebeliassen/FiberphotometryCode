#!/usr/bin/env python3
"""
Neighborhood Density & Dataset Scaling Benchmark (Fixed k Sweep)
================================================================
Scientific Objective:
Tests how neighbor past-state distance d_k and held-out predictive risk R_k^pred
scale as the candidate dataset size increases from N = 10% -> 100% of sessions.

Fixed:
  - k = 50 (and k = 100) nearest neighbors for stable conditional density estimation.
Swept:
  - Dataset fraction N_frac in [0.10, 0.25, 0.50, 0.75, 1.00] of candidate sessions.
Measures:
  - d_k(N): Mean past-state Hellinger distance to the k closest cross-session neighbors.
  - R_k^pred(N): Held-out predictive risk on anchor futures D_JS(Y_i, Y_bar_N).
  - R_k^disp(N): Within-neighborhood local future dispersion (1/k) sum D_JS(Y_j, Y_bar_N).
"""

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

LOG_FILE = os.path.join(LM_DIR, "output", "density_scaling_sweep.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED DENSITY SCALING SWEEP\n")

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
    session_meta.append({
        "sess_idx": i,
        "inj_frame": m.get("injection_frame", 0),
    })

FPS = 80.025
W_FRAMES = int(68.3 * FPS)
H_FRAMES = int(120.0 * FPS)
STRIDE_FRAMES = int(5.0 * FPS)
NUM_ACTIVE_SYLLS = 99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log("Extracting clean windows with direct bout onsets...")
sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    sp = np.zeros((N, 6), dtype=np.float32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = b["syllable"] - 1 # 1..99 -> 0..98
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            sp[sf:ef+1, :4] = vec[:4]; sp[sf:ef+1, 4] = vec[10]; sp[sf:ef+1, 5] = vec[12]

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
        "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq, "inj_frame": session_meta[sess_idx]["inj_frame"]
    })

raw_windows = []
for s in sessions_data:
    N = s["N"]
    if N < W_FRAMES + H_FRAMES + 200: continue
    inj_f = s["inj_frame"]

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

    # 1. Spatial context (14-D)
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    feat_space = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # 2. Repertoire (99-D)
    b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 3. Target Future (99-D)
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "space": feat_space[i],
            "rep": feat_rep[i],
            "target": target[i]
        })

N_win = len(raw_windows)
log(f"Loaded {N_win:,} evaluation windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep"] for w in raw_windows], dtype=np.float32)

scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])
X_rep_hellinger = np.sqrt(X_rep_raw)

# Balanced Hybrid Space (50/50)
X_hybrid = np.hstack([
    X_space_norm,
    X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
]) / np.sqrt(2.0)

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0); Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True); Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log2(P / M), dim=-1) + torch.sum(Q * torch.log2(Q / M), dim=-1))

X_tensor = torch.tensor(X_hybrid, dtype=torch.float32, device=DEVICE)
Y_tensor = torch.tensor(Y_targets, dtype=torch.float32, device=DEVICE)

global_prior = Y_tensor.mean(dim=0, keepdim=True)
null_js_per_window = js_divergence_pt(Y_tensor, global_prior.repeat(N_win, 1)).cpu().numpy()
mean_null_js = float(np.mean(null_js_per_window))
log(f"Global Null Divergence: {mean_null_js:.5f} bits")

# Sweep over Dataset Fractions with Fixed k = 50 and k = 100
DATASET_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]
K_FIXED_VALUES = [50, 100]
unique_sessions = np.unique(sess_ids)

results_scaling = []

for k_val in K_FIXED_VALUES:
    log("\n=================================================================")
    log(f"RUNNING DENSITY SCALING SWEEP WITH FIXED k = {k_val}")
    log("=================================================================")

    for frac in DATASET_FRACS:
        np.random.seed(42) # Reproducible candidate sampling

        pred_risks = np.zeros(N_win, dtype=np.float32)
        dispersions = np.zeros(N_win, dtype=np.float32)
        mean_dists = np.zeros(N_win, dtype=np.float32)

        for s_idx in unique_sessions:
            anchor_mask = (sess_ids == s_idx)
            cand_mask_all = (sess_ids != s_idx)

            anchor_indices = np.where(anchor_mask)[0]
            cand_sessions = [s for s in unique_sessions if s != s_idx]

            # Subsample candidate sessions to fraction `frac`
            n_sub_sess = max(1, int(len(cand_sessions) * frac))
            chosen_cand_sess = np.random.choice(cand_sessions, size=n_sub_sess, replace=False)
            sub_cand_mask = np.isin(sess_ids, chosen_cand_sess)
            cand_indices = np.where(sub_cand_mask)[0]

            if len(cand_indices) < k_val:
                effective_k = len(cand_indices)
            else:
                effective_k = k_val

            X_anchors = X_tensor[anchor_indices]
            X_cands = X_tensor[cand_indices]
            Y_anchors = Y_tensor[anchor_indices]
            Y_cands = Y_tensor[cand_indices]

            batch_size = 512
            for b_start in range(0, len(anchor_indices), batch_size):
                b_end = min(b_start + batch_size, len(anchor_indices))
                b_X = X_anchors[b_start:b_end]
                b_Y = Y_anchors[b_start:b_end]
                b_global_idx = anchor_indices[b_start:b_end]

                dists = torch.cdist(b_X, X_cands, p=2.0)
                topk_dists, topk_idx = torch.topk(dists, k=effective_k, largest=False, dim=1)

                neigh_Y = Y_cands[topk_idx] # [B, effective_k, 99]
                Y_bar_N = torch.mean(neigh_Y, dim=1)

                # 1. Held-Out Predictive Risk
                risk_pred = js_divergence_pt(b_Y, Y_bar_N)

                # 2. Local Future Dispersion
                Y_bar_expanded = Y_bar_N.unsqueeze(1).repeat(1, effective_k, 1)
                disp_k = js_divergence_pt(neigh_Y.reshape(-1, NUM_ACTIVE_SYLLS), Y_bar_expanded.reshape(-1, NUM_ACTIVE_SYLLS))
                disp_k = disp_k.reshape(-1, effective_k).mean(dim=1)

                pred_risks[b_global_idx] = risk_pred.cpu().numpy()
                dispersions[b_global_idx] = disp_k.cpu().numpy()
                mean_dists[b_global_idx] = topk_dists.mean(dim=1).cpu().numpy()

        # Session-clustered metrics
        sess_gains, sess_disps, sess_dists = [], [], []
        for s_idx in unique_sessions:
            s_mask = (sess_ids == s_idx)
            null_s = np.mean(null_js_per_window[s_mask])
            risk_s = np.mean(pred_risks[s_mask])
            disp_s = np.mean(dispersions[s_mask])
            dist_s = np.mean(mean_dists[s_mask])

            sess_gains.append(((null_s - risk_s) / null_s) * 100.0)
            sess_disps.append(disp_s)
            sess_dists.append(dist_s)

        m_gain, ci_gain = float(np.mean(sess_gains)), float(1.96 * np.std(sess_gains) / np.sqrt(len(sess_gains)))
        m_disp, ci_disp = float(np.mean(sess_disps)), float(1.96 * np.std(sess_disps) / np.sqrt(len(sess_disps)))
        m_risk = float(np.mean([np.mean(pred_risks[sess_ids == s]) for s in unique_sessions]))
        m_dist = float(np.mean(sess_dists))

        log(f"  [Frac = {frac*100:3.0f}% ({int(29*frac):2d} sess)] Past Dist: {m_dist:.4f} | Risk: {m_risk:.4f} bits | Gain: +{m_gain:5.2f}% ± {ci_gain:4.2f}% | Disp: {m_disp:.4f} bits")

        results_scaling.append({
            "k_neighbors": k_val,
            "Dataset_Fraction": frac,
            "Num_Candidate_Sessions": int(29 * frac),
            "Mean_Past_Distance": m_dist,
            "Predictive_Risk_bits": m_risk,
            "Predictive_Gain_pct": m_gain,
            "Gain_CI_95": ci_gain,
            "Local_Dispersion_bits": m_disp,
            "Dispersion_CI_95": ci_disp
        })

df_scaling = pd.DataFrame(results_scaling)
csv_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "density_scaling_sweep_results.csv")
df_scaling.to_csv(csv_path, index=False)
log(f"\nSaved scaling results to: {csv_path}")

# Plot Figure: Predictive Risk & Neighbor Distance vs. Dataset Size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for k_val, c, m in [(50, "#1f77b4", "o"), (100, "#2ca02c", "s")]:
    sub = df_scaling[df_scaling["k_neighbors"] == k_val].sort_values("Dataset_Fraction")

    # Left: Past Distance vs Dataset Fraction
    ax1.plot(sub["Dataset_Fraction"] * 100, sub["Mean_Past_Distance"], marker=m, linewidth=2.2, markersize=8, label=f"k = {k_val} Neighbors", color=c)

    # Right: Predictive Risk vs Past Distance (Asymptote curve)
    ax2.plot(sub["Mean_Past_Distance"], sub["Predictive_Risk_bits"], marker=m, linewidth=2.2, markersize=8, label=f"k = {k_val} Held-Out Risk $\\hat{{R}}_{{pred}}$", color=c)
    ax2.plot(sub["Mean_Past_Distance"], sub["Local_Dispersion_bits"], marker=m, linewidth=1.8, linestyle='--', markersize=6, alpha=0.7, label=f"k = {k_val} Local Dispersion $\\hat{{R}}_{{disp}}$", color=c)

ax1.set_title("A. Past-State Distance Tightens With Dataset Scale\n(Increasing $N$ tightens past matching $\\bar{d}_k \\rightarrow 0$ with fixed $k$)", fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel("Candidate Dataset Fraction Available (%)", fontsize=11)
ax1.set_ylabel("Mean Past Distance to $k$ Nearest Neighbors $\\bar{d}_k$", fontsize=11)
ax1.legend(loc='upper right', frameon=True, fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Reference lines on right panel
ax2.axhline(0.2751, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Population Null (0.275 bits)')
ax2.axhline(0.1804, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8, label='Transformer Error (0.180 bits)')

ax2.set_title("B. Predictive Risk vs. Past Distance Under Dataset Scaling\n(Fixed $k=50, 100$: Estimating Empirical Risk Asymptote)", fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel("Mean Past Distance $\\bar{d}_k$ (Tightens as Dataset Grows $\\rightarrow$)", fontsize=11)
ax2.set_ylabel("Divergence / Risk (bits, $D_{JS}$)", fontsize=11)
ax2.legend(loc='lower left', frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(LM_DIR, "output", "presentation", "figures", "fig7_density_scaling_asymptote.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"Saved Figure to: {fig_path}")
log("SCALING SWEEP COMPLETE!")
