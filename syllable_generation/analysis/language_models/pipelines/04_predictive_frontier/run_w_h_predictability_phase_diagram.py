#!/usr/bin/env python3
"""
W x H Predictability Phase Diagram (History Horizon vs. Forecast Horizon)
========================================================================
Scientific Objective:
Evaluates how predictability changes across:
  - Amount of measured behavioral past: W in {10s, 30s, 68.3s, 120s}
  - Forecast horizon into the future:   H in {5s, 15s, 30s, 60s, 120s}

Computes the Normalized Empirical Predictability Index:
    Resolved Fraction(W, H) = [ R_null(H) - R_cond(W, H) ] / [ R_null(H) - R_window(H) ]

Produces:
1. Complete W x H empirical matrix CSV table
2. Publication-quality 2D Phase Diagram Heatmap (fig14_predictability_phase_diagram_W_H.png)
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
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
TABLE_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

LOG_FILE = os.path.join(LM_DIR, "output", "w_h_predictability_phase_diagram.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED W x H PREDICTABILITY PHASE DIAGRAM\n")

FPS = 80.025
NUM_ACTIVE_SYLLS = 99
W_SECS = [10.0, 30.0, 68.3, 120.0]
H_SECS = [5.0, 15.0, 30.0, 60.0, 120.0]

# Load precomputed R_window from Step 1
floor_csv = os.path.join(TABLE_DIR, "finite_window_noise_floor_results.csv")
if os.path.exists(floor_csv):
    df_floor = pd.read_csv(floor_csv)
    R_WINDOW_DICT = dict(zip(df_floor["Horizon_sec"], df_floor["R_window_BoutBootstrap_bits"]))
    log(f"Loaded finite-window noise floors: {R_WINDOW_DICT}")
else:
    # Fallbacks from run_finite_window_noise_floor.py
    R_WINDOW_DICT = {5.0: 0.77625, 15.0: 0.55704, 30.0: 0.38774, 60.0: 0.23962, 120.0: 0.13761}

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
        "inj_frame": m.get("injection_frame", 0)
    })

log("Preprocessing session timecourses with direct discrete bout onsets...")
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
        s_id = b["syllable"] - 1
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            sp[sf:ef+1, :4] = vec[:4]
            sp[sf:ef+1, 4] = vec[10]
            sp[sf:ef+1, 5] = vec[12]
            
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
        "cum_spatial_sq": cum_spatial_sq, "meta": session_meta[sess_idx]
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running W x H Phase Diagram Engine on: {device}")

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0)
    Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    kl1 = torch.sum(P * torch.log2(P / M), dim=-1)
    kl2 = torch.sum(Q * torch.log2(Q / M), dim=-1)
    return 0.5 * (kl1 + kl2)

matrix_results = []
STRIDE_SEC = 10.0 # Fast 10s stride for complete 2D grid sweep
STRIDE_FRAMES = int(STRIDE_SEC * FPS)

for W_sec in W_SECS:
    W_frames = int(W_sec * FPS)
    
    for H_sec in H_SECS:
        H_frames = int(H_sec * FPS)
        log(f"\nEvaluating Cell: History W = {W_sec:5.1f}s | Forecast Horizon H = {H_sec:5.1f}s...")
        
        # Extract paired windows for this (W, H)
        windows = []
        for s in sessions_data:
            N = s["N"]
            if N < W_frames + H_frames + 200: continue
            inj_f = s["meta"]["inj_frame"]
            
            t_anchors = np.arange(W_frames, N - H_frames, STRIDE_FRAMES)
            t_tgt_start = s["next_bout_ptrs"][t_anchors]
            valid = (t_tgt_start + H_frames < N)
            t_anchors = t_anchors[valid]
            t_tgt_start = t_tgt_start[valid]
            
            past_start = t_anchors - W_frames
            future_end = t_tgt_start + H_frames
            no_straddle = (future_end < inj_f) | (past_start > inj_f)
            t_anchors = t_anchors[no_straddle]
            t_tgt_start = t_tgt_start[no_straddle]
            M = len(t_anchors)
            if M == 0: continue
            
            # Past Repertoire (99-D)
            b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_frames]
            feat_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)
            
            # Spatial context (14-D)
            t_norm = t_anchors / float(N)
            p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
            sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_frames]
            sp_mean = sp_sum / float(W_frames)
            sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_frames]
            sp_var = np.maximum(sp_sq / float(W_frames) - sp_mean**2, 1e-6)
            feat_space = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)
            
            # Target Repertoire in H (99-D)
            f_cnt = s["cum_bout"][t_tgt_start + H_frames] - s["cum_bout"][t_tgt_start]
            target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)
            
            for i in range(M):
                windows.append({
                    "sess_idx": s["sess_idx"],
                    "space": feat_space[i],
                    "rep": feat_rep[i],
                    "target": target[i]
                })
                
        N_w = len(windows)
        if N_w < 500:
            log(f"  Insufficient windows ({N_w}), skipping.")
            continue
            
        sess_ids = np.array([w["sess_idx"] for w in windows])
        Y_targets = np.array([w["target"] for w in windows], dtype=np.float32)
        X_space_raw = np.array([w["space"] for w in windows], dtype=np.float32)
        X_rep_raw = np.array([w["rep"] for w in windows], dtype=np.float32)
        
        scaler = StandardScaler().fit(X_space_raw)
        X_space_std = scaler.transform(X_space_raw).astype(np.float32)
        X_sp_norm = X_space_std / np.sqrt(X_space_std.shape[1])
        X_rep_hellinger = np.sqrt(X_rep_raw)
        X_rep_norm = X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
        X_hybrid = np.hstack([X_sp_norm, X_rep_norm]) / np.sqrt(2.0)
        
        X_tensor = torch.tensor(X_hybrid, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_targets, dtype=torch.float32, device=device)
        
        # 1. Unconditional Null (Random pairs)
        global_prior = Y_tensor.mean(dim=0, keepdim=True)
        r_null_arr = js_divergence_pt(Y_tensor, global_prior.repeat(N_w, 1)).cpu().numpy()
        R_null_H = float(np.mean(r_null_arr))
        
        # 2. Matched Twin Conditional Dispersion (Top 5% closest analogues / k=20)
        k_eval = 20
        disp_list = []
        unique_s = np.unique(sess_ids)
        
        for s_idx in unique_s:
            a_idx = np.where(sess_ids == s_idx)[0]
            c_idx = np.where(sess_ids != s_idx)[0]
            if len(a_idx) == 0 or len(c_idx) < k_eval: continue
            
            X_a = X_tensor[a_idx]
            X_c = X_tensor[c_idx]
            Y_c = Y_tensor[c_idx]
            
            dists = torch.cdist(X_a, X_c, p=2.0)
            _, topk_idx = torch.topk(dists, k=k_eval, largest=False, dim=1)
            neigh_Y = Y_c[topk_idx] # [B, k, 99]
            
            Y_bar = torch.mean(neigh_Y, dim=1, keepdim=True)
            disp_k = js_divergence_pt(neigh_Y.reshape(-1, NUM_ACTIVE_SYLLS), Y_bar.repeat(1, k_eval, 1).reshape(-1, NUM_ACTIVE_SYLLS))
            disp_k = disp_k.reshape(-1, k_eval).mean(dim=1)
            disp_list.append(disp_k.cpu().numpy())
            
        all_disp = np.concatenate(disp_list)
        R_cond = float(np.mean(all_disp))
        
        R_win = R_WINDOW_DICT.get(H_sec, 0.13761)
        
        # Normalized Predictability Index
        resolved_fraction = max(0.0, min(100.0, ((R_null_H - R_cond) / max(1e-4, (R_null_H - R_win))) * 100.0))
        raw_reduction = ((R_null_H - R_cond) / R_null_H) * 100.0
        
        log(f"  [W={W_sec:5.1f}s, H={H_sec:5.1f}s] R_null: {R_null_H:.4f} b | R_cond: {R_cond:.4f} b | R_win: {R_win:.4f} b | Resolved: {resolved_fraction:5.1f}% (Raw: -{raw_reduction:4.1f}%)")
        
        matrix_results.append({
            "History_W_sec": W_sec,
            "Horizon_H_sec": H_sec,
            "Windows_Evaluated": N_w,
            "R_null_bits": R_null_H,
            "R_conditional_bits": R_cond,
            "R_window_bits": R_win,
            "Resolved_Fraction_pct": resolved_fraction,
            "Raw_Reduction_pct": raw_reduction
        })

df_mat = pd.DataFrame(matrix_results)
mat_csv = os.path.join(TABLE_DIR, "w_h_predictability_phase_diagram.csv")
df_mat.to_csv(mat_csv, index=False)
log(f"\nSaved W x H matrix results to: {mat_csv}")

# -------------------------------------------------------------------------
# RENDER 2D HEATMAP / PHASE DIAGRAM
# -------------------------------------------------------------------------
pivot_resolved = df_mat.pivot(index="History_W_sec", columns="Horizon_H_sec", values="Resolved_Fraction_pct")
pivot_disp = df_mat.pivot(index="History_W_sec", columns="Horizon_H_sec", values="R_conditional_bits")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

w_labels = [f"{w:g}s" for w in W_SECS]
h_labels = [f"{h:g}s" for h in H_SECS]

# Panel 1: Normalized Resolved Fraction (%)
mat_resolved = pivot_resolved.values
im1 = ax1.imshow(mat_resolved, cmap="YlGnBu", aspect="auto", origin="lower")
plt.colorbar(im1, ax=ax1, label="Empirical Predictability Index (%)")

ax1.set_xticks(np.arange(len(H_SECS)))
ax1.set_xticklabels(h_labels)
ax1.set_yticks(np.arange(len(W_SECS)))
ax1.set_yticklabels(w_labels)

for i in range(len(W_SECS)):
    for j in range(len(H_SECS)):
        val = mat_resolved[i, j]
        color = "white" if val > 60 else "black"
        ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

ax1.set_title("A. Normalized Behavioral Predictability Index (% Gap Closed)\n$\\left( \\frac{R_{\\text{null}}(H) - R_{\\text{cond}}(W, H)}{R_{\\text{null}}(H) - R_{\\text{window}}(H)} \\times 100\\% \\right)$", fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel("Future Forecast Horizon $H$ (seconds)", fontsize=11)
ax1.set_ylabel("Measured History Conditioning Window $W$ (seconds)", fontsize=11)

# Panel 2: Raw Conditional Dispersion (bits)
mat_disp = pivot_disp.values
im2 = ax2.imshow(mat_disp, cmap="magma_r", aspect="auto", origin="lower")
plt.colorbar(im2, ax=ax2, label="Conditional Future Dispersion (bits, $D_{JS}$)")

ax2.set_xticks(np.arange(len(H_SECS)))
ax2.set_xticklabels(h_labels)
ax2.set_yticks(np.arange(len(W_SECS)))
ax2.set_yticklabels(w_labels)

for i in range(len(W_SECS)):
    for j in range(len(H_SECS)):
        val = mat_disp[i, j]
        color = "white" if val > 0.45 else "black"
        ax2.text(j, i, f"{val:.3f} b", ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)

ax2.set_title("B. Raw Conditional Future Dispersion $D(Y \\mid X \\approx x)$\n(Within-Neighborhood Future Divergence Among Matched Pasts)", fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel("Future Forecast Horizon $H$ (seconds)", fontsize=11)
ax2.set_ylabel("Measured History Conditioning Window $W$ (seconds)", fontsize=11)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig14_predictability_phase_diagram_W_H.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"Saved Phase Diagram Figure to: {fig_path}")

log("W x H PREDICTABILITY PHASE DIAGRAM COMPLETE!")
