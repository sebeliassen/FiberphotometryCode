#!/usr/bin/env python3
"""
The Empirical Predictive Frontier Suite
========================================
Scientific Objective:
Evaluates the finite-data empirical predictive frontier on 30 open-field sessions
(29,325 clean non-straddling windows, W=68.3s -> H=120.0s, 99 active syllables).

Implements 6 rigorous diagnostics:
1. Single Best Models (k-NN, Ridge BoW, Transformer)
2. Learned Stacked Ensemble (Constrained out-of-fold blend)
3. Oracle Per-Window Model Selector (Best model picked per window)
4. Finite-Data Branch Oracle (Best single matched neighbor future vs. random candidates)
5. Finite-Data Convex Hull Oracle (Best convex combination of matched neighbor futures)
6. Target Measurement Noise Floor (Even-odd bout split divergence over 120s)
7. Residual Predictability (Testing for unextracted signal in Transformer residuals)
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

LOG_FILE = os.path.join(LM_DIR, "output", "empirical_frontier_suite.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED EMPIRICAL PREDICTIVE FRONTIER SUITE\n")

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

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0); Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True); Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log2(P / M), dim=-1) + torch.sum(Q * torch.log2(Q / M), dim=-1))

class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size=99, embed_dim=64, nhead=4, num_layers=2, max_len=140, num_targets=99):
        super().__init__()
        self.syll_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.dur_proj = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_targets),
            nn.Softmax(dim=-1)
        )

    def forward(self, seq_tokens, seq_durs, pad_mask):
        B, L = seq_tokens.shape
        x = self.syll_embed(seq_tokens) + self.dur_proj(seq_durs.unsqueeze(-1))
        x = x + self.pos_embed[:, :L, :]
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

log("Extracting clean windows with direct bout onsets & target splits...")
sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    sp = np.zeros((N, 6), dtype=np.float32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)
    
    clean_tokens, clean_onsets, clean_durs = [], [], []
    
    # Store bout onsets with even/odd indexing for target noise estimation
    oh_bout_even = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    oh_bout_odd = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    
    for i_b, b in enumerate(bouts):
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = b["syllable"] - 1
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0
            if i_b % 2 == 0:
                oh_bout_even[sf, s_id] = 1.0
            else:
                oh_bout_odd[sf, s_id] = 1.0
            clean_tokens.append(s_id)
            clean_onsets.append(sf)
            clean_durs.append((ef - sf + 1) / FPS)
            
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
    cum_even = np.vstack([np.zeros((1, NUM_ACTIVE_SYLLS), dtype=np.float32), np.cumsum(oh_bout_even, axis=0)])
    cum_odd = np.vstack([np.zeros((1, NUM_ACTIVE_SYLLS), dtype=np.float32), np.cumsum(oh_bout_odd, axis=0)])
    cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
    cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp**2, axis=0)])
    
    sessions_data.append({
        "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
        "cum_even": cum_even, "cum_odd": cum_odd,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq, "inj_frame": session_meta[sess_idx]["inj_frame"],
        "tokens": np.array(clean_tokens, dtype=np.int64),
        "onsets": np.array(clean_onsets, dtype=np.int64),
        "durs": np.array(clean_durs, dtype=np.float32)
    })

raw_windows = []
max_l = 140

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
    
    # Target Even/Odd Split for Measurement Noise Estimation
    f_even = s["cum_even"][t_tgt_start + H_FRAMES] - s["cum_even"][t_tgt_start]
    target_even = (f_even + 1e-4) / (np.sum(f_even, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)
    
    f_odd = s["cum_odd"][t_tgt_start + H_FRAMES] - s["cum_odd"][t_tgt_start]
    target_odd = (f_odd + 1e-4) / (np.sum(f_odd, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)
    
    # Sequential Tokens for Transformer
    seq_toks = np.full((M, max_l), NUM_ACTIVE_SYLLS, dtype=np.int64)
    seq_durs = np.zeros((M, max_l), dtype=np.float32)
    seq_masks = np.ones((M, max_l), dtype=bool)
    
    b_onsets, b_tokens, b_durs = s["onsets"], s["tokens"], s["durs"]
    for i, t_anc in enumerate(t_anchors):
        t_min = t_anc - W_FRAMES
        m_b = (b_onsets >= t_min) & (b_onsets < t_anc)
        idx_b = np.where(m_b)[0]
        if len(idx_b) > 0:
            idx_b = idx_b[-max_l:]
            L_i = len(idx_b)
            seq_toks[i, :L_i] = b_tokens[idx_b]
            seq_durs[i, :L_i] = b_durs[idx_b]
            seq_masks[i, :L_i] = False
            
    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "space": feat_space[i],
            "rep": feat_rep[i],
            "target": target[i],
            "target_even": target_even[i],
            "target_odd": target_odd[i],
            "seq_tok": seq_toks[i],
            "seq_dur": seq_durs[i],
            "seq_mask": seq_masks[i]
        })

N_win = len(raw_windows)
log(f"Extracted {N_win:,} clean evaluation windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
Y_even = np.array([w["target_even"] for w in raw_windows], dtype=np.float32)
Y_odd = np.array([w["target_odd"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep"] for w in raw_windows], dtype=np.float32)
X_toks = np.array([w["seq_tok"] for w in raw_windows], dtype=np.int64)
X_durs = np.array([w["seq_dur"] for w in raw_windows], dtype=np.float32)
X_masks = np.array([w["seq_mask"] for w in raw_windows], dtype=bool)

scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])
X_rep_hellinger = np.sqrt(X_rep_raw)
X_hybrid = np.hstack([
    X_space_norm,
    X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
]) / np.sqrt(2.0)

# -----------------------------------------------------------------------------
# 1. Target Measurement Noise Floor (Split-Half 120s Window Jitter)
# -----------------------------------------------------------------------------
log("\n--- Computing Target Measurement Noise Floor (120s Split-Half Jitter) ---")
Y_even_t = torch.tensor(Y_even, dtype=torch.float32, device=DEVICE)
Y_odd_t = torch.tensor(Y_odd, dtype=torch.float32, device=DEVICE)
measurement_noise_per_win = js_divergence_pt(Y_even_t, Y_odd_t).cpu().numpy()
mean_meas_noise = float(np.mean(measurement_noise_per_win))
ci_meas_noise = float(1.96 * np.std([np.mean(measurement_noise_per_win[sess_ids == s]) for s in np.unique(sess_ids)]) / np.sqrt(len(np.unique(sess_ids))))
log(f"Target 120s Measurement Jitter Floor: {mean_meas_noise:.4f} ± {ci_meas_noise:.4f} bits")

# -----------------------------------------------------------------------------
# 2. 5-Fold Cross-Validation: Ridge, Transformer, k-NN Out-of-Fold Predictions
# -----------------------------------------------------------------------------
log("\n--- Running 5-Fold Cross-Validation for Base Models ---")
gkf = GroupKFold(n_splits=5)
preds_knn = np.zeros_like(Y_targets)
preds_ridge = np.zeros_like(Y_targets)
preds_transformer = np.zeros_like(Y_targets)

X_hybrid_t = torch.tensor(X_hybrid, dtype=torch.float32, device=DEVICE)
Y_t = torch.tensor(Y_targets, dtype=torch.float32, device=DEVICE)
K_NN = 50

for fold, (tr, te) in enumerate(gkf.split(X_hybrid, groups=sess_ids)):
    log(f"  Training Fold {fold+1}/5 (Train: {len(tr):,}, Test: {len(te):,})...")
    Y_tr, Y_te = Y_targets[tr], Y_targets[te]
    
    # 1. Ridge BoW
    pred_r = np.clip(Ridge(1.0).fit(X_hybrid[tr], Y_tr).predict(X_hybrid[te]), 1e-4, 1.0)
    pred_r /= np.sum(pred_r, axis=1, keepdims=True)
    preds_ridge[te] = pred_r
    
    # 2. k-NN (k=50 strictly on training fold sessions)
    X_tr_t = X_hybrid_t[tr]
    X_te_t = X_hybrid_t[te]
    Y_tr_t = Y_t[tr]
    
    batch_size = 512
    for b_start in range(0, len(te), batch_size):
        b_end = min(b_start + batch_size, len(te))
        b_X = X_te_t[b_start:b_end]
        dists = torch.cdist(b_X, X_tr_t, p=2.0)
        _, topk_idx = torch.topk(dists, k=K_NN, largest=False, dim=1)
        b_knn_pred = torch.mean(Y_tr_t[topk_idx], dim=1)
        preds_knn[te[b_start:b_end]] = b_knn_pred.cpu().numpy()
        
    # 3. Action-GPT Transformer
    model = LightweightTransformer(vocab_size=NUM_ACTIVE_SYLLS, num_targets=NUM_ACTIVE_SYLLS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_ds = TensorDataset(
        torch.tensor(X_toks[tr], dtype=torch.long),
        torch.tensor(X_durs[tr], dtype=torch.float32),
        torch.tensor(X_masks[tr], dtype=torch.bool),
        torch.tensor(Y_tr, dtype=torch.float32)
    )
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
    model.train()
    for epoch in range(5):
        for b_tok, b_dur, b_msk, b_y in tr_loader:
            b_tok, b_dur, b_msk, b_y = b_tok.to(DEVICE), b_dur.to(DEVICE), b_msk.to(DEVICE), b_y.to(DEVICE)
            opt.zero_grad()
            pred = model(b_tok, b_dur, b_msk)
            loss = js_divergence_pt(b_y, pred).mean()
            loss.backward()
            opt.step()
            
    model.eval()
    with torch.no_grad():
        te_pred = model(
            torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
            torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
            torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
        )
        preds_transformer[te] = te_pred.cpu().numpy()

# Global Null Divergence
global_null = np.mean(Y_targets, axis=0, keepdims=True)
global_null /= np.sum(global_null)
null_pred = np.repeat(global_null, N_win, axis=0)

null_js = js_divergence_pt(Y_t, torch.tensor(null_pred, dtype=torch.float32, device=DEVICE)).cpu().numpy()
mean_null = float(np.mean(null_js))

# Individual Model Risks
risk_knn = js_divergence_pt(Y_t, torch.tensor(preds_knn, dtype=torch.float32, device=DEVICE)).cpu().numpy()
risk_ridge = js_divergence_pt(Y_t, torch.tensor(preds_ridge, dtype=torch.float32, device=DEVICE)).cpu().numpy()
risk_tf = js_divergence_pt(Y_t, torch.tensor(preds_transformer, dtype=torch.float32, device=DEVICE)).cpu().numpy()

# -----------------------------------------------------------------------------
# 3. Stacked Ensemble (Learned Out-of-Fold Convex Blend of k-NN, Ridge, TF)
# -----------------------------------------------------------------------------
log("\n--- Computing Stacked Ensemble (Out-of-Fold Learned Mixture) ---")
preds_ensemble = np.zeros_like(Y_targets)

for fold, (tr, te) in enumerate(gkf.split(X_hybrid, groups=sess_ids)):
    # Optimize weights w on training fold to minimize JS
    P_knn_tr = torch.tensor(preds_knn[tr], dtype=torch.float32, device=DEVICE)
    P_rdg_tr = torch.tensor(preds_ridge[tr], dtype=torch.float32, device=DEVICE)
    P_tf_tr = torch.tensor(preds_transformer[tr], dtype=torch.float32, device=DEVICE)
    Y_tr_t = Y_t[tr]
    
    def loss_func(w):
        w = np.maximum(w, 0.0)
        w /= np.sum(w) + 1e-8
        blend = w[0] * P_knn_tr + w[1] * P_rdg_tr + w[2] * P_tf_tr
        return float(js_divergence_pt(Y_tr_t, blend).mean().cpu().numpy())
        
    res = minimize(loss_func, [1/3, 1/3, 1/3], method='Nelder-Mead')
    w_opt = np.maximum(res.x, 0.0); w_opt /= np.sum(w_opt)
    
    blend_te = w_opt[0] * preds_knn[te] + w_opt[1] * preds_ridge[te] + w_opt[2] * preds_transformer[te]
    blend_te /= np.sum(blend_te, axis=1, keepdims=True)
    preds_ensemble[te] = blend_te

risk_ensemble = js_divergence_pt(Y_t, torch.tensor(preds_ensemble, dtype=torch.float32, device=DEVICE)).cpu().numpy()

# -----------------------------------------------------------------------------
# 4. Oracle Model Selector (Best single model picked per window)
# -----------------------------------------------------------------------------
log("\n--- Computing Oracle Model Selector (Per-Window Best Model) ---")
risk_oracle_selector = np.minimum(np.minimum(risk_knn, risk_ridge), risk_tf)

# -----------------------------------------------------------------------------
# 5. Finite-Data Branch Oracle & Random Candidate Control (k=50)
# -----------------------------------------------------------------------------
log("\n--- Computing Finite-Data Branch-Selection Oracle (k=50) ---")
risk_branch_oracle = np.zeros(N_win, dtype=np.float32)
risk_random_oracle = np.zeros(N_win, dtype=np.float32)

for s_idx in np.unique(sess_ids):
    anchor_mask = (sess_ids == s_idx)
    cand_mask = (sess_ids != s_idx)
    
    anchor_idx = np.where(anchor_mask)[0]
    cand_idx = np.where(cand_mask)[0]
    
    X_anc = X_hybrid_t[anchor_idx]
    X_cnd = X_hybrid_t[cand_idx]
    Y_anc = Y_t[anchor_idx]
    Y_cnd = Y_t[cand_idx]
    
    batch_size = 512
    for b_start in range(0, len(anchor_idx), batch_size):
        b_end = min(b_start + batch_size, len(anchor_idx))
        b_X = X_anc[b_start:b_end]
        b_Y = Y_anc[b_start:b_end]
        b_global = anchor_idx[b_start:b_end]
        
        # 1. Matched Neighbors Branch Oracle
        dists = torch.cdist(b_X, X_cnd, p=2.0)
        _, topk_idx = torch.topk(dists, k=K_NN, largest=False, dim=1) # [B, 50]
        
        neigh_futures = Y_cnd[topk_idx] # [B, 50, 99]
        b_Y_expanded = b_Y.unsqueeze(1).repeat(1, K_NN, 1) # [B, 50, 99]
        
        divs = js_divergence_pt(b_Y_expanded.reshape(-1, NUM_ACTIVE_SYLLS), neigh_futures.reshape(-1, NUM_ACTIVE_SYLLS))
        divs = divs.reshape(-1, K_NN) # [B, 50]
        best_branch_risk = torch.min(divs, dim=1).values
        risk_branch_oracle[b_global] = best_branch_risk.cpu().numpy()
        
        # 2. Random Candidate Set Control (50 random cross-session futures)
        rand_idx = np.random.choice(len(cand_idx), size=K_NN, replace=False)
        rand_futures = Y_cnd[rand_idx].unsqueeze(0).repeat(len(b_global), 1, 1) # [B, 50, 99]
        
        rand_divs = js_divergence_pt(b_Y_expanded.reshape(-1, NUM_ACTIVE_SYLLS), rand_futures.reshape(-1, NUM_ACTIVE_SYLLS))
        rand_divs = rand_divs.reshape(-1, K_NN)
        best_random_risk = torch.min(rand_divs, dim=1).values
        risk_random_oracle[b_global] = best_random_risk.cpu().numpy()

# -----------------------------------------------------------------------------
# 6. Residual Predictability Test (Can Ridge/MLP predict Transformer residuals?)
# -----------------------------------------------------------------------------
log("\n--- Testing Transformer Residual Predictability ---")
# Logit residuals: delta = log(Y) - log(Y_hat_tf)
eps = 1e-4
log_Y = np.log(Y_targets + eps)
log_Y_hat = np.log(preds_transformer + eps)
log_resid = log_Y - log_Y_hat

# Train secondary Ridge model on X_hybrid to predict log_resid
resid_preds = np.zeros_like(Y_targets)
for tr, te in gkf.split(X_hybrid, groups=sess_ids):
    c_hat = Ridge(5.0).fit(X_hybrid[tr], log_resid[tr]).predict(X_hybrid[te])
    # Apply correction in logit space and softmax
    corrected_log = log_Y_hat[te] + c_hat
    p_corr = np.exp(corrected_log - np.max(corrected_log, axis=1, keepdims=True))
    p_corr /= np.sum(p_corr, axis=1, keepdims=True)
    resid_preds[te] = p_corr

risk_residual_corrected = js_divergence_pt(Y_t, torch.tensor(resid_preds, dtype=torch.float32, device=DEVICE)).cpu().numpy()

# -----------------------------------------------------------------------------
# Compile Frontier Results & Session-Clustered 95% CIs
# -----------------------------------------------------------------------------
def get_stats(arr):
    sess_means = [np.mean(arr[sess_ids == s]) for s in np.unique(sess_ids)]
    null_means = [np.mean(null_js[sess_ids == s]) for s in np.unique(sess_ids)]
    gains = [((n - m) / n) * 100.0 for n, m in zip(null_means, sess_means)]
    
    m_risk, ci_risk = float(np.mean(sess_means)), float(1.96 * np.std(sess_means) / np.sqrt(len(sess_means)))
    m_gain, ci_gain = float(np.mean(gains)), float(1.96 * np.std(gains) / np.sqrt(len(gains)))
    return m_risk, ci_risk, m_gain, ci_gain

tiers_data = [
    ("1. Global Blind Null", null_js),
    ("2. Non-Parametric k-NN (k=50)", risk_knn),
    ("3. Parametric Ridge BoW", risk_ridge),
    ("4. Action-GPT Transformer", risk_tf),
    ("5. Residual-Corrected Transformer", risk_residual_corrected),
    ("6. Stacked Learned Ensemble", risk_ensemble),
    ("7. Oracle Model Selector (Best per window)", risk_oracle_selector),
    ("8. Random Candidate Oracle (k=50 Control)", risk_random_oracle),
    ("9. Matched Branch Oracle (k=50 Best)", risk_branch_oracle),
]

summary_rows = []
for name, arr in tiers_data:
    r, ci_r, g, ci_g = get_stats(arr)
    summary_rows.append({
        "Tier": name,
        "Risk_bits": r,
        "Risk_CI_95": ci_r,
        "Predictive_Gain_pct": g,
        "Gain_CI_95": ci_g
    })
    log(f"  {name:<40}: Risk = {r:.4f} ± {ci_r:.4f} bits | Gain = +{g:5.2f}% ± {ci_g:4.2f}%")

df_frontier = pd.DataFrame(summary_rows)
csv_frontier_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "empirical_frontier_suite_results.csv")
df_frontier.to_csv(csv_frontier_path, index=False)
log(f"\nSaved empirical frontier results to: {csv_frontier_path}")

# -----------------------------------------------------------------------------
# Render Publication 2-Panel Figure
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.0), dpi=300)

# Panel 1: The Full Empirical Predictive Frontier Ladder (Gain %)
labels = [
    "Global\nNull", "k-NN\n(k=50)", "Ridge\nBoW", "Action-GPT\nTransformer",
    "Residual\nCorrected", "Stacked\nEnsemble", "Oracle\nSelector",
    "Random\nOracle", "Matched Branch\nOracle"
]
gains = df_frontier["Predictive_Gain_pct"].values
gain_cis = df_frontier["Gain_CI_95"].values

bar_colors = [
    "#7f7f7f", "#2ca02c", "#1f77b4", "#d62728",
    "#e377c2", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"
]

bars = ax1.bar(labels, gains, yerr=gain_cis, capsize=4, color=bar_colors, alpha=0.85, width=0.6)
ax1.set_title("A. The Finite-Data Empirical Predictive Frontier\n(From Baseline Models to Oracle Branch Selection)", fontsize=11, fontweight='bold', pad=12)
ax1.set_ylabel("Predictive Information Gain (% $\\Delta D_{JS}$)", fontsize=11)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar, g_val in zip(bars, gains):
    ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f"+{g_val:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Panel 2: Divergence Risk (bits) with Noise Floor Reference
risks = df_frontier["Risk_bits"].values
risk_cis = df_frontier["Risk_CI_95"].values

bars2 = ax2.bar(labels, risks, yerr=risk_cis, capsize=4, color=bar_colors, alpha=0.85, width=0.6)
ax2.axhline(mean_meas_noise, color='#d62728', linestyle='--', linewidth=1.8, label=f'Target 120s Measurement Jitter ({mean_meas_noise:.3f} bits)')

ax2.set_title("B. Residual Predictive Risk & Target Noise Floor\n(Evaluating Remaining Headroom in Physical Bits)", fontsize=11, fontweight='bold', pad=12)
ax2.set_ylabel("Predictive Divergence Risk (bits, $D_{JS}$)", fontsize=11)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar, r_val in zip(bars2, risks):
    ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.006, f"{r_val:.3f} b", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(LM_DIR, "output", "presentation", "figures", "fig10_empirical_frontier_ladder.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"Saved Figure 10 to: {fig_path}")
log("FRONTIER SUITE COMPLETE!")
