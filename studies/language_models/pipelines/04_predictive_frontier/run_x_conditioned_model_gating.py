#!/usr/bin/env python3
"""
Pipeline: X-Conditioned Model Gating & Oracle Predictability Diagnostic
=======================================================================
Investigates whether the advantage of the target-peeking oracle model selector
is itself predictable from measured behavioral history X, or only visible after
observing the realized future Y.

Diagnostics & Experiments:
1. Oracle-Winner Distribution & Winning Margin Analysis (Δ_i = L_second - L_best)
2. Hard-Gating Classifiers (Multinomial Logistic, GBDT, MLP) vs Controls (Majority, Random)
3. Soft-Gating Mixture-of-Experts (MoE) with Temperature Sweeps
4. Margin-Aware Analysis (All windows, Δ > q50, Δ > q75)
5. Soft-Regret Metric Evaluation (Regret = L_gated - L_oracle)
6. Behavioral Regime Attribution & Feature Importance Analysis
7. Render 4-Panel Publication Figure (fig12_x_conditioned_model_gating.png)
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)
OUT_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier")
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
MDS_DIR = os.path.join(LM_DIR, "mds", "04_predictive_frontier")

for d in [OUT_DIR, TAB_DIR, FIG_DIR, MDS_DIR]:
    os.makedirs(d, exist_ok=True)

FPS = 80.025
W_FRAMES = int(68.3 * FPS)
H_FRAMES = int(120.0 * FPS)
STRIDE_FRAMES = int(5.0 * FPS)
NUM_ACTIVE_SYLLS = 99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def js_divergence_np(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=-1, keepdims=True); Q = Q / np.sum(Q, axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    kl_p = np.sum(P * np.log2(P / M), axis=-1)
    kl_q = np.sum(Q * np.log2(Q / M), axis=-1)
    return 0.5 * (kl_p + kl_q)

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0); Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True); Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log2(P / M), dim=-1) + torch.sum(Q * torch.log2(Q / M), dim=-1))

class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size=99, embed_dim=64, nhead=4, num_layers=2, max_len=140, num_targets=99, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.syll_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.dur_proj = nn.Linear(1, embed_dim)
        if use_pos:
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
        if self.use_pos:
            x = x + self.pos_embed[:, :L, :]
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

class SoftGatingMoE(nn.Module):
    def __init__(self, input_dim, num_models=4, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_models)
        )
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits / self.temperature, dim=-1)

# -------------------------------------------------------------------------
# Step 1: Load Data & Extract Windows
# -------------------------------------------------------------------------
log("Loading continuous bout cache & session data...")
cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
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
        "spatial": getattr(s, "spatial_features", None),
        "pose": getattr(s, "pose_features", None)
    })

sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)
    clean_tokens, clean_onsets, clean_durs = [], [], []

    sp_raw = session_meta[sess_idx]["spatial"]
    po_raw = session_meta[sess_idx]["pose"]
    sp = sp_raw[:N] if sp_raw is not None and len(sp_raw) >= N else np.zeros((N, 6), dtype=np.float32)
    po = po_raw[:N] if po_raw is not None and len(po_raw) >= N else np.zeros((N, 30), dtype=np.float32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = b["syllable"] - 1
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0
            clean_tokens.append(s_id)
            clean_onsets.append(sf)
            clean_durs.append((ef - sf + 1) / FPS)

    b_starts = np.array([b["start_frame"] for b in bouts])
    for t_idx, b_st in enumerate(b_starts):
        prev_st = b_starts[t_idx-1] if t_idx > 0 else 0
        next_bout_ptrs[prev_st:b_st] = b_st
    if len(b_starts) > 0:
        next_bout_ptrs[b_starts[-1]:] = N

    cum_bout = np.vstack([np.zeros((1, NUM_ACTIVE_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    cum_sp = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
    cum_po = np.vstack([np.zeros((1, 30), dtype=np.float32), np.cumsum(po, axis=0)])

    sessions_data.append({
        "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
        "cum_sp": cum_sp, "cum_po": cum_po,
        "next_bout_ptrs": next_bout_ptrs, "inj_frame": session_meta[sess_idx]["inj_frame"],
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

    past_s = t_anchors - W_FRAMES
    p_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][past_s]
    p_sp = (s["cum_sp"][t_anchors] - s["cum_sp"][past_s]) / W_FRAMES
    p_po = (s["cum_po"][t_anchors] - s["cum_po"][past_s]) / W_FRAMES

    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    seq_toks = np.full((M, max_l), NUM_ACTIVE_SYLLS, dtype=np.int64)
    seq_durs = np.zeros((M, max_l), dtype=np.float32)
    seq_masks = np.ones((M, max_l), dtype=bool)
    seq_lens = np.zeros(M, dtype=np.int64)

    b_onsets, b_tokens, b_durs = s["onsets"], s["tokens"], s["durs"]
    entropy_arr = np.zeros(M, dtype=np.float32)
    unique_arr = np.zeros(M, dtype=np.float32)
    trans_rate_arr = np.zeros(M, dtype=np.float32)
    mean_dwell_arr = np.zeros(M, dtype=np.float32)
    last_syll_arr = np.full(M, NUM_ACTIVE_SYLLS, dtype=np.int64)

    for i, t_anc in enumerate(t_anchors):
        t_min = t_anc - W_FRAMES
        m_b = (b_onsets >= t_min) & (b_onsets < t_anc)
        idx_b = np.where(m_b)[0]
        if len(idx_b) > 0:
            sub_toks = b_tokens[idx_b]
            sub_durs = b_durs[idx_b]
            L_i = min(len(idx_b), max_l)
            seq_toks[i, :L_i] = sub_toks[-L_i:]
            seq_durs[i, :L_i] = sub_durs[-L_i:]
            seq_masks[i, :L_i] = False
            seq_lens[i] = L_i

            # Entropy & statistics
            p_dist = p_cnt[i] / max(np.sum(p_cnt[i]), 1e-6)
            pos_p = p_dist[p_dist > 0]
            entropy_arr[i] = -np.sum(pos_p * np.log2(pos_p))
            unique_arr[i] = np.sum(p_cnt[i] > 0)
            trans_rate_arr[i] = len(idx_b) / (W_FRAMES / FPS)
            mean_dwell_arr[i] = np.mean(sub_durs)
            last_syll_arr[i] = sub_toks[-1]

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "target": target[i],
            "bow": p_cnt[i],
            "spatial": p_sp[i],
            "pose": p_po[i],
            "seq_tok": seq_toks[i],
            "seq_dur": seq_durs[i],
            "seq_mask": seq_masks[i],
            "seq_len": seq_lens[i],
            "entropy": entropy_arr[i],
            "unique": unique_arr[i],
            "trans_rate": trans_rate_arr[i],
            "mean_dwell": mean_dwell_arr[i],
            "last_syll": last_syll_arr[i],
            "session_time": t_anchors[i] / N
        })

N_WIN = len(raw_windows)
log(f"Extracted {N_WIN:,} clean windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
X_bow = np.array([w["bow"] for w in raw_windows], dtype=np.float32)
X_sp = np.array([w["spatial"] for w in raw_windows], dtype=np.float32)
X_po = np.array([w["pose"] for w in raw_windows], dtype=np.float32)
X_toks = np.array([w["seq_tok"] for w in raw_windows], dtype=np.int64)
X_durs = np.array([w["seq_dur"] for w in raw_windows], dtype=np.float32)
X_masks = np.array([w["seq_mask"] for w in raw_windows], dtype=bool)

# Balanced distance space for k-NN
X_bow_norm = X_bow / (np.linalg.norm(X_bow, axis=1, keepdims=True) + 1e-6)
X_sp_norm = (X_sp - np.mean(X_sp, axis=0)) / (np.std(X_sp, axis=0) + 1e-6)
X_match_space = np.hstack([X_bow_norm, X_sp_norm * 0.5])
X_linear_full = np.hstack([X_bow_norm, X_sp, X_po])

# Rich interpretable gating feature matrix
X_entropy = np.array([w["entropy"] for w in raw_windows], dtype=np.float32).reshape(-1, 1)
X_unique = np.array([w["unique"] for w in raw_windows], dtype=np.float32).reshape(-1, 1)
X_trans = np.array([w["trans_rate"] for w in raw_windows], dtype=np.float32).reshape(-1, 1)
X_dwell = np.array([w["mean_dwell"] for w in raw_windows], dtype=np.float32).reshape(-1, 1)
X_time = np.array([w["session_time"] for w in raw_windows], dtype=np.float32).reshape(-1, 1)
X_last_syll = np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32)
for i, w in enumerate(raw_windows):
    if 0 <= w["last_syll"] < NUM_ACTIVE_SYLLS:
        X_last_syll[i, w["last_syll"]] = 1.0

# Compute Global Null Baseline
global_null = np.mean(Y_targets, axis=0, keepdims=True)
global_null /= np.sum(global_null)
null_losses = js_divergence_np(Y_targets, np.repeat(global_null, N_WIN, axis=0))
mean_null_risk = float(np.mean(null_losses))
log(f"Global Blind Null Risk: {mean_null_risk:.5f} bits")

# -------------------------------------------------------------------------
# Step 2: 5-Fold Cross-Validation for Base Models
# -------------------------------------------------------------------------
log("=" * 75)
log("TRAINING 4 VALIDATED BASE MODELS ACROSS 5 HELD-OUT FOLDS")
log("=" * 75)

gkf = GroupKFold(n_splits=5)
MODEL_NAMES = [
    "1. Non-Parametric k-NN (k=50)",
    "2. Parametric Ridge BoW",
    "3. Shuffled Transformer",
    "4. Ordered Transformer"
]
NUM_MODELS = len(MODEL_NAMES)

out_preds = np.zeros((N_WIN, NUM_MODELS, NUM_ACTIVE_SYLLS), dtype=np.float32)
knn_densities = np.zeros(N_WIN, dtype=np.float32)
cache_preds_path = os.path.join(OUT_DIR, "gating_base_predictions_cache.pkl")

if os.path.exists(cache_preds_path):
    log(f"Loading cached base model predictions from: {cache_preds_path}")
    with open(cache_preds_path, "rb") as f:
        cached_data = pickle.load(f)
        out_preds = cached_data["out_preds"]
        knn_densities = cached_data["knn_densities"]
else:
    for fold, (tr, te) in enumerate(gkf.split(X_match_space, groups=sess_ids)):
        log(f"  [Fold {fold+1}/5] Train N={len(tr):,}, Test N={len(te):,}...")
        Y_tr = Y_targets[tr]
        te_len = len(te)

        # 1. k-NN (k=50)
        X_tr_m = X_match_space[tr]
        X_te_m = X_match_space[te]

        # Compute in batches
        k_nn = 50
        p_knn = np.zeros((te_len, NUM_ACTIVE_SYLLS), dtype=np.float32)
        b_size = 500
        for b_i in range(0, te_len, b_size):
            b_idx = slice(b_i, min(b_i + b_size, te_len))
            dists = np.linalg.norm(X_te_m[b_idx, None, :] - X_tr_m[None, :, :], axis=2)
            top_k = np.argsort(dists, axis=1)[:, :k_nn]
            for row_i in range(top_k.shape[0]):
                p_knn[b_i + row_i] = np.mean(Y_tr[top_k[row_i]], axis=0)
                knn_densities[te[b_i + row_i]] = np.mean(dists[row_i, top_k[row_i]])

        p_knn = np.clip(p_knn, 1e-4, 1.0)
        p_knn /= np.sum(p_knn, axis=1, keepdims=True)
        out_preds[te, 0] = p_knn

        # 2. Ridge BoW
        p_ridge = np.clip(Ridge(alpha=10.0).fit(X_linear_full[tr], Y_tr).predict(X_linear_full[te]), 1e-4, 1.0)
        p_ridge /= np.sum(p_ridge, axis=1, keepdims=True)
        out_preds[te, 1] = p_ridge

        # PyTorch Tensors for Transformers
        tr_ds = TensorDataset(
            torch.tensor(X_toks[tr], dtype=torch.long),
            torch.tensor(X_durs[tr], dtype=torch.float32),
            torch.tensor(X_masks[tr], dtype=torch.bool),
            torch.tensor(Y_tr, dtype=torch.float32)
        )
        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)

        # 3. Shuffled Transformer (use_pos=False)
        tf_shuff = SequenceTransformer(vocab_size=NUM_ACTIVE_SYLLS, num_targets=NUM_ACTIVE_SYLLS, use_pos=False).to(DEVICE)
        opt_shuff = torch.optim.AdamW(tf_shuff.parameters(), lr=1e-3, weight_decay=1e-4)
        tf_shuff.train()
        for ep in range(5):
            for b_toks, b_durs, b_masks, b_y in tr_loader:
                b_toks, b_durs, b_masks, b_y = b_toks.to(DEVICE), b_durs.to(DEVICE), b_masks.to(DEVICE), b_y.to(DEVICE)
                opt_shuff.zero_grad()
                pred = tf_shuff(b_toks, b_durs, b_masks)
                loss = js_divergence_pt(b_y, pred).mean()
                loss.backward()
                opt_shuff.step()
        tf_shuff.eval()
        with torch.no_grad():
            p_shuff = tf_shuff(
                torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
                torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
                torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
            ).cpu().numpy()
        out_preds[te, 2] = p_shuff

        # 4. Ordered Transformer (use_pos=True)
        tf_ord = SequenceTransformer(vocab_size=NUM_ACTIVE_SYLLS, num_targets=NUM_ACTIVE_SYLLS, use_pos=True).to(DEVICE)
        opt_ord = torch.optim.AdamW(tf_ord.parameters(), lr=1e-3, weight_decay=1e-4)
        tf_ord.train()
        for ep in range(5):
            for b_toks, b_durs, b_masks, b_y in tr_loader:
                b_toks, b_durs, b_masks, b_y = b_toks.to(DEVICE), b_durs.to(DEVICE), b_masks.to(DEVICE), b_y.to(DEVICE)
                opt_ord.zero_grad()
                pred = tf_ord(b_toks, b_durs, b_masks)
                loss = js_divergence_pt(b_y, pred).mean()
                loss.backward()
                opt_ord.step()
        tf_ord.eval()
        with torch.no_grad():
            p_ord = tf_ord(
                torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
                torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
                torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
            ).cpu().numpy()
        out_preds[te, 3] = p_ord

    with open(cache_preds_path, "wb") as f:
        pickle.dump({"out_preds": out_preds, "knn_densities": knn_densities}, f)
    log(f"Cached base model predictions to: {cache_preds_path}")

# Assemble Loss Matrix L_im = D_JS(Y_i, \hat{Y}_i^{(m)})
loss_matrix = np.zeros((N_WIN, NUM_MODELS), dtype=np.float32)
for m_idx in range(NUM_MODELS):
    loss_matrix[:, m_idx] = js_divergence_np(Y_targets, out_preds[:, m_idx])

# -------------------------------------------------------------------------
# Step 3: Oracle Winner Distribution & Winning Margins (Diagnostic 1)
# -------------------------------------------------------------------------
log("=" * 75)
log("DIAGNOSTIC 1: ORACLE-WINNER DISTRIBUTION & MARGIN DYNAMICS")
log("=" * 75)

oracle_winners = np.argmin(loss_matrix, axis=1) # m_i^*
sorted_losses = np.sort(loss_matrix, axis=1)
winning_margins = sorted_losses[:, 1] - sorted_losses[:, 0] # \Delta_i = L_second - L_best

winner_counts = pd.Series(oracle_winners).value_counts().sort_index()
winner_fracs = winner_counts / N_WIN

oracle_winner_df = []
for m_idx, m_name in enumerate(MODEL_NAMES):
    mask_m = (oracle_winners == m_idx)
    frac_m = float(winner_fracs.get(m_idx, 0.0))
    mean_margin_m = float(np.mean(winning_margins[mask_m])) if np.sum(mask_m) > 0 else 0.0
    median_margin_m = float(np.median(winning_margins[mask_m])) if np.sum(mask_m) > 0 else 0.0
    base_risk = float(np.mean(loss_matrix[:, m_idx]))
    base_gain = ((mean_null_risk - base_risk) / mean_null_risk) * 100.0

    oracle_winner_df.append({
        "Model_ID": m_idx,
        "Model_Name": m_name,
        "Standalone_Risk_bits": base_risk,
        "Standalone_Gain_pct": base_gain,
        "Windows_Won_N": int(winner_counts.get(m_idx, 0)),
        "Win_Fraction_pct": frac_m * 100.0,
        "Mean_Win_Margin_bits": mean_margin_m,
        "Median_Win_Margin_bits": median_margin_m
    })
    log(f"  {m_name:30s} | Standalone: {base_risk:.4f} b (+{base_gain:5.1f}%) | Wins: {frac_m*100:5.2f}% ({winner_counts.get(m_idx,0):,}) | Margin: {mean_margin_m:.4f} b")

oracle_winner_df = pd.DataFrame(oracle_winner_df)
oracle_winner_df.to_csv(os.path.join(TAB_DIR, "oracle_winner_distribution.csv"), index=False)

q50_margin = float(np.percentile(winning_margins, 50))
q75_margin = float(np.percentile(winning_margins, 75))
q90_margin = float(np.percentile(winning_margins, 90))
log(f"Margin Quantiles: Median (q50) = {q50_margin:.4f} b, 75th (q75) = {q75_margin:.4f} b, 90th (q90) = {q90_margin:.4f} b")

# -------------------------------------------------------------------------
# Step 4: Gating Model Implementations (Strict Outer GroupKFold)
# -------------------------------------------------------------------------
log("=" * 75)
log("EXPERIMENT: X-CONDITIONED HARD & SOFT MODEL GATING")
log("=" * 75)

# Construct comprehensive X-feature matrix for gating
X_density = knn_densities.reshape(-1, 1)
X_gate_raw = np.hstack([
    X_bow_norm,        # 99-D
    X_sp,              # 6-D
    X_po[:, :10],      # 10-D continuous posture
    X_entropy,         # 1-D
    X_unique,          # 1-D
    X_trans,           # 1-D
    X_dwell,           # 1-D
    X_time,            # 1-D
    X_density          # 1-D
])

# Compute Target-Peeking Oracle Selector Performance
oracle_preds = np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32)
for i in range(N_WIN):
    oracle_preds[i] = out_preds[i, oracle_winners[i]]
oracle_risk_overall = float(np.mean(js_divergence_np(Y_targets, oracle_preds)))
oracle_gain_overall = ((mean_null_risk - oracle_risk_overall) / mean_null_risk) * 100.0
log(f"Target-Peeking Oracle Selector: {oracle_risk_overall:.5f} bits (+{oracle_gain_overall:.2f}%)")

# Gating storage
gated_preds = {
    "Majority Baseline": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Random Control": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Global Stacked Ensemble": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Logistic Regression Gate": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Gradient Boosted Gate (GBDT)": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "MLP Neural Gate": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Soft MoE Gate (T=0.1)": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
    "Soft MoE Gate (T=0.5)": np.zeros((N_WIN, NUM_ACTIVE_SYLLS), dtype=np.float32),
}

hard_selections = {
    "Majority Baseline": np.zeros(N_WIN, dtype=np.int64),
    "Random Control": np.zeros(N_WIN, dtype=np.int64),
    "Logistic Regression Gate": np.zeros(N_WIN, dtype=np.int64),
    "Gradient Boosted Gate (GBDT)": np.zeros(N_WIN, dtype=np.int64),
    "MLP Neural Gate": np.zeros(N_WIN, dtype=np.int64),
}

for fold, (tr, te) in enumerate(gkf.split(X_gate_raw, groups=sess_ids)):
    log(f"  [Gating Fold {fold+1}/5] Fitting gates on N={len(tr):,} windows...")

    # Scale features strictly on training fold
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_gate_raw[tr])
    Z_te = scaler.transform(X_gate_raw[te])

    y_tr_oracle = oracle_winners[tr]
    L_tr = loss_matrix[tr]

    # 1. Global Stacked Ensemble (Fixed Weights fitted on training fold)
    # Fit non-negative linear regression on training out-of-fold logits
    ens_ridge = Ridge(alpha=100.0, positive=True).fit(out_preds[tr].reshape(len(tr), -1), Y_targets[tr])
    p_ens = np.clip(ens_ridge.predict(out_preds[te].reshape(len(te), -1)), 1e-4, 1.0)
    p_ens /= np.sum(p_ens, axis=1, keepdims=True)
    gated_preds["Global Stacked Ensemble"][te] = p_ens

    # 2. Majority Class Baseline
    maj_winner = int(pd.Series(y_tr_oracle).mode()[0])
    hard_selections["Majority Baseline"][te] = maj_winner
    gated_preds["Majority Baseline"][te] = out_preds[te, maj_winner]

    # 3. Random-Label Control
    perm_labels = np.random.permutation(y_tr_oracle)
    clf_rand = LogisticRegression(C=1.0, max_iter=200, random_state=42).fit(Z_tr, perm_labels)
    pred_rand_m = clf_rand.predict(Z_te)
    hard_selections["Random Control"][te] = pred_rand_m
    for idx_local, global_i in enumerate(te):
        gated_preds["Random Control"][global_i] = out_preds[global_i, pred_rand_m[idx_local]]

    # 4. Logistic Regression Gate
    clf_log = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced", random_state=42).fit(Z_tr, y_tr_oracle)
    pred_log_m = clf_log.predict(Z_te)
    hard_selections["Logistic Regression Gate"][te] = pred_log_m
    for idx_local, global_i in enumerate(te):
        gated_preds["Logistic Regression Gate"][global_i] = out_preds[global_i, pred_log_m[idx_local]]

    # 5. Gradient Boosted Decision Trees
    clf_gbdt = HistGradientBoostingClassifier(max_iter=50, max_depth=4, class_weight="balanced", random_state=42).fit(Z_tr, y_tr_oracle)
    pred_gbdt_m = clf_gbdt.predict(Z_te)
    hard_selections["Gradient Boosted Gate (GBDT)"][te] = pred_gbdt_m
    for idx_local, global_i in enumerate(te):
        gated_preds["Gradient Boosted Gate (GBDT)"][global_i] = out_preds[global_i, pred_gbdt_m[idx_local]]

    # 6. MLP Neural Gate
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, alpha=0.01, random_state=42).fit(Z_tr, y_tr_oracle)
    pred_mlp_m = clf_mlp.predict(Z_te)
    hard_selections["MLP Neural Gate"][te] = pred_mlp_m
    for idx_local, global_i in enumerate(te):
        gated_preds["MLP Neural Gate"][global_i] = out_preds[global_i, pred_mlp_m[idx_local]]

    # 7. Soft Mixture-of-Experts (MoE Gate) with Temperature T=0.1 & T=0.5
    for T_val, name_t in [(0.1, "Soft MoE Gate (T=0.1)"), (0.5, "Soft MoE Gate (T=0.5)")]:
        # Soft preference targets: p_im \propto \exp(-L_im / T)
        soft_targets = np.exp(-L_tr / T_val)
        soft_targets /= np.sum(soft_targets, axis=1, keepdims=True)

        # Train PyTorch gating network
        moe_net = SoftGatingMoE(input_dim=Z_tr.shape[1], num_models=NUM_MODELS, hidden_dim=64, temperature=T_val).to(DEVICE)
        opt_moe = torch.optim.AdamW(moe_net.parameters(), lr=3e-3, weight_decay=1e-4)

        ds_moe = TensorDataset(torch.tensor(Z_tr, dtype=torch.float32), torch.tensor(soft_targets, dtype=torch.float32))
        loader_moe = DataLoader(ds_moe, batch_size=256, shuffle=True)

        moe_net.train()
        for ep in range(8):
            for b_z, b_p in loader_moe:
                b_z, b_p = b_z.to(DEVICE), b_p.to(DEVICE)
                opt_moe.zero_grad()
                w_pred = moe_net(b_z)
                # KL loss between predicted weights and soft preferences
                loss_moe = torch.sum(b_p * torch.log((b_p + 1e-8) / (w_pred + 1e-8)), dim=-1).mean()
                loss_moe.backward(); opt_moe.step()

        moe_net.eval()
        with torch.no_grad():
            w_test = moe_net(torch.tensor(Z_te, dtype=torch.float32, device=DEVICE)).cpu().numpy()

        # Combine model predictions using soft weights
        for idx_local, global_i in enumerate(te):
            p_combo = np.zeros(NUM_ACTIVE_SYLLS, dtype=np.float32)
            for m_i in range(NUM_MODELS):
                p_combo += w_test[idx_local, m_i] * out_preds[global_i, m_i]
            p_combo = np.clip(p_combo, 1e-4, 1.0)
            p_combo /= np.sum(p_combo)
            gated_preds[name_t][global_i] = p_combo

# -------------------------------------------------------------------------
# Step 5: Comprehensive Performance & Regret Audit
# -------------------------------------------------------------------------
log("=" * 75)
log("PERFORMANCE EVALUATION & REGRET SUMMARY")
log("=" * 75)

# Session-level confidence intervals helper
def calc_session_ci(losses, sessions):
    sess_u = np.unique(sessions)
    s_means = [np.mean(losses[sessions == s]) for s in sess_u]
    m = float(np.mean(s_means))
    ci = float(1.96 * np.std(s_means) / np.sqrt(len(sess_u)))
    return m, ci

global_ens_losses = js_divergence_np(Y_targets, gated_preds["Global Stacked Ensemble"])
ens_risk_mean, ens_risk_ci = calc_session_ci(global_ens_losses, sess_ids)
oracle_losses = js_divergence_np(Y_targets, oracle_preds)
orc_risk_mean, orc_risk_ci = calc_session_ci(oracle_losses, sess_ids)

results_summary = []
all_model_eval_names = (
    MODEL_NAMES +
    ["5. Global Stacked Ensemble", "6. Majority Baseline", "7. Random Control"] +
    ["8. Logistic Regression Gate", "9. Gradient Boosted Gate (GBDT)", "10. MLP Neural Gate"] +
    ["11. Soft MoE Gate (T=0.1)", "12. Soft MoE Gate (T=0.5)"] +
    ["13. Target-Peeking Oracle Selector"]
)

for m_name in all_model_eval_names:
    if m_name in MODEL_NAMES:
        m_idx = MODEL_NAMES.index(m_name)
        pred_mat = out_preds[:, m_idx]
        method_type = "Deployable Base Model"
    elif m_name == "13. Target-Peeking Oracle Selector":
        pred_mat = oracle_preds
        method_type = "Target-Peeking Oracle"
    else:
        key = m_name.split(". ")[1]
        pred_mat = gated_preds[key]
        method_type = "Deployable Gated Ensemble" if "Gate" in m_name or "Ensemble" in m_name else "Baseline Control"

    losses = js_divergence_np(Y_targets, pred_mat)
    r_mean, r_ci = calc_session_ci(losses, sess_ids)
    gain_mean = ((mean_null_risk - r_mean) / mean_null_risk) * 100.0
    gain_ci = (r_ci / mean_null_risk) * 100.0

    # Regret metric: Regret_i = L_{i, model} - L_{i, oracle}
    regrets = losses - oracle_losses
    reg_mean, reg_ci = calc_session_ci(regrets, sess_ids)

    # Recovery ratio of oracle advantage: (R_ens - R_model) / (R_ens - R_oracle)
    denom = (ens_risk_mean - orc_risk_mean)
    rec_ratio = ((ens_risk_mean - r_mean) / denom * 100.0) if denom > 0 else 0.0

    # Classification metrics if hard gating
    key_short = m_name.split(". ")[1] if ". " in m_name else m_name
    if key_short in hard_selections:
        acc = float(accuracy_score(oracle_winners, hard_selections[key_short]) * 100.0)
        bal_acc = float(balanced_accuracy_score(oracle_winners, hard_selections[key_short]) * 100.0)
    else:
        acc, bal_acc = np.nan, np.nan

    results_summary.append({
        "Model_Name": m_name,
        "Methodology_Type": method_type,
        "Risk_bits": r_mean,
        "Risk_CI_95": r_ci,
        "Predictive_Gain_pct": gain_mean,
        "Gain_CI_95": gain_ci,
        "Mean_Regret_bits": reg_mean,
        "Regret_CI_95": reg_ci,
        "Oracle_Recovery_Ratio_pct": rec_ratio,
        "Classification_Acc_pct": acc,
        "Balanced_Acc_pct": bal_acc
    })

    log(f"  {m_name:36s} | Risk: {r_mean:.4f} ± {r_ci:.4f} b (+{gain_mean:5.1f}%) | Regret: {reg_mean:.4f} b | Recov: {rec_ratio:5.1f}% | Acc: {acc:.1f}%")

gating_results_df = pd.DataFrame(results_summary)
gating_results_df.to_csv(os.path.join(TAB_DIR, "gating_model_results.csv"), index=False)

# -------------------------------------------------------------------------
# Step 6: Margin-Stratified Regret Analysis (Control D & E)
# -------------------------------------------------------------------------
log("=" * 75)
log("MARGIN-STRATIFIED REGRET AUDIT (All Windows vs. High Margin Subsets)")
log("=" * 75)

margin_subsets = [
    ("All Windows (N=29,325)", np.ones(N_WIN, dtype=bool)),
    (f"Moderate Margin Δ > q50 ({q50_margin:.3f} b)", winning_margins > q50_margin),
    (f"High Margin Δ > q75 ({q75_margin:.3f} b)", winning_margins > q75_margin),
    (f"Decisive Margin Δ > q90 ({q90_margin:.3f} b)", winning_margins > q90_margin)
]

margin_results = []
for subset_name, mask_sub in margin_subsets:
    N_sub = int(np.sum(mask_sub))
    sub_sess = sess_ids[mask_sub]
    sub_orc_l = oracle_losses[mask_sub]

    for m_name in ["6. Majority Baseline", "7. Random Control", "5. Global Stacked Ensemble", "8. Logistic Regression Gate", "9. Gradient Boosted Gate (GBDT)", "11. Soft MoE Gate (T=0.1)"]:
        key = m_name.split(". ")[1]
        p_mat = gated_preds[key][mask_sub]
        l_sub = js_divergence_np(Y_targets[mask_sub], p_mat)
        reg_sub = l_sub - sub_orc_l
        r_m, r_ci = calc_session_ci(l_sub, sub_sess)
        reg_m, reg_ci = calc_session_ci(reg_sub, sub_sess)

        margin_results.append({
            "Margin_Stratum": subset_name,
            "Subset_Size_N": N_sub,
            "Model_Name": m_name,
            "Risk_bits": r_m,
            "Risk_CI": r_ci,
            "Regret_bits": reg_m,
            "Regret_CI": reg_ci
        })
        log(f"  [{subset_name[:25]:25s}] {m_name:30s} | Regret: {reg_m:.4f} ± {reg_ci:.4f} b | Risk: {r_m:.4f} b")

margin_df = pd.DataFrame(margin_results)
margin_df.to_csv(os.path.join(TAB_DIR, "gating_regret_results.csv"), index=False)

# -------------------------------------------------------------------------
# Step 7: Behavioral Feature Importance & Regime Attribution
# -------------------------------------------------------------------------
log("=" * 75)
log("BEHAVIORAL REGIME & FEATURE ATTRIBUTION")
log("=" * 75)

# Fit Logistic Regression on full scaled dataset for feature inspection
scaler_full = StandardScaler()
Z_full = scaler_full.fit_transform(X_gate_raw)
clf_inspect = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced", random_state=42).fit(Z_full, oracle_winners)

feature_names = [f"BoW_Syll_{i}" for i in range(NUM_ACTIVE_SYLLS)] + [
    "Spatial_X", "Spatial_Y", "Spatial_Speed", "Spatial_AngVel", "Spatial_Rad", "Spatial_BoundDist"
] + [f"Pose_PC_{i}" for i in range(10)] + [
    "Repertoire_Entropy", "Unique_Syllable_Count", "Transition_Rate_Hz", "Mean_Bout_Dwell_s",
    "Session_Elapsed_Time", "KNN_Local_Distance"
]

feat_importance_df = []
for m_idx, m_name in enumerate(MODEL_NAMES):
    coefs = clf_inspect.coef_[m_idx]
    top_pos = np.argsort(coefs)[-5:][::-1]
    top_neg = np.argsort(coefs)[:5]

    for f_idx in range(len(feature_names)):
        feat_importance_df.append({
            "Model_ID": m_idx,
            "Model_Name": m_name,
            "Feature_Name": feature_names[f_idx],
            "Logistic_Coefficient": float(coefs[f_idx])
        })
    log(f"  {m_name:30s} Top Positive Features: {[feature_names[p] for p in top_pos]}")

feat_importance_df = pd.DataFrame(feat_importance_df)
feat_importance_df.to_csv(os.path.join(TAB_DIR, "gating_feature_importance.csv"), index=False)

# -------------------------------------------------------------------------
# Step 8: Render 4-Panel Publication Figure (fig12_x_conditioned_model_gating.png)
# -------------------------------------------------------------------------
log("=" * 75)
log("RENDERING 4-PANEL PUBLICATION FIGURE (Fig 12)")
log("=" * 75)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Colors
PALETTE = ["#4A90E2", "#50E3C2", "#F5A623", "#E74C3C", "#9B59B6", "#34495E"]

# Panel A: Predictive Hierarchy Ladder (Single -> Global Ensemble -> Gated -> Oracle)
ax = axes[0, 0]
plot_models = [
    ("Ridge BoW", gating_results_df.loc[gating_results_df["Model_Name"] == "2. Parametric Ridge BoW", "Predictive_Gain_pct"].values[0],
                  gating_results_df.loc[gating_results_df["Model_Name"] == "2. Parametric Ridge BoW", "Gain_CI_95"].values[0], "#7F8C8D"),
    ("k-NN (k=50)", gating_results_df.loc[gating_results_df["Model_Name"] == "1. Non-Parametric k-NN (k=50)", "Predictive_Gain_pct"].values[0],
                   gating_results_df.loc[gating_results_df["Model_Name"] == "1. Non-Parametric k-NN (k=50)", "Gain_CI_95"].values[0], "#95A5A6"),
    ("Ordered TF", gating_results_df.loc[gating_results_df["Model_Name"] == "4. Ordered Transformer", "Predictive_Gain_pct"].values[0],
                  gating_results_df.loc[gating_results_df["Model_Name"] == "4. Ordered Transformer", "Gain_CI_95"].values[0], "#3498DB"),
    ("Shuffled TF", gating_results_df.loc[gating_results_df["Model_Name"] == "3. Shuffled Transformer", "Predictive_Gain_pct"].values[0],
                   gating_results_df.loc[gating_results_df["Model_Name"] == "3. Shuffled Transformer", "Gain_CI_95"].values[0], "#2980B9"),
    ("Global Ensemble", gating_results_df.loc[gating_results_df["Model_Name"] == "5. Global Stacked Ensemble", "Predictive_Gain_pct"].values[0],
                       gating_results_df.loc[gating_results_df["Model_Name"] == "5. Global Stacked Ensemble", "Gain_CI_95"].values[0], "#2ECC71"),
    ("X-Gate (GBDT)", gating_results_df.loc[gating_results_df["Model_Name"] == "9. Gradient Boosted Gate (GBDT)", "Predictive_Gain_pct"].values[0],
                     gating_results_df.loc[gating_results_df["Model_Name"] == "9. Gradient Boosted Gate (GBDT)", "Gain_CI_95"].values[0], "#F39C12"),
    ("X-Gate (Soft MoE)", gating_results_df.loc[gating_results_df["Model_Name"] == "11. Soft MoE Gate (T=0.1)", "Predictive_Gain_pct"].values[0],
                         gating_results_df.loc[gating_results_df["Model_Name"] == "11. Soft MoE Gate (T=0.1)", "Gain_CI_95"].values[0], "#E67E22"),
    ("Oracle Selector*", gating_results_df.loc[gating_results_df["Model_Name"] == "13. Target-Peeking Oracle Selector", "Predictive_Gain_pct"].values[0],
                        gating_results_df.loc[gating_results_df["Model_Name"] == "13. Target-Peeking Oracle Selector", "Gain_CI_95"].values[0], "#E74C3C")
]

x_pos = np.arange(len(plot_models))
names = [p[0] for p in plot_models]
gains = [p[1] for p in plot_models]
cis = [p[2] for p in plot_models]
cols = [p[3] for p in plot_models]

bars = ax.bar(x_pos, gains, yerr=cis, capsize=5, color=cols, edgecolor="black", alpha=0.85, width=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10, fontweight="bold")
ax.set_ylabel(r"Predictive Gain (% $\Delta D_{JS}$ over Null)", fontsize=11, fontweight="bold")
ax.set_title("A. Predictive Spectrum: Deployable Gating vs. Oracle Selector", fontsize=12, fontweight="bold", pad=10)
ax.axhline(gains[4], color="#2ECC71", linestyle="--", alpha=0.7, label=f"Global Ensemble ({gains[4]:.1f}%)")
ax.axhline(gains[-1], color="#E74C3C", linestyle="--", alpha=0.7, label=f"Oracle Selector* ({gains[-1]:.1f}%)")
ax.legend(loc="upper left", frameon=True, fontsize=10)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")
ax.set_ylim(0, 50)

# Panel B: Oracle Winner Breakdown & Margin Distribution
ax = axes[0, 1]
ax2 = ax.twinx()

# Winner percentages bar
m_short_names = ["k-NN", "Ridge", "Shuffled TF", "Ordered TF"]
win_pcts = [oracle_winner_df.loc[oracle_winner_df["Model_ID"] == i, "Win_Fraction_pct"].values[0] for i in range(4)]
bars_w = ax.bar(np.arange(4) - 0.15, win_pcts, width=0.3, color=["#95A5A6", "#7F8C8D", "#2980B9", "#3498DB"], edgecolor="black", alpha=0.85, label="Windows Won (%)")
ax.set_ylabel("Fraction of Windows Won (%)", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_xticks(range(4))
ax.set_xticklabels(m_short_names, fontsize=10, fontweight="bold")

# Winning margin boxplots / mean margins
margin_means = [oracle_winner_df.loc[oracle_winner_df["Model_ID"] == i, "Mean_Win_Margin_bits"].values[0] for i in range(4)]
bars_m = ax2.bar(np.arange(4) + 0.15, margin_means, width=0.3, color="#F1C40F", edgecolor="black", alpha=0.85, label="Mean Winning Margin (bits)")
ax2.set_ylabel(r"Mean Winning Margin $\Delta_i$ (bits)", fontsize=11, fontweight="bold", color="#B7950B")
ax2.grid(False)
ax.set_title("B. Oracle-Winner Distribution & Winning Margins", fontsize=12, fontweight="bold", pad=10)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=10)

# Panel C: Soft-Regret Curves across Margin Strata
ax = axes[1, 0]
strata_keys = ["All Windows", "Moderate Margin", "High Margin", "Decisive Margin"]
strata_labels = ["All Windows", "Δ > q50", "Δ > q75", "Δ > q90"]
comp_models = [
    ("5. Global Stacked Ensemble", "Global Ensemble", "#2ECC71", "-o"),
    ("6. Majority Baseline", "Majority Baseline", "#7F8C8D", "--s"),
    ("8. Logistic Regression Gate", "Logistic Gate", "#3498DB", "-^"),
    ("9. Gradient Boosted Gate (GBDT)", "GBDT Gate", "#F39C12", "-D"),
    ("11. Soft MoE Gate (T=0.1)", "Soft MoE Gate", "#E67E22", "-v"),
]

for m_key, m_label, col, mk in comp_models:
    reg_vals = [margin_df.loc[(margin_df["Model_Name"] == m_key) & (margin_df["Margin_Stratum"].str.startswith(k)), "Regret_bits"].values[0] for k in strata_keys]
    ax.plot(range(4), reg_vals, mk, color=col, linewidth=2, markersize=8, label=m_label)

ax.set_xticks(range(4))
ax.set_xticklabels(strata_labels, fontsize=10, fontweight="bold")
ax.set_ylabel(r"Mean Regret ($L_{\hat{m}} - L_{m^*}$ bits)", fontsize=11, fontweight="bold")
ax.set_title("C. Margin-Stratified Regret ($L_{gated} - L_{oracle}$)", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.set_ylim(0, 0.10)

# Panel D: Behavioral Feature Importance for Model Selection
ax = axes[1, 1]
key_feats = [
    "Repertoire_Entropy", "Unique_Syllable_Count", "Transition_Rate_Hz",
    "Mean_Bout_Dwell_s", "Spatial_Rad", "Spatial_Speed", "KNN_Local_Distance", "Session_Elapsed_Time"
]

feat_matrix = np.zeros((len(key_feats), 4), dtype=np.float32)
for f_i, f_name in enumerate(key_feats):
    for m_i in range(4):
        v = feat_importance_df.loc[(feat_importance_df["Model_ID"] == m_i) & (feat_importance_df["Feature_Name"] == f_name), "Logistic_Coefficient"]
        feat_matrix[f_i, m_i] = v.values[0] if len(v) > 0 else 0.0

sns.heatmap(
    feat_matrix.T, annot=True, fmt=".2f", cmap="vlag", center=0.0,
    xticklabels=key_feats, yticklabels=m_short_names, ax=ax, cbar_kws={'label': 'Logistic Gating Coefficient'}
)
ax.set_xticklabels(key_feats, rotation=40, ha="right", fontsize=9, fontweight="bold")
ax.set_yticklabels(m_short_names, fontsize=10, fontweight="bold")
ax.set_title("D. Behavioral Feature Attribution for Model Preference", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig12_x_conditioned_model_gating.png")
plt.savefig(fig_path)
plt.close()
log(f"Saved publication figure: {fig_path}")

log("=" * 75)
log("X-CONDITIONED MODEL GATING EXPERIMENT COMPLETED SUCCESSFULLY!")
log("=" * 75)
