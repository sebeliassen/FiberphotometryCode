#!/usr/bin/env python3
"""
Reconciliation Audit: Shuffled vs. Ordered Transformer
======================================================
Investigates the exact mechanism behind the ~0.9 pp difference between:
1. Architecture A: Order-Agnostic Bag Transformer (use_pos=False)
2. Architecture B: Ordered Action Transformer (use_pos=True)
3. Test-Time Shuffle Ablation: Shuffling tokens at inference on Architecture B.
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

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
    session_meta.append({"sess_idx": i, "inj_frame": m.get("injection_frame", 0)})

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

sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)
    clean_tokens, clean_onsets, clean_durs = [], [], []
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
    sessions_data.append({
        "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
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

    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    seq_toks = np.full((M, max_l), NUM_ACTIVE_SYLLS, dtype=np.int64)
    seq_durs = np.zeros((M, max_l), dtype=np.float32)
    seq_masks = np.ones((M, max_l), dtype=bool)
    seq_lens = np.zeros(M, dtype=np.int64)

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
            seq_lens[i] = L_i

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "target": target[i],
            "seq_tok": seq_toks[i],
            "seq_dur": seq_durs[i],
            "seq_mask": seq_masks[i],
            "seq_len": seq_lens[i]
        })

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
X_toks = np.array([w["seq_tok"] for w in raw_windows], dtype=np.int64)
X_durs = np.array([w["seq_dur"] for w in raw_windows], dtype=np.float32)
X_masks = np.array([w["seq_mask"] for w in raw_windows], dtype=bool)
X_lens = np.array([w["seq_len"] for w in raw_windows], dtype=np.int64)

Y_t = torch.tensor(Y_targets, dtype=torch.float32, device=DEVICE)
global_null = np.mean(Y_targets, axis=0, keepdims=True)
global_null /= np.sum(global_null)
null_pred = np.repeat(global_null, len(raw_windows), axis=0)
null_js = js_divergence_pt(Y_t, torch.tensor(null_pred, dtype=torch.float32, device=DEVICE)).cpu().numpy()

gkf = GroupKFold(n_splits=5)
gains_shuff_arch = []
gains_ord_arch = []
gains_ord_test_shuffled = []

for fold, (tr, te) in enumerate(gkf.split(X_toks, groups=sess_ids)):
    Y_tr = Y_targets[tr]
    te_t = Y_t[te]
    null_fold = float(np.mean(null_js[te]))

    tr_ds = TensorDataset(
        torch.tensor(X_toks[tr], dtype=torch.long),
        torch.tensor(X_durs[tr], dtype=torch.float32),
        torch.tensor(X_masks[tr], dtype=torch.bool),
        torch.tensor(Y_tr, dtype=torch.float32)
    )
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)

    # 1. Shuffled Architecture (use_pos=False)
    m_shuff = SequenceTransformer(vocab_size=NUM_ACTIVE_SYLLS, num_targets=NUM_ACTIVE_SYLLS, use_pos=False).to(DEVICE)
    opt1 = torch.optim.AdamW(m_shuff.parameters(), lr=1e-3, weight_decay=1e-4)
    m_shuff.train()
    for ep in range(5):
        for b_tok, b_dur, b_msk, b_y in tr_loader:
            b_tok, b_dur, b_msk, b_y = b_tok.to(DEVICE), b_dur.to(DEVICE), b_msk.to(DEVICE), b_y.to(DEVICE)
            opt1.zero_grad()
            l = js_divergence_pt(b_y, m_shuff(b_tok, b_dur, b_msk)).mean()
            l.backward(); opt1.step()
    m_shuff.eval()
    with torch.no_grad():
        p1 = m_shuff(torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE), torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE), torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE))
        r1 = float(js_divergence_pt(te_t, p1).mean().cpu().numpy())
        g1 = ((null_fold - r1) / null_fold) * 100.0
        gains_shuff_arch.append(g1)

    # 2. Ordered Architecture (use_pos=True)
    m_ord = SequenceTransformer(vocab_size=NUM_ACTIVE_SYLLS, num_targets=NUM_ACTIVE_SYLLS, use_pos=True).to(DEVICE)
    opt2 = torch.optim.AdamW(m_ord.parameters(), lr=1e-3, weight_decay=1e-4)
    m_ord.train()
    for ep in range(5):
        for b_tok, b_dur, b_msk, b_y in tr_loader:
            b_tok, b_dur, b_msk, b_y = b_tok.to(DEVICE), b_dur.to(DEVICE), b_msk.to(DEVICE), b_y.to(DEVICE)
            opt2.zero_grad()
            l = js_divergence_pt(b_y, m_ord(b_tok, b_dur, b_msk)).mean()
            l.backward(); opt2.step()
    m_ord.eval()
    with torch.no_grad():
        # Ordered evaluation
        p2 = m_ord(torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE), torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE), torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE))
        r2 = float(js_divergence_pt(te_t, p2).mean().cpu().numpy())
        g2 = ((null_fold - r2) / null_fold) * 100.0
        gains_ord_arch.append(g2)

        # Test-Time Shuffle on Ordered Model
        tok_te_shuf = torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE)
        dur_te_shuf = torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE)
        for i_s in range(len(tok_te_shuf)):
            L_s = int(X_lens[te[i_s]])
            if L_s > 1:
                perm = torch.randperm(L_s, device=DEVICE)
                tok_te_shuf[i_s, :L_s] = tok_te_shuf[i_s, perm]
                dur_te_shuf[i_s, :L_s] = dur_te_shuf[i_s, perm]
        p3 = m_ord(tok_te_shuf, dur_te_shuf, torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE))
        r3 = float(js_divergence_pt(te_t, p3).mean().cpu().numpy())
        g3 = ((null_fold - r3) / null_fold) * 100.0
        gains_ord_test_shuffled.append(g3)

    print(f"Fold {fold+1}: Shuffled Arch = +{g1:.2f}% | Ordered Arch = +{g2:.2f}% | Ordered Model w/ Test Shuffled = +{g3:.2f}%")

print("\n" + "=" * 70)
print("SUMMARY RECONCILIATION:")
print(f"  1. Order-Agnostic Transformer (Trained without Pos Embeds): +{np.mean(gains_shuff_arch):.2f}% ± {1.96*np.std(gains_shuff_arch)/np.sqrt(5):.2f}%")
print(f"  2. Ordered Transformer (Trained with Pos Embeds):          +{np.mean(gains_ord_arch):.2f}% ± {1.96*np.std(gains_ord_arch)/np.sqrt(5):.2f}%")
print(f"  3. Ordered Transformer with TEST-TIME Token Scrambling:     +{np.mean(gains_ord_test_shuffled):.2f}% ± {1.96*np.std(gains_ord_test_shuffled)/np.sqrt(5):.2f}%")
print("=" * 70)
