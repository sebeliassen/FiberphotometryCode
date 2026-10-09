#!/usr/bin/env python3
"""
Audit & Re-run of Transformer and Ridge BoW on Clean Direct Bouts
=================================================================
Compares:
1. Historical rasterized path (with 1-frame Syllable 0 gap onsets).
2. Clean direct bout sequence (true 99 active syllables, zero Syllable 0 gaps).
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
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
    session_meta.append({
        "sess_idx": i,
        "inj_frame": m.get("injection_frame", 0),
    })

FPS = 80.025
W_FRAMES = int(68.3 * FPS)
H_FRAMES = int(120.0 * FPS)
STRIDE_FRAMES = int(5.0 * FPS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0)
    Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    kl1 = torch.sum(P * torch.log2(P / M), dim=-1)
    kl2 = torch.sum(Q * torch.log2(Q / M), dim=-1)
    return 0.5 * (kl1 + kl2)

class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=64, nhead=4, num_layers=2, max_len=140, num_targets=100):
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
        # Pool valid tokens
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

def run_pipeline(clean_mode=False):
    num_sylls = 99 if clean_mode else 100
    sessions = []

    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        N = bouts[-1]["end_frame"] + 1

        if clean_mode:
            # Clean direct bout onsets
            fs = np.full(N, -1, dtype=np.int32)
            oh_bout = np.zeros((N, num_sylls), dtype=np.float32)
            next_bout_ptrs = np.full(N, N, dtype=np.int32)

            clean_tokens = []
            clean_onsets = []
            clean_durs = []

            for b in bouts:
                sf, ef = b["start_frame"], b["end_frame"]
                s_id = b["syllable"] - 1 # 1..99 -> 0..98
                if 0 <= s_id < num_sylls:
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

            cum_bout = np.vstack([np.zeros((1, num_sylls), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

            sessions.append({
                "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
                "next_bout_ptrs": next_bout_ptrs, "inj_frame": session_meta[sess_idx]["inj_frame"],
                "tokens": np.array(clean_tokens, dtype=np.int64),
                "onsets": np.array(clean_onsets, dtype=np.int64),
                "durs": np.array(clean_durs, dtype=np.float32)
            })
        else:
            # Historical rasterized path
            fs = np.zeros(N, dtype=np.int32)
            for b in bouts:
                fs[b["start_frame"]:b["end_frame"]+1] = min(b["syllable"], num_sylls - 1)
            diffs = np.where(fs[1:] != fs[:-1])[0] + 1
            onsets = np.zeros(N, dtype=bool); onsets[0] = True; onsets[diffs] = True

            oh_bout = np.zeros((N, num_sylls), dtype=np.float32)
            oh_bout[onsets, fs[onsets]] = 1.0
            cum_bout = np.vstack([np.zeros((1, num_sylls), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

            next_bout_ptrs = np.zeros(N, dtype=np.int32); cur = 0
            for d in diffs: next_bout_ptrs[cur:d] = d; cur = d
            next_bout_ptrs[cur:] = N

            bout_seq = fs[onsets]
            bout_onsets = np.where(onsets)[0]
            bout_durs = np.zeros(len(bout_onsets), dtype=np.float32)
            for i in range(len(bout_onsets) - 1):
                bout_durs[i] = (bout_onsets[i+1] - bout_onsets[i]) / FPS
            bout_durs[-1] = (N - bout_onsets[-1]) / FPS

            sessions.append({
                "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
                "next_bout_ptrs": next_bout_ptrs, "inj_frame": session_meta[sess_idx]["inj_frame"],
                "tokens": bout_seq.astype(np.int64),
                "onsets": bout_onsets.astype(np.int64),
                "durs": bout_durs
            })

    # Window slicing
    all_bow, all_targets, all_sess = [], [], []
    all_seq_toks, all_seq_durs, all_seq_masks = [], [], []
    max_l = 140

    for s in sessions:
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

        # BoW
        b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        bow = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * num_sylls)

        # Target
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * num_sylls)

        # Sequential Tokens for Transformer
        seq_toks = np.full((M, max_l), num_sylls, dtype=np.int64) # padding index = num_sylls
        seq_durs = np.zeros((M, max_l), dtype=np.float32)
        seq_masks = np.ones((M, max_l), dtype=bool) # True = masked

        b_onsets = s["onsets"]
        b_tokens = s["tokens"]
        b_durs = s["durs"]

        for i, t_anc in enumerate(t_anchors):
            t_min = t_anc - W_FRAMES
            # Find bouts strictly within [t_min, t_anc]
            m_b = (b_onsets >= t_min) & (b_onsets < t_anc)
            idx_b = np.where(m_b)[0]
            if len(idx_b) > 0:
                idx_b = idx_b[-max_l:] # Take last max_l bouts
                L_i = len(idx_b)
                seq_toks[i, :L_i] = b_tokens[idx_b]
                seq_durs[i, :L_i] = b_durs[idx_b]
                seq_masks[i, :L_i] = False

        all_bow.append(bow); all_targets.append(target); all_sess.append(np.full(M, s["sess_idx"], dtype=int))
        all_seq_toks.append(seq_toks); all_seq_durs.append(seq_durs); all_seq_masks.append(seq_masks)

    X_bow = np.vstack(all_bow)
    Y_tgt = np.vstack(all_targets)
    groups = np.concatenate(all_sess)
    X_toks = np.vstack(all_seq_toks)
    X_durs = np.vstack(all_seq_durs)
    X_masks = np.vstack(all_seq_masks)

    print(f"\n--- Running Pipeline (Clean={clean_mode}, Dims={num_sylls}, Windows={len(Y_tgt):,}) ---")

    # 5-fold CV evaluation
    gkf = GroupKFold(n_splits=5)
    g_bow = []
    g_tf = []
    null_bits = []

    for fold, (tr, te) in enumerate(gkf.split(X_bow, groups=groups)):
        Y_tr, Y_te = Y_tgt[tr], Y_tgt[te]
        p_glob = np.mean(Y_tr, axis=0, keepdims=True)
        p_glob /= np.sum(p_glob)

        # Baseline Null
        Y_te_t = torch.tensor(Y_te, dtype=torch.float32, device=DEVICE)
        p_glob_t = torch.tensor(p_glob, dtype=torch.float32, device=DEVICE).repeat(len(te), 1)
        jb_t = js_divergence_pt(Y_te_t, p_glob_t)
        mean_jb = float(jb_t.mean().cpu().numpy())
        null_bits.append(mean_jb)

        # 1. Ridge BoW
        pred_bow = np.clip(Ridge(1.0).fit(X_bow[tr], Y_tr).predict(X_bow[te]), 1e-4, 1.0)
        pred_bow /= np.sum(pred_bow, axis=1, keepdims=True)
        pred_bow_t = torch.tensor(pred_bow, dtype=torch.float32, device=DEVICE)
        js_bow = js_divergence_pt(Y_te_t, pred_bow_t)
        gain_bow = float(((mean_jb - js_bow.mean().cpu().numpy()) / mean_jb) * 100.0)
        g_bow.append(gain_bow)

        # 2. Fast Transformer (Train 5 epochs per fold)
        model = LightweightTransformer(vocab_size=num_sylls, num_targets=num_sylls).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

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
                optimizer.zero_grad()
                pred = model(b_tok, b_dur, b_msk)
                loss = js_divergence_pt(b_y, pred).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            te_pred = model(
                torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
                torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
                torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
            )
            js_tf = js_divergence_pt(Y_te_t, te_pred)
            gain_tf = float(((mean_jb - js_tf.mean().cpu().numpy()) / mean_jb) * 100.0)
            g_tf.append(gain_tf)

        print(f"  Fold {fold+1}: Null = {mean_jb:.4f} bits | BoW Gain = +{gain_bow:.2f}% | Transformer Gain = +{gain_tf:.2f}%")

    ci = lambda arr: float(1.96 * np.std(arr) / np.sqrt(5))
    return {
        "null_bits": (np.mean(null_bits), ci(null_bits)),
        "bow_gain": (np.mean(g_bow), ci(g_bow)),
        "tf_gain": (np.mean(g_tf), ci(g_tf))
    }

if __name__ == "__main__":
    print("=" * 75)
    print("COMPARATIVE RE-RUN: HISTORICAL RASTERIZED VS. CLEAN DIRECT BOUTS")
    print("=" * 75)

    print("\n1. Running Historical Rasterized Path (with Syllable 0 gap artifact)...")
    res_old = run_pipeline(clean_mode=False)

    print("\n2. Running Clean Direct Bout Sequence (99 active syllables, zero Syllable 0)...")
    res_new = run_pipeline(clean_mode=True)

    print("\n" + "=" * 75)
    print("FINAL COMPARISON TABLE:")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Historical (Rasterized)':<22} | {'Clean (Direct Bouts)':<22}")
    print("-" * 75)
    print(f"{'Global Null Divergence':<25} | {res_old['null_bits'][0]:.4f} ± {res_old['null_bits'][1]:.4f} bits    | {res_new['null_bits'][0]:.4f} ± {res_new['null_bits'][1]:.4f} bits")
    print(f"{'Parametric Ridge BoW':<25} | +{res_old['bow_gain'][0]:5.2f}% ± {res_old['bow_gain'][1]:.2f}%       | +{res_new['bow_gain'][0]:5.2f}% ± {res_new['bow_gain'][1]:.2f}%")
    print(f"{'Sequential Transformer':<25} | +{res_old['tf_gain'][0]:5.2f}% ± {res_old['tf_gain'][1]:.2f}%       | +{res_new['tf_gain'][0]:5.2f}% ± {res_new['tf_gain'][1]:.2f}%")
    print("=" * 75)
