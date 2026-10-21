#!/usr/bin/env python3
"""
Complete P_n Information Ladder Audit: All Tiers from P0 to P6b
===============================================================
Evaluates side-by-side:
- Historical Rasterized Path (with Syllable 0 gap artifact)
- Clean Direct Bout Sequence (99 active syllables, zero gap artifact)

Tiers evaluated:
  P0:   Global Blind Null
  P1:   Time Drift
  P2:   Spatial Context
  P2.5: Continuous Pose
  P3:   Current Syllable
  P3.5: Hybrid Instantaneous (Pose + Syllable + Space)
  P4:   Repertoire Bout Counts Only
  P4.5: Repertoire Counts + Dwell Time
  P5:   Full Linear Bag-of-Words Ceiling
  P6a:  Shuffled Transformer (Order-Agnostic Attention)
  P6b:  Ordered Transformer (Action-GPT / Sequential Syntax)
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

def run_ladder(clean_mode=True):
    num_sylls = 99 if clean_mode else 100
    sessions = []

    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        N = bouts[-1]["end_frame"] + 1
        fs = np.full(N, -1, dtype=np.int32)
        sp = np.zeros((N, 6), dtype=np.float32)
        pose = np.zeros((N, 30), dtype=np.float32)

        oh_bout = np.zeros((N, num_sylls), dtype=np.float32)
        oh_frame = np.zeros((N, num_sylls), dtype=np.float32)
        next_bout_ptrs = np.full(N, N, dtype=np.int32)

        clean_tokens = []
        clean_onsets = []
        clean_durs = []

        if clean_mode:
            for b in bouts:
                sf, ef = b["start_frame"], b["end_frame"]
                s_id = b["syllable"] - 1 # 1..99 -> 0..98
                if 0 <= s_id < num_sylls:
                    fs[sf:ef+1] = s_id
                    oh_bout[sf, s_id] = 1.0
                    oh_frame[sf:ef+1, s_id] = 1.0
                    clean_tokens.append(s_id)
                    clean_onsets.append(sf)
                    clean_durs.append((ef - sf + 1) / FPS)

            b_starts = np.array([b["start_frame"] for b in bouts])
            for t_idx, b_st in enumerate(b_starts):
                prev_st = b_starts[t_idx-1] if t_idx > 0 else 0
                next_bout_ptrs[prev_st:b_st] = b_st
            if len(b_starts) > 0:
                next_bout_ptrs[b_starts[-1]:] = N
        else:
            fs = np.zeros(N, dtype=np.int32)
            for b in bouts:
                sf, ef = b["start_frame"], b["end_frame"]
                s_id = min(b["syllable"], num_sylls - 1)
                fs[sf:ef+1] = s_id
                oh_frame[sf:ef+1, s_id] = 1.0

            diffs = np.where(fs[1:] != fs[:-1])[0] + 1
            onsets = np.zeros(N, dtype=bool); onsets[0] = True; onsets[diffs] = True
            oh_bout[onsets, fs[onsets]] = 1.0

            next_bout_ptrs = np.zeros(N, dtype=np.int32); cur = 0
            for d in diffs: next_bout_ptrs[cur:d] = d; cur = d
            next_bout_ptrs[cur:] = N

            clean_tokens = fs[onsets]
            clean_onsets = np.where(onsets)[0]
            clean_durs = np.zeros(len(clean_onsets), dtype=np.float32)
            for i in range(len(clean_onsets) - 1):
                clean_durs[i] = (clean_onsets[i+1] - clean_onsets[i]) / FPS
            clean_durs[-1] = (N - clean_onsets[-1]) / FPS

        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                sp[sf:ef+1, :4] = vec[:4]; sp[sf:ef+1, 4] = vec[10]; sp[sf:ef+1, 5] = vec[12]
                if len(vec) >= 30: pose[sf:ef+1] = vec[:30]

        cum_bout = np.vstack([np.zeros((1, num_sylls), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
        cum_dwell = np.vstack([np.zeros((1, num_sylls), dtype=np.float32), np.cumsum(oh_frame, axis=0)]) / FPS
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp**2, axis=0)])

        sessions.append({
            "sess_idx": sess_idx, "N": N, "fs": fs, "pose": pose,
            "cum_bout": cum_bout, "cum_dwell": cum_dwell,
            "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
            "cum_spatial_sq": cum_spatial_sq, "inj_frame": session_meta[sess_idx]["inj_frame"],
            "tokens": np.array(clean_tokens, dtype=np.int64),
            "onsets": np.array(clean_onsets, dtype=np.int64),
            "durs": np.array(clean_durs, dtype=np.float32)
        })

    all_p1, all_p2, all_p25, all_p3, all_p35, all_p4, all_p45, all_p5, all_tgt, all_sess = [], [], [], [], [], [], [], [], [], []
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

        # P1: Time
        t_norm = t_anchors / float(N)
        p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)

        # P2: Spatial Context
        sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
        sp_mean = sp_sum / float(W_FRAMES)
        sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
        sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
        p2 = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

        # P2.5: Continuous Pose
        p25 = np.hstack([p2, s["pose"][t_anchors]]).astype(np.float32)

        # P3: Current Syllable
        curr_syll = s["fs"][t_anchors]
        curr_oh = np.zeros((M, num_sylls), dtype=np.float32)
        valid_syll = (curr_syll >= 0) & (curr_syll < num_sylls)
        curr_oh[np.arange(M)[valid_syll], curr_syll[valid_syll]] = 1.0
        p3 = np.hstack([p2, curr_oh]).astype(np.float32)

        # P3.5: Hybrid Instantaneous (Pose + Syllable + Space)
        p35 = np.hstack([p25, curr_oh]).astype(np.float32)

        # P4: Repertoire Counts
        b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        p_rep_cnt = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * num_sylls)
        p4 = np.hstack([p3, p_rep_cnt]).astype(np.float32)

        # P4.5: Repertoire Counts + Dwell Time
        d_cnt = s["cum_dwell"][t_anchors] - s["cum_dwell"][t_anchors - W_FRAMES]
        p_rep_dwell = (d_cnt + 1e-4) / (np.sum(d_cnt, axis=1, keepdims=True) + 1e-4 * num_sylls)
        p45 = np.hstack([p4, p_rep_dwell]).astype(np.float32)

        # P5: Full BoW Ceiling
        p5 = np.hstack([p35, p_rep_cnt, p_rep_dwell]).astype(np.float32)

        # Target
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * num_sylls)

        # Tokens
        seq_toks = np.full((M, max_l), num_sylls, dtype=np.int64)
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

        all_p1.append(p1); all_p2.append(p2); all_p25.append(p25)
        all_p3.append(p3); all_p35.append(p35); all_p4.append(p4)
        all_p45.append(p45); all_p5.append(p5); all_tgt.append(target)
        all_sess.append(np.full(M, s["sess_idx"], dtype=int))
        all_seq_toks.append(seq_toks); all_seq_durs.append(seq_durs); all_seq_masks.append(seq_masks)

    X_p1, X_p2, X_p25, X_p3, X_p35, X_p4, X_p45, X_p5 = [np.vstack(x) for x in [all_p1, all_p2, all_p25, all_p3, all_p35, all_p4, all_p45, all_p5]]
    Y_tgt = np.vstack(all_tgt)
    groups = np.concatenate(all_sess)
    X_toks = np.vstack(all_seq_toks)
    X_durs = np.vstack(all_seq_durs)
    X_masks = np.vstack(all_seq_masks)

    print(f"Loaded {len(Y_tgt):,} windows. Running 5-fold cross-validation...")
    gkf = GroupKFold(n_splits=5)

    results = {k: [] for k in ["Null", "P1", "P2", "P2.5", "P3", "P3.5", "P4", "P4.5", "P5", "P6a_Shuffled", "P6b_Ordered"]}

    for fold, (tr, te) in enumerate(gkf.split(X_p1, groups=groups)):
        Y_tr, Y_te = Y_tgt[tr], Y_tgt[te]
        p_glob = np.mean(Y_tr, axis=0, keepdims=True)
        p_glob /= np.sum(p_glob)

        Y_te_t = torch.tensor(Y_te, dtype=torch.float32, device=DEVICE)
        p_glob_t = torch.tensor(p_glob, dtype=torch.float32, device=DEVICE).repeat(len(te), 1)
        jb_t = js_divergence_pt(Y_te_t, p_glob_t)
        mean_jb = float(jb_t.mean().cpu().numpy())
        results["Null"].append(mean_jb)

        # Linear models
        for name, feat, alpha in [
            ("P1", X_p1, 5.0), ("P2", X_p2, 5.0), ("P2.5", X_p25, 5.0),
            ("P3", X_p3, 5.0), ("P3.5", X_p35, 5.0), ("P4", X_p4, 1.0),
            ("P4.5", X_p45, 1.0), ("P5", X_p5, 1.0)
        ]:
            pred = np.clip(Ridge(alpha).fit(feat[tr], Y_tr).predict(feat[te]), 1e-4, 1.0)
            pred /= np.sum(pred, axis=1, keepdims=True)
            pred_t = torch.tensor(pred, dtype=torch.float32, device=DEVICE)
            js_m = js_divergence_pt(Y_te_t, pred_t)
            gain = float(((mean_jb - js_m.mean().cpu().numpy()) / mean_jb) * 100.0)
            results[name].append(gain)

        # P6a: Shuffled Transformer (no positional embeddings)
        tf_shuff = LightweightTransformer(vocab_size=num_sylls, num_targets=num_sylls, use_pos=False).to(DEVICE)
        opt_shuff = torch.optim.AdamW(tf_shuff.parameters(), lr=1e-3, weight_decay=1e-4)

        tr_ds = TensorDataset(
            torch.tensor(X_toks[tr], dtype=torch.long),
            torch.tensor(X_durs[tr], dtype=torch.float32),
            torch.tensor(X_masks[tr], dtype=torch.bool),
            torch.tensor(Y_tr, dtype=torch.float32)
        )
        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)

        tf_shuff.train()
        for epoch in range(5):
            for b_tok, b_dur, b_msk, b_y in tr_loader:
                b_tok, b_dur, b_msk, b_y = b_tok.to(DEVICE), b_dur.to(DEVICE), b_msk.to(DEVICE), b_y.to(DEVICE)
                opt_shuff.zero_grad()
                pred = tf_shuff(b_tok, b_dur, b_msk)
                loss = js_divergence_pt(b_y, pred).mean()
                loss.backward()
                opt_shuff.step()

        tf_shuff.eval()
        with torch.no_grad():
            pred_shuff = tf_shuff(
                torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
                torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
                torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
            )
            js_shuff = js_divergence_pt(Y_te_t, pred_shuff)
            results["P6a_Shuffled"].append(float(((mean_jb - js_shuff.mean().cpu().numpy()) / mean_jb) * 100.0))

        # P6b: Ordered Transformer (with positional embeddings)
        tf_ord = LightweightTransformer(vocab_size=num_sylls, num_targets=num_sylls, use_pos=True).to(DEVICE)
        opt_ord = torch.optim.AdamW(tf_ord.parameters(), lr=1e-3, weight_decay=1e-4)

        tf_ord.train()
        for epoch in range(5):
            for b_tok, b_dur, b_msk, b_y in tr_loader:
                b_tok, b_dur, b_msk, b_y = b_tok.to(DEVICE), b_dur.to(DEVICE), b_msk.to(DEVICE), b_y.to(DEVICE)
                opt_ord.zero_grad()
                pred = tf_ord(b_tok, b_dur, b_msk)
                loss = js_divergence_pt(b_y, pred).mean()
                loss.backward()
                opt_ord.step()

        tf_ord.eval()
        with torch.no_grad():
            pred_ord = tf_ord(
                torch.tensor(X_toks[te], dtype=torch.long, device=DEVICE),
                torch.tensor(X_durs[te], dtype=torch.float32, device=DEVICE),
                torch.tensor(X_masks[te], dtype=torch.bool, device=DEVICE)
            )
            js_ord = js_divergence_pt(Y_te_t, pred_ord)
            results["P6b_Ordered"].append(float(((mean_jb - js_ord.mean().cpu().numpy()) / mean_jb) * 100.0))

        print(f"  Fold {fold+1}: Null={mean_jb:.4f} bits | P4={results['P4'][-1]:.1f}% | P4.5={results['P4.5'][-1]:.1f}% | P5={results['P5'][-1]:.1f}% | P6a={results['P6a_Shuffled'][-1]:.1f}% | P6b={results['P6b_Ordered'][-1]:.1f}%")

    ci = lambda arr: float(1.96 * np.std(arr) / np.sqrt(5))
    summary = {}
    for k, v in results.items():
        summary[k] = (np.mean(v), ci(v))
    return summary

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING COMPLETE P_n INFORMATION LADDER (P0 -> P6b) ON CLEAN 99-D BOUTS")
    print("=" * 80)
    clean_res = run_ladder(clean_mode=True)

    print("\n" + "=" * 80)
    print("COMPLETE CLEAN P_n INFORMATION LADDER SUMMARY:")
    print("=" * 80)
    print(f"{'Tier':<10} | {'Level Description':<36} | {'Predictive Gain (% ΔD_JS)':<25}")
    print("-" * 80)
    print(f"{'P0':<10} | {'Global Blind Null Divergence':<36} | {clean_res['Null'][0]:.4f} ± {clean_res['Null'][1]:.4f} bits (0.0%)")
    print(f"{'P1':<10} | {'Session Time Drift':<36} | +{clean_res['P1'][0]:5.2f}% ± {clean_res['P1'][1]:.2f}%")
    print(f"{'P2':<10} | {'Spatial Context (Arena Geometry)':<36} | +{clean_res['P2'][0]:5.2f}% ± {clean_res['P2'][1]:.2f}%")
    print(f"{'P2.5':<10} | {'Continuous Pose (30-D PCA)':<36} | +{clean_res['P2.5'][0]:5.2f}% ± {clean_res['P2.5'][1]:.2f}%")
    print(f"{'P3':<10} | {'Discrete Current Syllable':<36} | +{clean_res['P3'][0]:5.2f}% ± {clean_res['P3'][1]:.2f}%")
    print(f"{'P3.5':<10} | {'Hybrid Instantaneous (Pose+Syll)':<36} | +{clean_res['P3.5'][0]:5.2f}% ± {clean_res['P3.5'][1]:.2f}%")
    print(f"{'P4':<10} | {'Repertoire Counts Only (68.3s BoW)':<36} | +{clean_res['P4'][0]:5.2f}% ± {clean_res['P4'][1]:.2f}%")
    print(f"{'P4.5':<10} | {'Repertoire Counts + Dwell Time':<36} | +{clean_res['P4.5'][0]:5.2f}% ± {clean_res['P4.5'][1]:.2f}%")
    print(f"{'P5':<10} | {'Full Linear BoW Ceiling':<36} | +{clean_res['P5'][0]:5.2f}% ± {clean_res['P5'][1]:.2f}%")
    print(f"{'P6a':<10} | {'Shuffled Transformer (Order-Agnostic)':<36} | +{clean_res['P6a_Shuffled'][0]:5.2f}% ± {clean_res['P6a_Shuffled'][1]:.2f}%")
    print(f"{'P6b':<10} | {'Ordered Transformer (Action-GPT)':<36} | +{clean_res['P6b_Ordered'][0]:5.2f}% ± {clean_res['P6b_Ordered'][1]:.2f}%")
    print("=" * 80)
