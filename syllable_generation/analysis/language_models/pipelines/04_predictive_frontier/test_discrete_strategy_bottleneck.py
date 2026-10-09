#!/usr/bin/env python3
"""
===============================================================================
Discrete Strategy Bottleneck Benchmark: Upgrading the 4-bit (K=16) Latent States
===============================================================================
Core Question:
  If we compress past behavioral sequences into 16 discrete states (4 bits),
  can we achieve higher predictive power by clustering the latent representations
  of the top 3 models (Duration-BoW, Mamba-1, and Action-GPT) instead of the
  traditional transition SVD?

Evaluated under 5-Fold Session-Grouped CV (W = 68.3s -> H = 120.0s):
  1. Standard Baseline: Transition-SVD -> K-Means (K=16) (+26.75% reference)
  2. Duration-BoW Latents: [Counts + Dwell Times] -> TruncatedSVD(24) -> K-Means (K=16)
  3. Mamba-1 SSM Latents: Mamba Penultimate Latent Space (64D) -> K-Means (K=16)
  4. Causal Action-GPT Latents: Transformer Penultimate Space (64D) -> K-Means (K=16)

All notifications are automatically sent to: https://ntfy.sh/woody_videos_seb
===============================================================================
"""

import os
import pickle
import sys
import time
import urllib.request

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

from lib.models.action_transformer import CausalActionTransformer
from lib.models.selective_ssm import MambaSequenceModel

OUT_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "discrete_bottleneck_benchmark.log")
from fpcode.common.notify import send_ntfy

def log(msg: str, notify: bool = False):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if notify:
        send_ntfy(msg)

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True); Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))

def js_loss_torch(P_pred: torch.Tensor, Q_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    P = torch.clamp(P_pred, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True)
    Q = torch.clamp(Q_true, min=eps, max=1.0)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    kl_p = torch.sum(P * torch.log2(P / M), dim=-1)
    kl_q = torch.sum(Q * torch.log2(Q / M), dim=-1)
    return torch.mean(0.5 * (kl_p + kl_q))


def extract_latents(model: nn.Module, tok_tensor: torch.Tensor, dur_tensor: torch.Tensor, mask_tensor: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    """Extracts penultimate pooled representations (h in R^64) from deep models."""
    model.eval()
    latents = []
    n_samples = len(tok_tensor)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_tok = tok_tensor[i : i + batch_size]
            b_dur = dur_tensor[i : i + batch_size]
            b_mask = mask_tensor[i : i + batch_size]

            # Forward pass up to pooling
            dur_log = torch.log1p(torch.clamp(b_dur, min=0.0)).unsqueeze(-1)

            if hasattr(model, 'layers'): # Mamba
                x = model.token_embed(b_tok) + model.dur_proj(dur_log)
                for layer, norm in zip(model.layers, model.norms):
                    x = x + layer(norm(x), b_mask)
                x = model.final_norm(x)
            else: # Transformer
                positions = torch.arange(b_tok.shape[1], device=b_tok.device).unsqueeze(0).expand(b_tok.shape[0], b_tok.shape[1])
                x = model.token_embed(b_tok) + model.dur_proj(dur_log) + model.pos_embed(positions)
                x = model.dropout(x)
                for block in model.blocks:
                    x = block(x, b_mask)
                x = model.final_ln(x)

            mask_f = b_mask.unsqueeze(-1).float()
            pool = torch.sum(x * mask_f, dim=1) / torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
            latents.append(pool.cpu().numpy())
    return np.vstack(latents)


def main():
    log("=" * 80, notify=True)
    log("LAUNCHING DISCRETE STRATEGY (K=16) BOTTLENECK UPGRADE EXPERIMENT", notify=True)
    log("=" * 80)

    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    sessions = []
    t0 = time.time()

    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        max_frame = bouts[-1]["end_frame"]
        N = max_frame + 1

        frame_sylls = np.zeros(N, dtype=np.int32)
        frame_spatial = np.zeros((N, 6), dtype=np.float32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLS - 1)
            frame_sylls[sf : ef + 1] = s_id
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                frame_spatial[sf : ef + 1, :4] = vec[:4]
                frame_spatial[sf : ef + 1, 4] = vec[10]
                frame_spatial[sf : ef + 1, 5] = vec[12]

        oh_frame = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        oh_frame[np.arange(N), frame_sylls] = 1.0
        cum_frame = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_frame, axis=0)])

        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool)
        onsets[0] = True; onsets[diffs] = True

        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        oh_bout[onsets, frame_sylls[onsets]] = 1.0
        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

        next_bout_ptrs = np.zeros(N, dtype=np.int32)
        cur = 0
        for d in diffs:
            next_bout_ptrs[cur:d] = d
            cur = d
        next_bout_ptrs[cur:] = N

        bout_seq = frame_sylls[onsets]
        bout_onsets = np.where(onsets)[0]
        bout_durations = np.zeros(len(bout_onsets), dtype=np.float32)
        for i in range(len(bout_onsets) - 1):
            bout_durations[i] = (bout_onsets[i+1] - bout_onsets[i]) / FPS
        bout_durations[-1] = (N - bout_onsets[-1]) / FPS

        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])

        sessions.append({
            "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
            "cum_frame": cum_frame, "cum_bout": cum_bout,
            "next_bout_ptrs": next_bout_ptrs,
            "cum_spatial": cum_spatial, "cum_spatial_sq": cum_spatial_sq,
            "bout_seq": bout_seq, "bout_onsets": bout_onsets,
            "bout_durations": bout_durations
        })

    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s.")

    all_time, all_sp, all_curr_oh = [], [], []
    all_bow_cnt, all_bow_time, all_trans_b = [], [], []
    all_seq_tokens, all_seq_durations, all_seq_lens = [], [], []
    all_tgt, all_sess = [], []

    MAX_SEQ_LEN = 128

    for s in sessions:
        sess_idx = s["sess_idx"]
        N = s["N"]
        if N < W_FRAMES + H_FRAMES + 200: continue

        t_anchors = np.arange(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES)
        t_tgt_start = s["next_bout_ptrs"][t_anchors]
        valid = (t_tgt_start + H_FRAMES < N)
        t_anchors = t_anchors[valid]
        t_tgt_start = t_tgt_start[valid]
        M = len(t_anchors)
        if M == 0: continue

        curr_syll = s["frame_sylls"][t_anchors]

        t_norm = t_anchors / float(N)
        time_feat = np.column_stack([t_norm, t_norm**2]).astype(np.float32)

        sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
        sp_mean = sp_sum / float(W_FRAMES)
        sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
        sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
        spatial_feat = np.hstack([sp_mean, np.sqrt(sp_var)]).astype(np.float32)

        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        pb_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        pf_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        # Bout Transitions
        b_onsets = s["bout_onsets"]
        b_seq = s["bout_seq"]
        b_dur = s["bout_durations"]
        trans_b = np.zeros((M, NUM_SYLLS * NUM_SYLLS), dtype=np.float32)
        i_starts = np.searchsorted(b_onsets, t_anchors - W_FRAMES)
        i_ends = np.searchsorted(b_onsets, t_anchors)

        tok_arr = np.zeros((M, MAX_SEQ_LEN), dtype=np.int64)
        dur_arr = np.zeros((M, MAX_SEQ_LEN), dtype=np.float32)
        len_arr = np.zeros(M, dtype=np.int32)

        for i in range(M):
            i0, i1 = i_starts[i], i_ends[i]
            if i1 - i0 > 1:
                sub_b = b_seq[i0:i1]
                sub_bigrams = sub_b[:-1] * NUM_SYLLS + sub_b[1:]
                b_c = np.bincount(sub_bigrams, minlength=NUM_SYLLS * NUM_SYLLS).reshape(NUM_SYLLS, NUM_SYLLS)
                rs = b_c.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
                trans_b[i] = ((b_c + 1e-4) / (rs + 1e-4 * NUM_SYLLS)).flatten()
            else:
                trans_b[i] = 1.0 / NUM_SYLLS

            sub_seq = b_seq[i0:i1]
            sub_dur = b_dur[i0:i1]
            L = min(len(sub_seq), MAX_SEQ_LEN)
            if L > 0:
                tok_arr[i, :L] = sub_seq[-L:]
                dur_arr[i, :L] = sub_dur[-L:]
                len_arr[i] = L

        curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
        curr_oh[np.arange(M), curr_syll] = 1.0

        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        all_time.append(time_feat); all_sp.append(spatial_feat); all_curr_oh.append(curr_oh)
        all_bow_cnt.append(pb_dist); all_bow_time.append(pf_dist); all_trans_b.append(trans_b)
        all_seq_tokens.append(tok_arr); all_seq_durations.append(dur_arr); all_seq_lens.append(len_arr)
        all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    X_time = np.vstack(all_time); X_spatial = np.vstack(all_sp); X_curr_oh = np.vstack(all_curr_oh)
    X_bow_cnt = np.vstack(all_bow_cnt); X_bow_time = np.vstack(all_bow_time)
    X_bow_full = np.hstack([X_bow_cnt, X_bow_time])
    X_tb = np.vstack(all_trans_b)

    X_tok = np.vstack(all_seq_tokens); X_dur = np.vstack(all_seq_durations); X_len = np.concatenate(all_seq_lens)
    Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

    log(f"Extracted features for {len(Y_tgt):,} samples.")

    # -------------------------------------------------------------------------
    # 5-Fold GroupKFold Cross-Validation Loop
    # -------------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)

    results = {
        "1. Base Macro + Posture (P_3)": [],
        "2. Traditional Strategy K=16 (Transition-SVD)": [],
        "3. Duration-BoW Strategy K=16 (Counts+Dwell SVD)": [],
        "4. Mamba-1 Strategy K=16 (SSM Latents)": [],
        "5. Action-GPT Strategy K=16 (Transformer Latents)": [],
        "6. Continuous Transformer (Upper Bound)": []
    }

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_bow_full, groups=groups)):
        fold_num = fold + 1
        log(f"\n>>> Running Bottleneck Upgrade Fold {fold_num}/5 <<<")

        Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        js_blind = js_vec(Y_te, p_glob)

        X_tr_p3 = np.hstack([X_time[tr_idx], X_spatial[tr_idx], X_curr_oh[tr_idx]])
        X_te_p3 = np.hstack([X_time[te_idx], X_spatial[te_idx], X_curr_oh[te_idx]])

        # 1. Base Macro + Posture (P3)
        p_p3 = np.clip(Ridge(5.0).fit(X_tr_p3, Y_tr).predict(X_te_p3), 1e-4, 1.0)
        p_p3 /= np.sum(p_p3, axis=1, keepdims=True)
        gain_p3 = ((js_blind - js_vec(Y_te, p_p3)) / js_blind) * 100.0
        results["1. Base Macro + Posture (P_3)"].append(gain_p3)
        log(f"  [Fold {fold_num}] 1. Macro + Posture (P3): +{gain_p3:.2f}%")

        # 2. Traditional Strategy K=16 (Transition SVD)
        svd_trans = TruncatedSVD(n_components=24, random_state=42)
        z_tr_trans = normalize(svd_trans.fit_transform(X_tb[tr_idx]), norm='l2')
        z_te_trans = normalize(svd_trans.transform(X_tb[te_idx]), norm='l2')
        km_trans = KMeans(n_clusters=16, random_state=42, n_init=5).fit(z_tr_trans)

        oh_tr_trans = np.eye(16, dtype=np.float32)[km_trans.labels_]
        oh_te_trans = np.eye(16, dtype=np.float32)[km_trans.predict(z_te_trans)]

        p_p4_trans = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, oh_tr_trans]), Y_tr).predict(np.hstack([X_te_p3, oh_te_trans])), 1e-4, 1.0)
        p_p4_trans /= np.sum(p_p4_trans, axis=1, keepdims=True)
        gain_trans = ((js_blind - js_vec(Y_te, p_p4_trans)) / js_blind) * 100.0
        results["2. Traditional Strategy K=16 (Transition-SVD)"].append(gain_trans)
        log(f"  [Fold {fold_num}] 2. Traditional K=16: +{gain_trans:.2f}%")

        # 3. Duration-BoW Strategy K=16 (SVD on 200D counts + times)
        svd_bow = TruncatedSVD(n_components=24, random_state=42)
        z_tr_bow = normalize(svd_bow.fit_transform(X_bow_full[tr_idx]), norm='l2')
        z_te_bow = normalize(svd_bow.transform(X_bow_full[te_idx]), norm='l2')
        km_bow = KMeans(n_clusters=16, random_state=42, n_init=5).fit(z_tr_bow)

        oh_tr_bow = np.eye(16, dtype=np.float32)[km_bow.labels_]
        oh_te_bow = np.eye(16, dtype=np.float32)[km_bow.predict(z_te_bow)]

        p_p4_bow = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, oh_tr_bow]), Y_tr).predict(np.hstack([X_te_p3, oh_te_bow])), 1e-4, 1.0)
        p_p4_bow /= np.sum(p_p4_bow, axis=1, keepdims=True)
        gain_bow = ((js_blind - js_vec(Y_te, p_p4_bow)) / js_blind) * 100.0
        results["3. Duration-BoW Strategy K=16 (Counts+Dwell SVD)"].append(gain_bow)
        log(f"  [Fold {fold_num}] 3. Duration-BoW K=16: +{gain_bow:.2f}%")

        # 4. PyTorch Setup for Deep Models
        tok_tr = torch.tensor(X_tok[tr_idx], dtype=torch.long, device=DEVICE)
        dur_tr = torch.tensor(X_dur[tr_idx], dtype=torch.float32, device=DEVICE)
        len_tr = torch.tensor(X_len[tr_idx], dtype=torch.long, device=DEVICE)
        mask_tr = torch.arange(MAX_SEQ_LEN, device=DEVICE).unsqueeze(0) < len_tr.unsqueeze(1)
        y_tr_t = torch.tensor(Y_tr, dtype=torch.float32, device=DEVICE)

        tok_te = torch.tensor(X_tok[te_idx], dtype=torch.long, device=DEVICE)
        dur_te = torch.tensor(X_dur[te_idx], dtype=torch.float32, device=DEVICE)
        len_te = torch.tensor(X_len[te_idx], dtype=torch.long, device=DEVICE)
        mask_te = torch.arange(MAX_SEQ_LEN, device=DEVICE).unsqueeze(0) < len_te.unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(tok_tr, dur_tr, mask_tr, y_tr_t),
            batch_size=128,
            shuffle=True
        )

        # 5. Mamba-1 Strategy K=16
        torch.cuda.empty_cache()
        mamba = MambaSequenceModel(vocab_size=NUM_SYLLS, d_model=64, d_state=16, num_layers=2).to(DEVICE)
        opt_m = optim.AdamW(mamba.parameters(), lr=0.003, weight_decay=1e-4)
        mamba.train()
        for ep in range(12):
            for b_tok, b_dur, b_mask, b_y in train_loader:
                opt_m.zero_grad()
                loss = js_loss_torch(mamba(b_tok, b_dur, b_mask), b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
                opt_m.step()

        h_tr_mamba = normalize(extract_latents(mamba, tok_tr, dur_tr, mask_tr, batch_size=128), norm='l2')
        h_te_mamba = normalize(extract_latents(mamba, tok_te, dur_te, mask_te, batch_size=128), norm='l2')

        km_mamba = KMeans(n_clusters=16, random_state=42, n_init=5).fit(h_tr_mamba)
        oh_tr_mamba = np.eye(16, dtype=np.float32)[km_mamba.labels_]
        oh_te_mamba = np.eye(16, dtype=np.float32)[km_mamba.predict(h_te_mamba)]

        p_p4_mamba = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, oh_tr_mamba]), Y_tr).predict(np.hstack([X_te_p3, oh_te_mamba])), 1e-4, 1.0)
        p_p4_mamba /= np.sum(p_p4_mamba, axis=1, keepdims=True)
        gain_mamba = ((js_blind - js_vec(Y_te, p_p4_mamba)) / js_blind) * 100.0
        results["4. Mamba-1 Strategy K=16 (SSM Latents)"].append(gain_mamba)
        log(f"  [Fold {fold_num}] 4. Mamba K=16: +{gain_mamba:.2f}%")

        # 6. Action-GPT Strategy K=16
        torch.cuda.empty_cache()
        xformer = CausalActionTransformer(vocab_size=NUM_SYLLS, d_model=64, n_heads=4, num_layers=2).to(DEVICE)
        opt_x = optim.AdamW(xformer.parameters(), lr=0.003, weight_decay=1e-4)
        xformer.train()
        for ep in range(12):
            for b_tok, b_dur, b_mask, b_y in train_loader:
                opt_x.zero_grad()
                loss = js_loss_torch(xformer(b_tok, b_dur, b_mask), b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(xformer.parameters(), 1.0)
                opt_x.step()

        h_tr_xform = normalize(extract_latents(xformer, tok_tr, dur_tr, mask_tr, batch_size=128), norm='l2')
        h_te_xform = normalize(extract_latents(xformer, tok_te, dur_te, mask_te, batch_size=128), norm='l2')

        km_xform = KMeans(n_clusters=16, random_state=42, n_init=5).fit(h_tr_xform)
        oh_tr_xform = np.eye(16, dtype=np.float32)[km_xform.labels_]
        oh_te_xform = np.eye(16, dtype=np.float32)[km_xform.predict(h_te_xform)]

        p_p4_xform = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, oh_tr_xform]), Y_tr).predict(np.hstack([X_te_p3, oh_te_xform])), 1e-4, 1.0)
        p_p4_xform /= np.sum(p_p4_xform, axis=1, keepdims=True)
        gain_xform = ((js_blind - js_vec(Y_te, p_p4_xform)) / js_blind) * 100.0
        results["5. Action-GPT Strategy K=16 (Transformer Latents)"].append(gain_xform)
        log(f"  [Fold {fold_num}] 5. Action-GPT K=16: +{gain_xform:.2f}%")

        # Continuous Transformer upper bound reference
        xformer.eval()
        with torch.no_grad():
            p_cont = []
            for i in range(0, len(tok_te), 128):
                p_cont.append(xformer(tok_te[i:i+128], dur_te[i:i+128], mask_te[i:i+128]).cpu().numpy())
            p_cont = np.vstack(p_cont)
        gain_cont = ((js_blind - js_vec(Y_te, p_cont)) / js_blind) * 100.0
        results["6. Continuous Transformer (Upper Bound)"].append(gain_cont)
        log(f"  [Fold {fold_num}] 6. Continuous Upper Bound: +{gain_cont:.2f}%")

        send_ntfy(f"✅ Bottleneck Fold {fold_num}/5 Done! Trad K=16: +{gain_trans:.1f}%, BoW K=16: +{gain_bow:.1f}%, GPT K=16: +{gain_xform:.1f}%")

    # -------------------------------------------------------------------------
    # Final Compilation
    # -------------------------------------------------------------------------
    log("\n" + "=" * 90, notify=True)
    log("DISCRETE STRATEGY (K=16) BOTTLENECK COMPARISON RESULTS (W=68.3s -> H=120s)", notify=True)
    log("=" * 90)

    summary_rows = []
    trad_mean = np.mean(results["2. Traditional Strategy K=16 (Transition-SVD)"])

    log(f"{'Strategy Clustering Model (4-Bit Bottleneck)':<50} | {'Predictive Gain':<18} | {'Delta vs Trad K=16':<20}")
    log("-" * 92)
    for k, vals in results.items():
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
        delta = m - trad_mean
        status = "Baseline (4-bit)" if "Traditional" in k else f"{delta:+5.2f}%"
        log(f"{k:<50} | +{m:5.2f}% ± {ci:4.2f}% | {status:<20}")
        summary_rows.append({
            "Clustering_Pipeline": k,
            "Mean_Predictive_Gain": m,
            "CI_95": ci,
            "Delta_vs_Traditional_K16": delta
        })
    log("=" * 90)

    # Save CSV
    df_res = pd.DataFrame(summary_rows)
    csv_path = os.path.join(TAB_DIR, "discrete_bottleneck_comparison.csv")
    df_res.to_csv(csv_path, index=False)
    log(f"Saved CSV table: {csv_path}")

    # Render Bar Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    archs = [r["Clustering_Pipeline"] for r in summary_rows]
    means = [r["Mean_Predictive_Gain"] for r in summary_rows]
    cis = [r["CI_95"] for r in summary_rows]
    colors = ['#9B9B9B', '#F5A623', '#4A90E2', '#BD10E0', '#7ED321', '#50E3C2']

    bars = ax.bar(archs, means, yerr=cis, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.axhline(trad_mean, color='#F5A623', linestyle='--', linewidth=1.8, label=f'Traditional SVD K=16 (+{trad_mean:.2f}%)')

    ax.set_ylabel('Predictive Gain (% $D_{JS}$ Reduction vs. Blind Null)', fontsize=12, fontweight='bold')
    ax.set_title('Upgrading the 4-Bit Strategy Bottleneck (K=16 Discrete Clusters)\n(30 Sessions, 5-Fold GroupKFold CV, W = 68.3s, H = 120s)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 42)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_xticklabels(archs, rotation=22, ha='right', fontsize=10, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.8),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "discrete_bottleneck_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log(f"Saved figure: {fig_path}", notify=True)

    best_discrete = df_res.iloc[df_res.iloc[:5]['Mean_Predictive_Gain'].argmax()]
    send_ntfy(f"🏆 Bottleneck Upgrade Complete! Best 4-bit Strategy: {best_discrete['Clustering_Pipeline']} (+{best_discrete['Mean_Predictive_Gain']:.2f}%)")


if __name__ == "__main__":
    main()
