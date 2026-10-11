#!/usr/bin/env python3
"""
===============================================================================
Predictor Hierarchy & Information Decomposition Benchmark (P_0 -> P_5)
===============================================================================
Evaluates how increasing tiers of input information elevate predictive power
across the top continuous architectures under 5-fold session-grouped CV:
  - P_0: Blind Null (Marginal frequency, 0% reference)
  - P_1: Time Drift {t, t^2}
  - P_2: Macro Context (Time + Arena Centroid, Velocity, Dispersion)
  - P_3: Markovian + Macro (P_2 + Instantaneous Posture S_t)
  - P_4: Discrete Strategy (P_3 + 4-Bit K=16 Strategy Clusters)
  - P_5: Full Continuous Past Sequence (P_3 + Continuous Model Latents):
      * P_5 (Linear BoW): [Counts + Dwell Times] in R^200
      * P_5 (Action2Vec): Mikolov SGNS (p=32)
      * P_5 (Mamba-1 SSM): Selective Recurrent Latents (64D)
      * P_5 (Causal Action-GPT): Multi-Head Self-Attention Latents (64D)

Push notifications are sent to: https://ntfy.sh/woody_videos_seb
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

from studies.behavior.language_models.lib.models.action2vec import train_action2vec
from studies.behavior.language_models.lib.models.action_transformer import CausalActionTransformer
from studies.behavior.language_models.lib.models.selective_ssm import MambaSequenceModel

OUT_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "predictor_hierarchy_benchmark.log")
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
    model.eval()
    latents = []
    n_samples = len(tok_tensor)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_tok = tok_tensor[i : i + batch_size]
            b_dur = dur_tensor[i : i + batch_size]
            b_mask = mask_tensor[i : i + batch_size]

            dur_log = torch.log1p(torch.clamp(b_dur, min=0.0)).unsqueeze(-1)

            if hasattr(model, 'layers'):  # Mamba
                x = model.token_embed(b_tok) + model.dur_proj(dur_log)
                for layer, norm in zip(model.layers, model.norms):
                    x = x + layer(norm(x), b_mask)
                x = model.final_norm(x)
            else:  # Transformer
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
    log("STARTING PREDICTOR HIERARCHY (P_0 -> P_5) INFORMATION DECOMPOSITION", notify=True)
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

    all_p1, all_p2, all_p3 = [], [], []
    all_bow_cnt, all_bow_time = [], []
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

        # P_1: Time drift
        t_norm = t_anchors / float(N)
        p1_feat = np.column_stack([t_norm, t_norm**2]).astype(np.float32)

        # P_2: Macro context (Spatial + Time)
        sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
        sp_mean = sp_sum / float(W_FRAMES)
        sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
        sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
        spatial_feat = np.hstack([sp_mean, np.sqrt(sp_var)]).astype(np.float32)
        p2_feat = np.hstack([p1_feat, spatial_feat])

        # P_3: Markovian + Macro (P_2 + Posture)
        curr_syll = s["frame_sylls"][t_anchors]
        curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
        curr_oh[np.arange(M), curr_syll] = 1.0
        p3_feat = np.hstack([p2_feat, curr_oh])

        # P_5 Continuous Features
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        pb_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        pf_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        b_onsets = s["bout_onsets"]
        b_seq = s["bout_seq"]
        b_dur = s["bout_durations"]
        i_starts = np.searchsorted(b_onsets, t_anchors - W_FRAMES)
        i_ends = np.searchsorted(b_onsets, t_anchors)

        tok_arr = np.zeros((M, MAX_SEQ_LEN), dtype=np.int64)
        dur_arr = np.zeros((M, MAX_SEQ_LEN), dtype=np.float32)
        len_arr = np.zeros(M, dtype=np.int32)

        for i in range(M):
            i0, i1 = i_starts[i], i_ends[i]
            sub_seq = b_seq[i0:i1]
            sub_dur = b_dur[i0:i1]
            L = min(len(sub_seq), MAX_SEQ_LEN)
            if L > 0:
                tok_arr[i, :L] = sub_seq[-L:]
                dur_arr[i, :L] = sub_dur[-L:]
                len_arr[i] = L

        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        all_p1.append(p1_feat); all_p2.append(p2_feat); all_p3.append(p3_feat)
        all_bow_cnt.append(pb_dist); all_bow_time.append(pf_dist)
        all_seq_tokens.append(tok_arr); all_seq_durations.append(dur_arr); all_seq_lens.append(len_arr)
        all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    X_p1 = np.vstack(all_p1); X_p2 = np.vstack(all_p2); X_p3 = np.vstack(all_p3)
    X_bow_cnt = np.vstack(all_bow_cnt); X_bow_time = np.vstack(all_bow_time)
    X_bow_full = np.hstack([X_bow_cnt, X_bow_time])

    X_tok = np.vstack(all_seq_tokens); X_dur = np.vstack(all_seq_durations); X_len = np.concatenate(all_seq_lens)
    Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

    log(f"Extracted features for {len(Y_tgt):,} samples across {len(np.unique(groups))} sessions.")

    # -------------------------------------------------------------------------
    # 5-Fold GroupKFold Cross-Validation Loop
    # -------------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)

    results = {
        "P_0 (Blind Null)": [],
        "P_1 (Time Drift)": [],
        "P_2 (Macro Context)": [],
        "P_3 (Markovian + Macro)": [],
        "P_4 (Discrete K=16 Strategy)": [],
        "P_5 (Linear BoW Ceiling)": [],
        "P_5 (Action2Vec Continuous)": [],
        "P_5 (Mamba-1 SSM Continuous)": [],
        "P_5 (Action-GPT Transformer)": []
    }

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_p3, groups=groups)):
        fold_num = fold + 1
        log(f"\n>>> Running Predictor Hierarchy Fold {fold_num}/5 <<<")

        Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        js_blind = js_vec(Y_te, p_glob)

        # P_0: Blind Null
        results["P_0 (Blind Null)"].append(0.0)

        # P_1: Time Drift
        p_p1 = np.clip(Ridge(5.0).fit(X_p1[tr_idx], Y_tr).predict(X_p1[te_idx]), 1e-4, 1.0)
        p_p1 /= np.sum(p_p1, axis=1, keepdims=True)
        gain_p1 = ((js_blind - js_vec(Y_te, p_p1)) / js_blind) * 100.0
        results["P_1 (Time Drift)"].append(gain_p1)

        # P_2: Macro Context (Time + Space)
        p_p2 = np.clip(Ridge(5.0).fit(X_p2[tr_idx], Y_tr).predict(X_p2[te_idx]), 1e-4, 1.0)
        p_p2 /= np.sum(p_p2, axis=1, keepdims=True)
        gain_p2 = ((js_blind - js_vec(Y_te, p_p2)) / js_blind) * 100.0
        results["P_2 (Macro Context)"].append(gain_p2)

        # P_3: Markovian + Macro (P_2 + Posture)
        p_p3 = np.clip(Ridge(5.0).fit(X_p3[tr_idx], Y_tr).predict(X_p3[te_idx]), 1e-4, 1.0)
        p_p3 /= np.sum(p_p3, axis=1, keepdims=True)
        gain_p3 = ((js_blind - js_vec(Y_te, p_p3)) / js_blind) * 100.0
        results["P_3 (Markovian + Macro)"].append(gain_p3)

        # P_5 (Linear BoW Ceiling)
        X_tr_bow = np.hstack([X_p3[tr_idx], X_bow_full[tr_idx]])
        X_te_bow = np.hstack([X_p3[te_idx], X_bow_full[te_idx]])
        p_p5_bow = np.clip(Ridge(1.0).fit(X_tr_bow, Y_tr).predict(X_te_bow), 1e-4, 1.0)
        p_p5_bow /= np.sum(p_p5_bow, axis=1, keepdims=True)
        gain_p5_bow = ((js_blind - js_vec(Y_te, p_p5_bow)) / js_blind) * 100.0
        results["P_5 (Linear BoW Ceiling)"].append(gain_p5_bow)

        # P_5 (Action2Vec Continuous)
        tr_sess = np.unique(groups[tr_idx])
        tr_seqs = [s["bout_seq"] for s in sessions if s["sess_idx"] in tr_sess]
        a2v_embeds = train_action2vec(session_bout_sequences=tr_seqs, vocab_size=NUM_SYLLS, embedding_dim=32, window_size=3, epochs=10, device=DEVICE)

        X_a2v_tr = np.hstack([X_p3[tr_idx], X_bow_cnt[tr_idx] @ a2v_embeds])
        X_a2v_te = np.hstack([X_p3[te_idx], X_bow_cnt[te_idx] @ a2v_embeds])
        p_p5_a2v = np.clip(Ridge(1.0).fit(X_a2v_tr, Y_tr).predict(X_a2v_te), 1e-4, 1.0)
        p_p5_a2v /= np.sum(p_p5_a2v, axis=1, keepdims=True)
        gain_p5_a2v = ((js_blind - js_vec(Y_te, p_p5_a2v)) / js_blind) * 100.0
        results["P_5 (Action2Vec Continuous)"].append(gain_p5_a2v)

        # Deep Sequence Models (PyTorch Tensors)
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

        # P_5 (Mamba-1 SSM Continuous)
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

        h_tr_mamba = extract_latents(mamba, tok_tr, dur_tr, mask_tr, batch_size=128)
        h_te_mamba = extract_latents(mamba, tok_te, dur_te, mask_te, batch_size=128)

        X_mamba_tr = np.hstack([X_p3[tr_idx], h_tr_mamba])
        X_mamba_te = np.hstack([X_p3[te_idx], h_te_mamba])
        p_p5_mamba = np.clip(Ridge(1.0).fit(X_mamba_tr, Y_tr).predict(X_mamba_te), 1e-4, 1.0)
        p_p5_mamba /= np.sum(p_p5_mamba, axis=1, keepdims=True)
        gain_p5_mamba = ((js_blind - js_vec(Y_te, p_p5_mamba)) / js_blind) * 100.0
        results["P_5 (Mamba-1 SSM Continuous)"].append(gain_p5_mamba)

        # P_5 (Action-GPT Transformer)
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

        h_tr_xform = extract_latents(xformer, tok_tr, dur_tr, mask_tr, batch_size=128)
        h_te_xform = extract_latents(xformer, tok_te, dur_te, mask_te, batch_size=128)

        X_xform_tr = np.hstack([X_p3[tr_idx], h_tr_xform])
        X_xform_te = np.hstack([X_p3[te_idx], h_te_xform])
        p_p5_xform = np.clip(Ridge(1.0).fit(X_xform_tr, Y_tr).predict(X_xform_te), 1e-4, 1.0)
        p_p5_xform /= np.sum(p_p5_xform, axis=1, keepdims=True)
        gain_p5_xform = ((js_blind - js_vec(Y_te, p_p5_xform)) / js_blind) * 100.0
        results["P_5 (Action-GPT Transformer)"].append(gain_p5_xform)

        # P_4: Discrete Action-GPT Strategy K=16
        km_xform = KMeans(n_clusters=16, random_state=42, n_init=5).fit(normalize(h_tr_xform, norm='l2'))
        oh_tr_xform = np.eye(16, dtype=np.float32)[km_xform.labels_]
        oh_te_xform = np.eye(16, dtype=np.float32)[km_xform.predict(normalize(h_te_xform, norm='l2'))]

        p_p4 = np.clip(Ridge(5.0).fit(np.hstack([X_p3[tr_idx], oh_tr_xform]), Y_tr).predict(np.hstack([X_p3[te_idx], oh_te_xform])), 1e-4, 1.0)
        p_p4 /= np.sum(p_p4, axis=1, keepdims=True)
        gain_p4 = ((js_blind - js_vec(Y_te, p_p4)) / js_blind) * 100.0
        results["P_4 (Discrete K=16 Strategy)"].append(gain_p4)

        log(f"  [Fold {fold_num}] P1: +{gain_p1:.2f}%, P2: +{gain_p2:.2f}%, P3: +{gain_p3:.2f}%, P4(K=16): +{gain_p4:.2f}%, P5(BoW): +{gain_p5_bow:.2f}%, P5(GPT): +{gain_p5_xform:.2f}%")
        send_ntfy(f"✅ P_n Fold {fold_num}/5 Done! P3(Posture): +{gain_p3:.1f}%, P4(K=16): +{gain_p4:.1f}%, P5(GPT): +{gain_p5_xform:.1f}%")

    # -------------------------------------------------------------------------
    # Final Compilation & Reporting
    # -------------------------------------------------------------------------
    log("\n" + "=" * 90, notify=True)
    log("FINAL PREDICTOR HIERARCHY RESULTS (P_0 -> P_5) (W=68.3s -> H=120s)", notify=True)
    log("=" * 90)

    summary_rows = []
    log(f"{'Predictor Tier (Information Level)':<42} | {'Mean Gain (% D_JS)':<22} | {'95% CI':<15}")
    log("-" * 85)

    for k, vals in results.items():
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
        log(f"{k:<42} | +{m:5.2f}%                | ± {ci:4.2f}%")
        summary_rows.append({
            "Predictor_Tier": k,
            "Mean_Predictive_Gain": m,
            "CI_95": ci
        })
    log("=" * 90)

    # Save CSV
    df_res = pd.DataFrame(summary_rows)
    csv_path = os.path.join(TAB_DIR, "predictor_hierarchy_results.csv")
    df_res.to_csv(csv_path, index=False)
    log(f"Saved CSV table: {csv_path}")

    # -------------------------------------------------------------------------
    # Publication Figures
    # -------------------------------------------------------------------------
    # Figure 1: Cumulative Information Hierarchy
    fig, ax = plt.subplots(figsize=(11, 6))

    tiers = [r["Predictor_Tier"] for r in summary_rows]
    means = [r["Mean_Predictive_Gain"] for r in summary_rows]
    cis = [r["CI_95"] for r in summary_rows]
    colors = ['#4A4A4A', '#F5A623', '#F8E71C', '#7ED321', '#50E3C2', '#4A90E2', '#9013FE', '#BD10E0', '#D0021B']

    bars = ax.bar(tiers, means, yerr=cis, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Predictive Gain (% $D_{JS}$ Reduction vs. Blind Null)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Information Hierarchy ($\\mathcal{P}_0 \to \\mathcal{P}_5$)\n(30 Sessions, 5-Fold GroupKFold CV, W = 68.3s, H = 120s)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 42)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_xticklabels(tiers, rotation=25, ha='right', fontsize=10, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.8),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "predictor_hierarchy_cumulative.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    log(f"Saved Cumulative Figure: {fig1_path}", notify=True)

    # Final Mobile Push Notification
    gpt_p5 = df_res[df_res['Predictor_Tier'].str.contains('Action-GPT')].iloc[0]
    p4_stat = df_res[df_res['Predictor_Tier'].str.contains('P_4')].iloc[0]
    p3_stat = df_res[df_res['Predictor_Tier'].str.contains('P_3')].iloc[0]
    send_ntfy(f"🏆 P_n Benchmark Complete! P3(Posture): +{p3_stat['Mean_Predictive_Gain']:.1f}%, P4(K=16): +{p4_stat['Mean_Predictive_Gain']:.1f}%, P5(GPT): +{gpt_p5['Mean_Predictive_Gain']:.1f}%")


if __name__ == "__main__":
    main()
