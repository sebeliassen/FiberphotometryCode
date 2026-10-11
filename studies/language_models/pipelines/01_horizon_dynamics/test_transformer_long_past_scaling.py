#!/usr/bin/env python3
"""
===============================================================================
Experiment: Long-Past History Scaling & Dilution Resistance in Deep Sequence Models
===============================================================================
Evaluates whether a Causal Action Transformer (with self-attention & positional
encodings) can effectively leverage extended past memory windows (W = 30s to 360s)
for a fixed future horizon (H = 120s) without suffering from the temporal dilution
decay that affects linear Bag-of-Words models.

Experimental Matrix:
  - Fixed Future Target: H = 120.0s (2 minutes)
  - Past Windows (W): [30s, 68.3s, 120s, 240s, 360s]
  - Cross-Validation: Strict Session-Grouped 5-Fold (GroupKFold across 30 sessions)
  - Models:
      1. Linear Bag-of-Words (Counts + Dwell Times in R^200)
      2. Causal Action Transformer (2 layers, 4 heads, d=64, max_len=1024)
===============================================================================
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

from studies.behavior.language_models.lib.models.action_transformer import CausalActionTransformer

OUT_DIR = os.path.join(LM_DIR, "output", "01_horizon_dynamics")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "transformer_long_past_scaling.log")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/518dcf13-db16-4fec-bc0b-b0a6ea54a502"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

FPS = 80.025
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 10.0  # Clean sampling stride for fast execution
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
PAST_WINDOWS_SEC = [30.0, 68.3, 120.0, 240.0, 360.0]
NUM_SYLLS = 100
MAX_SEQ_LEN = 768
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

def load_preprocessed_sessions():
    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    log(f"Loading session bout cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    sessions = []
    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        max_frame = bouts[-1]["end_frame"]
        N = max_frame + 1

        frame_sylls = np.zeros(N, dtype=np.int32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLS - 1)
            frame_sylls[sf : ef + 1] = s_id

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

        sessions.append({
            "sess_idx": sess_idx, "N": N,
            "cum_frame": cum_frame, "cum_bout": cum_bout,
            "next_bout_ptrs": next_bout_ptrs,
            "bout_seq": bout_seq, "bout_onsets": bout_onsets,
            "bout_durations": bout_durations
        })

    log(f"Successfully prepared {len(sessions)} sessions on device {DEVICE}.")
    return sessions

def extract_features_for_window(sessions, w_sec):
    w_frames = int(w_sec * FPS)
    max_w_frames = int(max(PAST_WINDOWS_SEC) * FPS)

    all_bow, all_tokens, all_durations, all_lens, all_targets, all_sess = [], [], [], [], [], []

    for s in sessions:
        sess_idx = s["sess_idx"]
        N = s["N"]
        if N < max_w_frames + H_FRAMES + 200: continue

        # Align anchors identically across window sizes
        t_anchors = np.arange(max_w_frames, N - H_FRAMES, STRIDE_FRAMES)
        t_tgt_start = s["next_bout_ptrs"][t_anchors]
        valid = (t_tgt_start + H_FRAMES < N)
        t_anchors = t_anchors[valid]
        t_tgt_start = t_tgt_start[valid]
        M = len(t_anchors)
        if M == 0: continue

        # 1. Bag of Words (Counts + Dwell Times in past W)
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - w_frames]
        pb_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - w_frames]
        pf_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        bow_feat = np.hstack([pb_dist, pf_dist])

        # 2. Sequential Tokens in past W
        b_onsets = s["bout_onsets"]
        b_seq = s["bout_seq"]
        b_dur = s["bout_durations"]
        i_starts = np.searchsorted(b_onsets, t_anchors - w_frames)
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

        # 3. Future Target (Fixed H = 120s)
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        all_bow.append(bow_feat)
        all_tokens.append(tok_arr); all_durations.append(dur_arr); all_lens.append(len_arr)
        all_targets.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    return (
        np.vstack(all_bow),
        np.vstack(all_tokens),
        np.vstack(all_durations),
        np.concatenate(all_lens),
        np.vstack(all_targets),
        np.concatenate(all_sess)
    )

def main():
    log("=" * 80)
    log("STARTING LONG-PAST HISTORY SCALING BENCHMARK (BoW vs. Transformer)")
    log("=" * 80)

    sessions = load_preprocessed_sessions()
    gkf = GroupKFold(n_splits=5)

    results_summary = []

    for w_sec in PAST_WINDOWS_SEC:
        log("\n" + "-" * 60)
        log(f"EVALUATING PAST MEMORY WINDOW W = {w_sec:.1f}s (Target H = 120s)")
        log("-" * 60)

        X_bow, X_tok, X_dur, X_len, Y_tgt, groups = extract_features_for_window(sessions, w_sec)
        log(f"Extracted {len(Y_tgt):,} evaluation samples across {len(np.unique(groups))} sessions. Median sequence length: {int(np.median(X_len))} bouts.")

        fold_gains_bow = []
        fold_gains_xform = []

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_bow, groups=groups)):
            fold_num = fold + 1
            Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]

            # Blind Prior (Null)
            p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
            p_glob /= np.sum(p_glob, axis=1, keepdims=True)
            js_blind = js_vec(Y_te, p_glob)

            # 1. Linear Bag-of-Words (Ridge)
            p_bow = np.clip(Ridge(1.0).fit(X_bow[tr_idx], Y_tr).predict(X_bow[te_idx]), 1e-4, 1.0)
            p_bow /= np.sum(p_bow, axis=1, keepdims=True)
            gain_bow = ((js_blind - js_vec(Y_te, p_bow)) / js_blind) * 100.0
            fold_gains_bow.append(gain_bow)

            # 2. Causal Action Transformer
            torch.cuda.empty_cache()
            max_l_active = max(16, int(np.percentile(X_len[tr_idx], 99)))

            tok_tr = torch.tensor(X_tok[tr_idx, :max_l_active], dtype=torch.long, device=DEVICE)
            dur_tr = torch.tensor(X_dur[tr_idx, :max_l_active], dtype=torch.float32, device=DEVICE)
            len_tr = torch.tensor(X_len[tr_idx], dtype=torch.long, device=DEVICE)
            mask_tr = torch.arange(max_l_active, device=DEVICE).unsqueeze(0) < len_tr.unsqueeze(1)
            y_tr_t = torch.tensor(Y_tr, dtype=torch.float32, device=DEVICE)

            tok_te = torch.tensor(X_tok[te_idx, :max_l_active], dtype=torch.long, device=DEVICE)
            dur_te = torch.tensor(X_dur[te_idx, :max_l_active], dtype=torch.float32, device=DEVICE)
            len_te = torch.tensor(X_len[te_idx], dtype=torch.long, device=DEVICE)
            mask_te = torch.arange(max_l_active, device=DEVICE).unsqueeze(0) < len_te.unsqueeze(1)

            train_loader = DataLoader(
                TensorDataset(tok_tr, dur_tr, mask_tr, y_tr_t),
                batch_size=128,
                shuffle=True
            )

            xformer = CausalActionTransformer(
                vocab_size=NUM_SYLLS,
                d_model=64,
                n_heads=4,
                num_layers=2,
                max_len=max(1024, max_l_active + 10),
                dropout=0.05
            ).to(DEVICE)

            opt = optim.AdamW(xformer.parameters(), lr=0.003, weight_decay=1e-4)

            xformer.train()
            for ep in range(10):
                for b_tok, b_dur, b_mask, b_y in train_loader:
                    opt.zero_grad()
                    pred = xformer(b_tok, b_dur, b_mask)
                    loss = js_loss_torch(pred, b_y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(xformer.parameters(), 1.0)
                    opt.step()

            # Evaluation
            xformer.eval()
            with torch.no_grad():
                preds_te = []
                for i in range(0, len(tok_te), 128):
                    b_tok = tok_te[i : i + 128]
                    b_dur = dur_te[i : i + 128]
                    b_mask = mask_te[i : i + 128]
                    pred = xformer(b_tok, b_dur, b_mask)
                    preds_te.append(pred.cpu().numpy())
                p_xform = np.vstack(preds_te)

            gain_xform = ((js_blind - js_vec(Y_te, p_xform)) / js_blind) * 100.0
            fold_gains_xform.append(gain_xform)

            log(f"  [Fold {fold_num}/5] W={w_sec:5.1f}s -> BoW: +{gain_bow:.2f}%, Transformer: +{gain_xform:.2f}% (Delta: +{gain_xform - gain_bow:.2f}%)")

        m_bow = float(np.mean(fold_gains_bow))
        ci_bow = float(1.96 * np.std(fold_gains_bow) / np.sqrt(len(fold_gains_bow)))
        m_xf = float(np.mean(fold_gains_xform))
        ci_xf = float(1.96 * np.std(fold_gains_xform) / np.sqrt(len(fold_gains_xform)))

        log(f"==> W={w_sec:5.1f}s SUMMARY: Linear BoW = +{m_bow:.2f}% ± {ci_bow:.2f}% | Transformer = +{m_xf:.2f}% ± {ci_xf:.2f}% | Delta = +{m_xf - m_bow:.2f}%")

        results_summary.append({
            "past_window_sec": w_sec,
            "bow_mean": m_bow,
            "bow_ci": ci_bow,
            "xform_mean": m_xf,
            "xform_ci": ci_xf,
            "delta_xform_vs_bow": m_xf - m_bow
        })

    # Save Results Table
    df_res = pd.DataFrame(results_summary)
    csv_path = os.path.join(TAB_DIR, "transformer_long_past_scaling.csv")
    df_res.to_csv(csv_path, index=False)
    log(f"\nSaved CSV Results Table to: {csv_path}")

    # -------------------------------------------------------------------------
    # Render Publication / Presentation Comparison Figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    w_arr = df_res["past_window_sec"].values
    bow_arr = df_res["bow_mean"].values
    bow_ci = df_res["bow_ci"].values
    xf_arr = df_res["xform_mean"].values
    xf_ci = df_res["xform_ci"].values

    # Lines & Error Bars
    ax.errorbar(w_arr, bow_arr, yerr=bow_ci, fmt='o--', color='#1f77b4', ecolor='#1f77b4',
                elinewidth=1.5, capsize=4, markersize=7, linewidth=2.0, label='Linear Bag-of-Words (Uniform Counts)')

    ax.errorbar(w_arr, xf_arr, yerr=xf_ci, fmt='s-', color='#2ca02c', ecolor='#2ca02c',
                elinewidth=1.8, capsize=4, markersize=8, linewidth=2.4, label='Causal Action Transformer (Self-Attention)')

    # Operational Window Marker
    ax.axvline(68.3, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8, label='Standard Operational Window (68.3s)')

    # Highlight Dilution Gap
    ax.fill_between(w_arr, bow_arr, xf_arr, color='#2ca02c', alpha=0.10, label='Attention Advantage over Dilution')

    # Formatting
    ax.set_xlim(15, 375)
    ax.set_ylim(22, 40)
    ax.set_xlabel("Past Memory Window $W$ (seconds)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$ at Fixed $H=120\\text{s}$)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title("Can Attention Models Leverage Longer Behavioral Past?\n(Fixed Future Horizon $H = 120\\text{s}$)", fontsize=12, fontweight='bold', pad=10)

    ax.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc', fontsize=9.2)

    plt.tight_layout()

    fig_png = os.path.join(FIG_DIR, "transformer_vs_bow_long_past.png")
    fig_pdf = os.path.join(FIG_DIR, "transformer_vs_bow_long_past.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    # Copy to artifact dir for instant view
    artifact_png = os.path.join(ARTIFACT_DIR, "transformer_vs_bow_long_past.png")
    shutil.copy(fig_png, artifact_png)

    log(f"Saved Comparison Figure to:\n  - {fig_png}\n  - {artifact_png}")
    log("=" * 80)
    log("EXPERIMENT COMPLETE!")
    log("=" * 80)

if __name__ == "__main__":
    main()
