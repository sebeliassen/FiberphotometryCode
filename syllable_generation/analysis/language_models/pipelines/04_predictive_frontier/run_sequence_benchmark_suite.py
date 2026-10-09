#!/usr/bin/env python3
"""
===============================================================================
Master Benchmark: Canonical Sequence Modeling Suite vs. Linear BoW Ceiling
===============================================================================
Canonical Architectures Benchmarked under 5-Fold Session-Grouped CV:
  1. Bag-of-Words (BoW Baseline): Linear Ridge on [u_count, u_time] in R^200
  2. Syllable Embeddings (Action2Vec): Mikolov SGNS (p=32, window c=±3)
  3. Variable-Order Markov Model (PST): Ron, Singer, Tishby Suffix Tree Features
  4. Hidden Semi-Markov Model (HSMM): Murphy / Linderman Duration Filter (M=16)
  5. Selective State-Space Model (Mamba-1): Gu & Dao (2023) Mamba Block (< 1GB VRAM)
  6. Causal Action Transformer: Radford / Vaswani Action-GPT (d_model=64, 4 heads)
  7. Order-Sensitivity Shuffle Ablation: Shuffled Action Transformer

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

from lib.models.action2vec import train_action2vec
from lib.models.action_transformer import CausalActionTransformer
from lib.models.hsmm_duration import ExplicitDurationHSMM
from lib.models.selective_ssm import MambaSequenceModel
from lib.models.vomm_pst import ProbabilisticSuffixTree

OUT_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "sequential_benchmark.log")
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_ntfy(msg: str):
    """Sends push notification to the user's phone via ntfy.sh."""
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode('utf-8'),
            headers={'Title': 'Antigravity Model Benchmark', 'Priority': 'default'}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[ntfy warning] Failed to send ntfy notification: {e}", flush=True)

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


def predict_batched(model: nn.Module, tok_tensor: torch.Tensor, dur_tensor: torch.Tensor, mask_tensor: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    """Evaluates PyTorch model in small batches to guarantee < 1GB VRAM usage."""
    model.eval()
    preds = []
    n_samples = len(tok_tensor)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_tok = tok_tensor[i : i + batch_size]
            b_dur = dur_tensor[i : i + batch_size]
            b_mask = mask_tensor[i : i + batch_size]
            out = model(b_tok, b_dur, b_mask)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)


def main():
    log("=" * 80, notify=True)
    log("STARTING CANONICAL SEQUENCE BENCHMARK (6 ARCHITECTURES, GPU-OPTIMIZED)", notify=True)
    log("=" * 80)

    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    log(f"Loading continuous bout cache: {cache_path}")
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
            "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
            "cum_frame": cum_frame, "cum_bout": cum_bout,
            "next_bout_ptrs": next_bout_ptrs,
            "bout_seq": bout_seq, "bout_onsets": bout_onsets,
            "bout_durations": bout_durations
        })

    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------------------
    # Extract Datasets for all Evaluation Anchors
    # -------------------------------------------------------------------------
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

        # 1. BoW Counts & Frame Time
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        pb_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        pf_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        # 2. Sequence of Bouts in W window
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

        # Target: Future Discrete Bout Distribution (120s budget)
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        all_bow_cnt.append(pb_dist); all_bow_time.append(pf_dist)
        all_seq_tokens.append(tok_arr); all_seq_durations.append(dur_arr); all_seq_lens.append(len_arr)
        all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    X_bow_cnt = np.vstack(all_bow_cnt)
    X_bow_time = np.vstack(all_bow_time)
    X_bow_full = np.hstack([X_bow_cnt, X_bow_time])

    X_tok = np.vstack(all_seq_tokens)
    X_dur = np.vstack(all_seq_durations)
    X_len = np.concatenate(all_seq_lens)

    Y_tgt = np.vstack(all_tgt)
    groups = np.concatenate(all_sess)

    log(f"Extracted features for {len(Y_tgt):,} evaluation samples across {len(np.unique(groups))} sessions.")

    # -------------------------------------------------------------------------
    # 5-Fold GroupKFold Cross-Validation Loop
    # -------------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)

    results = {
        "1. Bag-of-Words (BoW Baseline)": [],
        "2. Syllable Embeddings (Action2Vec)": [],
        "3. Variable-Order Markov (PST)": [],
        "4. Explicit-Duration HSMM": [],
        "5. Selective State-Space (Mamba-1)": [],
        "6. Causal Action Transformer": [],
        "7. Transformer (Shuffle Ablation)": []
    }

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_bow_full, groups=groups)):
        fold_num = fold + 1
        log(f"\n>>> Running Fold {fold_num}/5 (Held-out sessions: {len(np.unique(groups[te_idx]))}) <<<")

        Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        js_blind = js_vec(Y_te, p_glob)

        # ---------------------------------------------------------------------
        # 1. Bag-of-Words Linear Ceiling (Benchmark)
        # ---------------------------------------------------------------------
        p_bow = np.clip(Ridge(1.0).fit(X_bow_full[tr_idx], Y_tr).predict(X_bow_full[te_idx]), 1e-4, 1.0)
        p_bow /= np.sum(p_bow, axis=1, keepdims=True)
        gain_bow = ((js_blind - js_vec(Y_te, p_bow)) / js_blind) * 100.0
        results["1. Bag-of-Words (BoW Baseline)"].append(gain_bow)
        log(f"  [Fold {fold_num}] 1. BoW Baseline: +{gain_bow:.2f}%")

        # ---------------------------------------------------------------------
        # 2. Action2Vec (Mikolov SGNS p=32)
        # ---------------------------------------------------------------------
        tr_sess = np.unique(groups[tr_idx])
        tr_seqs = [s["bout_seq"] for s in sessions if s["sess_idx"] in tr_sess]

        a2v_embeds = train_action2vec(
            session_bout_sequences=tr_seqs,
            vocab_size=NUM_SYLLS,
            embedding_dim=32,
            window_size=3,
            epochs=10,
            batch_size=4096,
            device=DEVICE
        )

        # Semantic projection: (N, 100) @ (100, 32) -> (N, 32)
        X_a2v_tr = X_bow_cnt[tr_idx] @ a2v_embeds
        X_a2v_te = X_bow_cnt[te_idx] @ a2v_embeds

        p_a2v = np.clip(Ridge(1.0).fit(X_a2v_tr, Y_tr).predict(X_a2v_te), 1e-4, 1.0)
        p_a2v /= np.sum(p_a2v, axis=1, keepdims=True)
        gain_a2v = ((js_blind - js_vec(Y_te, p_a2v)) / js_blind) * 100.0
        results["2. Syllable Embeddings (Action2Vec)"].append(gain_a2v)
        log(f"  [Fold {fold_num}] 2. Action2Vec: +{gain_a2v:.2f}%")

        # ---------------------------------------------------------------------
        # 3. Variable-Order Markov Model (PST Suffix Representation)
        # ---------------------------------------------------------------------
        pst = ProbabilisticSuffixTree(vocab_size=NUM_SYLLS, max_depth=5, min_count=5, divergence_ratio=1.05)
        pst.fit(tr_seqs)

        # Extract PST conditional predictions as features for 120s forecasting
        p_pst_tr = np.array([pst.predict_distribution(list(X_tok[tr_idx[i], :X_len[tr_idx[i]]])) for i in range(len(tr_idx))], dtype=np.float32)
        p_pst_te = np.array([pst.predict_distribution(list(X_tok[te_idx[i], :X_len[te_idx[i]]])) for i in range(len(te_idx))], dtype=np.float32)

        p_pst = np.clip(Ridge(1.0).fit(p_pst_tr, Y_tr).predict(p_pst_te), 1e-4, 1.0)
        p_pst /= np.sum(p_pst, axis=1, keepdims=True)
        gain_pst = ((js_blind - js_vec(Y_te, p_pst)) / js_blind) * 100.0
        results["3. Variable-Order Markov (PST)"].append(gain_pst)
        log(f"  [Fold {fold_num}] 3. VOMM PST: +{gain_pst:.2f}%")

        # ---------------------------------------------------------------------
        # 4. Explicit-Duration HSMM (Filtered Posterior State Representation)
        # ---------------------------------------------------------------------
        tr_durs = [s["bout_durations"] for s in sessions if s["sess_idx"] in tr_sess]
        hsmm = ExplicitDurationHSMM(num_states=16, vocab_size=NUM_SYLLS, random_state=42)
        hsmm.fit(tr_seqs, tr_durs)

        gamma_tr = np.array([hsmm.filter_state(list(X_tok[tr_idx[i], :X_len[tr_idx[i]]])) for i in range(len(tr_idx))], dtype=np.float32)
        gamma_te = np.array([hsmm.filter_state(list(X_tok[te_idx[i], :X_len[te_idx[i]]])) for i in range(len(te_idx))], dtype=np.float32)

        p_hsmm = np.clip(Ridge(1.0).fit(gamma_tr, Y_tr).predict(gamma_te), 1e-4, 1.0)
        p_hsmm /= np.sum(p_hsmm, axis=1, keepdims=True)
        gain_hsmm = ((js_blind - js_vec(Y_te, p_hsmm)) / js_blind) * 100.0
        results["4. Explicit-Duration HSMM"].append(gain_hsmm)
        log(f"  [Fold {fold_num}] 4. Explicit-Duration HSMM: +{gain_hsmm:.2f}%")

        # ---------------------------------------------------------------------
        # PyTorch Tensors for Deep Sequence Models
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 5. Selective State-Space Model (Mamba-1)
        # ---------------------------------------------------------------------
        torch.cuda.empty_cache()
        mamba = MambaSequenceModel(vocab_size=NUM_SYLLS, d_model=64, d_state=16, num_layers=2).to(DEVICE)
        opt_m = optim.AdamW(mamba.parameters(), lr=0.003, weight_decay=1e-4)

        mamba.train()
        for ep in range(12):
            for b_tok, b_dur, b_mask, b_y in train_loader:
                opt_m.zero_grad()
                pred = mamba(b_tok, b_dur, b_mask)
                loss = js_loss_torch(pred, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
                opt_m.step()

        # Batched evaluation to keep VRAM < 500MB
        p_mamba = predict_batched(mamba, tok_te, dur_te, mask_te, batch_size=128)
        gain_mamba = ((js_blind - js_vec(Y_te, p_mamba)) / js_blind) * 100.0
        results["5. Selective State-Space (Mamba-1)"].append(gain_mamba)
        log(f"  [Fold {fold_num}] 5. Mamba-1 SSM: +{gain_mamba:.2f}%")

        # ---------------------------------------------------------------------
        # 6. Causal Action Transformer
        # ---------------------------------------------------------------------
        torch.cuda.empty_cache()
        xformer = CausalActionTransformer(vocab_size=NUM_SYLLS, d_model=64, n_heads=4, num_layers=2).to(DEVICE)
        opt_x = optim.AdamW(xformer.parameters(), lr=0.003, weight_decay=1e-4)

        xformer.train()
        for ep in range(12):
            for b_tok, b_dur, b_mask, b_y in train_loader:
                opt_x.zero_grad()
                pred = xformer(b_tok, b_dur, b_mask)
                loss = js_loss_torch(pred, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(xformer.parameters(), 1.0)
                opt_x.step()

        # Batched evaluation
        p_xformer = predict_batched(xformer, tok_te, dur_te, mask_te, batch_size=128)
        gain_xformer = ((js_blind - js_vec(Y_te, p_xformer)) / js_blind) * 100.0
        results["6. Causal Action Transformer"].append(gain_xformer)
        log(f"  [Fold {fold_num}] 6. Causal Action Transformer: +{gain_xformer:.2f}%")

        # ---------------------------------------------------------------------
        # 7. Shuffle Ablation Test (Transformer with Shuffled Sequences)
        # ---------------------------------------------------------------------
        tok_te_shuf = tok_te.clone()
        dur_te_shuf = dur_te.clone()
        for i in range(len(tok_te_shuf)):
            L = int(len_te[i].item())
            if L > 1:
                perm = torch.randperm(L, device=DEVICE)
                tok_te_shuf[i, :L] = tok_te[i, perm]
                dur_te_shuf[i, :L] = dur_te[i, perm]

        p_shuf = predict_batched(xformer, tok_te_shuf, dur_te_shuf, mask_te, batch_size=128)
        gain_shuf = ((js_blind - js_vec(Y_te, p_shuf)) / js_blind) * 100.0
        results["7. Transformer (Shuffle Ablation)"].append(gain_shuf)
        log(f"  [Fold {fold_num}] 7. Transformer Shuffled: +{gain_shuf:.2f}%")

        # Send push notification for this fold
        send_ntfy(f"✅ Fold {fold_num}/5 Done! BoW: +{gain_bow:.1f}%, A2V: +{gain_a2v:.1f}%, Mamba: +{gain_mamba:.1f}%, Xformer: +{gain_xformer:.1f}%")

    # -------------------------------------------------------------------------
    # Final Compilation & Summary
    # -------------------------------------------------------------------------
    log("\n" + "=" * 90, notify=True)
    log("FINAL BENCHMARK RESULTS ACROSS ALL 5 FOLDS (W=68.3s -> H=120s)", notify=True)
    log("=" * 90)

    summary_rows = []
    log(f"{'Sequence Modeling Architecture':<42} | {'Predictive Gain (% D_JS Red.)':<30} | {'Status vs BoW Baseline':<20}")
    log("-" * 96)

    bow_mean = np.mean(results["1. Bag-of-Words (BoW Baseline)"])

    for k, vals in results.items():
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
        delta = m - bow_mean
        status = "Baseline" if "BoW" in k else f"{delta:+5.2f}%"
        log(f"{k:<42} | +{m:5.2f}% ± {ci:4.2f}%              | {status:<20}")
        summary_rows.append({
            "Architecture": k,
            "Mean_Predictive_Gain": m,
            "CI_95": ci,
            "Delta_vs_BoW": delta
        })

    log("=" * 90)

    # Save CSV table
    df_res = pd.DataFrame(summary_rows)
    csv_path = os.path.join(TAB_DIR, "sequence_benchmark_results.csv")
    df_res.to_csv(csv_path, index=False)
    log(f"Saved benchmark CSV table: {csv_path}")

    # -------------------------------------------------------------------------
    # Render Publication Figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    archs = [r["Architecture"] for r in summary_rows]
    means = [r["Mean_Predictive_Gain"] for r in summary_rows]
    cis = [r["CI_95"] for r in summary_rows]

    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#BD10E0', '#D0021B', '#7ED321', '#9B9B9B']

    bars = ax.bar(archs, means, yerr=cis, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.axhline(bow_mean, color='#4A90E2', linestyle='--', linewidth=1.8, label=f'Linear BoW Ceiling (+{bow_mean:.2f}%)')

    ax.set_ylabel('Predictive Gain (% $D_{JS}$ Reduction vs. Blind Null)', fontsize=12, fontweight='bold')
    ax.set_title('Canonical Sequence Modeling Suite vs. Linear Bag-of-Words Ceiling\n(30 Sessions, 5-Fold GroupKFold CV, W = 68.3s, H = 120s)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 40)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_xticklabels(archs, rotation=22, ha='right', fontsize=10, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.8),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "sequential_vs_bow_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log(f"Saved publication figure: {fig_path}", notify=True)

    # Final Mobile Push Notification
    top_model = df_res.iloc[df_res['Mean_Predictive_Gain'].argmax()]
    send_ntfy(f"🏆 Sequence Benchmark Complete! Best Model: {top_model['Architecture']} (+{top_model['Mean_Predictive_Gain']:.2f}% ± {top_model['CI_95']:.2f}%)")


if __name__ == "__main__":
    main()
