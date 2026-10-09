#!/usr/bin/env python3
"""
===============================================================================
Duration-Augmented RLE Bout Experiment: Bag-of-Words vs. Sequence Modeling
===============================================================================
Tests whether augmenting discrete RLE bouts with their continuous duration length
(mean duration per syllable or total dwell time) improves predictive power over
pure discrete bout counts or pure frame time budgets.

Evaluations at W = 68.3s -> Future Bouts H = 120.0s:
  1. Pure Bout Counts Alone (How many times each action was initiated)
  2. Pure Frame Time Alone (Total seconds spent in each action)
  3. Hybrid: Bout Counts + Mean Bout Durations
  4. Hybrid: Bout Counts + Total Frame Dwell Times
  5. Full Ceiling Hybrid: Macro + Posture + Counts + Durations + SVD Transitions
===============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "02_frame_vs_bout")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "duration_augmented_rle.log")

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True); Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))


def main():
    log("=" * 80)
    log("LAUNCHING DURATION-AUGMENTED RLE EXPERIMENT (W = 68.3s -> Future Bouts H = 120s)")
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
                
        # 1. Unigram Frame Prefix Sums
        oh_frame = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        oh_frame[np.arange(N), frame_sylls] = 1.0
        cum_frame = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_frame, axis=0)])
        
        # 2. Discrete Bout Prefix Sums (onsets only)
        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool)
        onsets[0] = True; onsets[diffs] = True
        
        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        oh_bout[onsets, frame_sylls[onsets]] = 1.0
        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
        
        # 3. Next non-self pointer
        next_bout_ptrs = np.zeros(N, dtype=np.int32)
        cur = 0
        for d in diffs:
            next_bout_ptrs[cur:d] = d
            cur = d
        next_bout_ptrs[cur:] = N
        
        # 4. Spatial prefix sums
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])
        
        # 5. Fast onset indices and bout sequence for binary search
        bout_seq = frame_sylls[onsets]
        bout_onsets = np.where(onsets)[0]
        
        sessions.append({
            "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
            "cum_frame": cum_frame, "cum_bout": cum_bout,
            "next_bout_ptrs": next_bout_ptrs,
            "cum_spatial": cum_spatial, "cum_spatial_sq": cum_spatial_sq,
            "bout_seq": bout_seq, "bout_onsets": bout_onsets
        })
        
    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s.")
    
    # Extract Features
    all_time, all_sp, all_curr_oh = [], [], []
    all_pf_dist, all_pb_dist, all_p_mean_dur = [], [], []
    all_tb = []
    all_tgt, all_sess = [], []
    
    t_feat = time.time()
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
        
        # 1. Past Frame Unigrams (Time Spent)
        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        pf_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        # 2. Past Bout Unigrams (Event Counts)
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        pb_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        # 3. Mean Bout Duration per Syllable (Time Spent / Event Count) / FPS
        mean_dur = np.zeros_like(pf_cnt)
        nz_mask = (pb_cnt > 0)
        mean_dur[nz_mask] = (pf_cnt[nz_mask] / pb_cnt[nz_mask]) / FPS
        
        # Bout Transitions
        b_onsets = s["bout_onsets"]
        b_seq = s["bout_seq"]
        trans_b = np.zeros((M, NUM_SYLLS * NUM_SYLLS), dtype=np.float32)
        i_starts = np.searchsorted(b_onsets, t_anchors - W_FRAMES)
        i_ends = np.searchsorted(b_onsets, t_anchors)
        for i in range(M):
            i0, i1 = i_starts[i], i_ends[i]
            if i1 - i0 > 1:
                sub_b = b_seq[i0:i1]
                sub_bigrams = sub_b[:-1] * NUM_SYLLS + sub_b[1:]
                b_cnt = np.bincount(sub_bigrams, minlength=NUM_SYLLS * NUM_SYLLS).reshape(NUM_SYLLS, NUM_SYLLS)
                rs = b_cnt.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
                trans_b[i] = ((b_cnt + 1e-4) / (rs + 1e-4 * NUM_SYLLS)).flatten()
            else:
                trans_b[i] = 1.0 / NUM_SYLLS
                
        curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
        curr_oh[np.arange(M), curr_syll] = 1.0
        
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        all_time.append(time_feat); all_sp.append(spatial_feat)
        all_curr_oh.append(curr_oh); all_pf_dist.append(pf_dist)
        all_pb_dist.append(pb_dist); all_p_mean_dur.append(mean_dur)
        all_tb.append(trans_b); all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    X_time = np.vstack(all_time); X_spatial = np.vstack(all_sp)
    X_curr_oh = np.vstack(all_curr_oh)
    X_pf = np.vstack(all_pf_dist); X_pb = np.vstack(all_pb_dist)
    X_dur = np.vstack(all_p_mean_dur); X_tb = np.vstack(all_tb)
    Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

    log(f"Extracted all features for {len(Y_tgt):,} anchor samples in {time.time()-t_feat:.2f}s.")

    # SVD on Bout Transitions
    svd_b = TruncatedSVD(n_components=24, random_state=42)
    latent_b = svd_b.fit_transform(X_tb)

    # 5-Fold Cross-Validation
    gkf = GroupKFold(n_splits=5)
    res = {
        "1. Pure Bout Counts Alone": [],
        "2. Pure Frame Time Alone": [],
        "3. Hybrid: Bout Counts + Mean Durations": [],
        "4. Hybrid: Bout Counts + Total Frame Time": [],
        "5. Full Ceiling Hybrid (P_max)": []
    }

    for tr_idx, te_idx in gkf.split(X_time, groups=groups):
        Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        js_blind = js_vec(Y_te, p_glob)
        
        X_tr_p3 = np.hstack([X_time[tr_idx], X_spatial[tr_idx], X_curr_oh[tr_idx]])
        X_te_p3 = np.hstack([X_time[te_idx], X_spatial[te_idx], X_curr_oh[te_idx]])
        
        # 1. Pure Bout Counts Alone
        p1 = np.clip(Ridge(1.0).fit(X_pb[tr_idx], Y_tr).predict(X_pb[te_idx]), 1e-4, 1.0)
        p1 /= np.sum(p1, axis=1, keepdims=True)
        res["1. Pure Bout Counts Alone"].append(((js_blind - js_vec(Y_te, p1)) / js_blind) * 100.0)
        
        # 2. Pure Frame Time Alone
        p2 = np.clip(Ridge(1.0).fit(X_pf[tr_idx], Y_tr).predict(X_pf[te_idx]), 1e-4, 1.0)
        p2 /= np.sum(p2, axis=1, keepdims=True)
        res["2. Pure Frame Time Alone"].append(((js_blind - js_vec(Y_te, p2)) / js_blind) * 100.0)
        
        # 3. Bout Count + Mean Duration
        X_tr_bd = np.hstack([X_pb[tr_idx], X_dur[tr_idx]])
        X_te_bd = np.hstack([X_pb[te_idx], X_dur[te_idx]])
        p3 = np.clip(Ridge(1.0).fit(X_tr_bd, Y_tr).predict(X_te_bd), 1e-4, 1.0)
        p3 /= np.sum(p3, axis=1, keepdims=True)
        res["3. Hybrid: Bout Counts + Mean Durations"].append(((js_blind - js_vec(Y_te, p3)) / js_blind) * 100.0)
        
        # 4. Bout Count + Frame Time
        X_tr_bf = np.hstack([X_pb[tr_idx], X_pf[tr_idx]])
        X_te_bf = np.hstack([X_pb[te_idx], X_pf[te_idx]])
        p4 = np.clip(Ridge(1.0).fit(X_tr_bf, Y_tr).predict(X_te_bf), 1e-4, 1.0)
        p4 /= np.sum(p4, axis=1, keepdims=True)
        res["4. Hybrid: Bout Counts + Total Frame Time"].append(((js_blind - js_vec(Y_te, p4)) / js_blind) * 100.0)
        
        # 5. Full Hybrid Ceiling: P3 + Bout Unigrams + Frame Time + SVD Transitions
        X_tr_full = np.hstack([X_tr_p3, X_pb[tr_idx], X_pf[tr_idx], latent_b[tr_idx]])
        X_te_full = np.hstack([X_te_p3, X_pb[te_idx], X_pf[te_idx], latent_b[te_idx]])
        p5 = np.clip(Ridge(1.0).fit(X_tr_full, Y_tr).predict(X_te_full), 1e-4, 1.0)
        p5 /= np.sum(p5, axis=1, keepdims=True)
        res["5. Full Ceiling Hybrid (P_max)"].append(((js_blind - js_vec(Y_te, p5)) / js_blind) * 100.0)

    log("\n" + "=" * 90)
    log("DURATION-AUGMENTED RLE EXPERIMENT RESULTS (W = 68.3s -> Future Bouts H = 120s)")
    log("=" * 90)
    log(f"{'Input Representation':<44} | {'Predictive Gain (% D_JS Red.)':<30}")
    log("-" * 90)
    for k, vals in res.items():
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(5)
        log(f"{k:<44} | +{m:5.2f}% ± {ci:4.2f}%")
    log("=" * 90)


if __name__ == "__main__":
    main()
