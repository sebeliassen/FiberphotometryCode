#!/usr/bin/env python3
"""
===============================================================================
2x2 Factorial Cross-Prediction Experiment: Input Representation vs. Target Space
===============================================================================
Tests whether higher predictive gain (+40% in Frame Space vs. +30% in Bout Space)
is driven by the Target Space (Time Allocation vs. Action Frequency) or the Input
Representation (Frame-Weighted vs. Event-Weighted Past History).

Factorial Design:
  1. Condition (Frame In -> Frame Tgt): Time-weighted input -> Future Time Allocation
  2. Condition (Bout In  -> Frame Tgt): Discrete syntax input -> Future Time Allocation
  3. Condition (Frame In -> Bout Tgt):  Time-weighted input -> Future Action Frequency
  4. Condition (Bout In  -> Bout Tgt):  Discrete syntax input -> Future Action Frequency

Horizon: H = 120.0s (2 Minutes lookahead)
Past Window: W = 60.0s
Cross-Validation: Session-Grouped 5-Fold (GroupKFold on session_id).
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository setup
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "02_frame_vs_bout")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "2x2_cross_prediction.log")

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

FPS = 80.025
PAST_WINDOW_SEC = 60.0
PAST_WINDOW_FRAMES = int(PAST_WINDOW_SEC * FPS)
HORIZON_SEC = 120.0
HORIZON_FRAMES = int(HORIZON_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLABLES = 100

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True); Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))


def load_and_preprocess_sessions():
    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    log(f"Loading continuous bout cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    sessions = []
    total_bouts = 0
    t0 = time.time()
    
    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        max_frame = bouts[-1]["end_frame"]
        N = max_frame + 1
        
        frame_sylls = np.zeros(N, dtype=np.int32)
        frame_spatial = np.zeros((N, 6), dtype=np.float32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLABLES - 1)
            frame_sylls[sf : ef + 1] = s_id
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                frame_spatial[sf : ef + 1, :4] = vec[:4]
                frame_spatial[sf : ef + 1, 4] = vec[10]
                frame_spatial[sf : ef + 1, 5] = vec[12]
        
        # 1. Unigram Frame Prefix Sums (Time Allocation)
        oh_sylls = np.zeros((N, NUM_SYLLABLES), dtype=np.float32)
        oh_sylls[np.arange(N), frame_sylls] = 1.0
        cum_frame_sylls = np.vstack([np.zeros((1, NUM_SYLLABLES), dtype=np.float32), np.cumsum(oh_sylls, axis=0)])
        
        # 2. Bigram Frame Indices (N-1)
        bigram_frame_idx = frame_sylls[:-1] * NUM_SYLLABLES + frame_sylls[1:]
        
        # 3. Discrete Bout Prefix Sums (Action Count Frequency)
        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool)
        onsets[0] = True
        onsets[diffs] = True
        
        oh_bouts = np.zeros((N, NUM_SYLLABLES), dtype=np.float32)
        oh_bouts[onsets, frame_sylls[onsets]] = 1.0
        cum_bout_sylls = np.vstack([np.zeros((1, NUM_SYLLABLES), dtype=np.float32), np.cumsum(oh_bouts, axis=0)])
        
        # 4. Spatial Prefix Sums
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])
        
        # 5. Next non-self pointer
        next_bout_ptrs = np.zeros(N, dtype=np.int32)
        cur = 0
        for d in diffs:
            next_bout_ptrs[cur:d] = d
            cur = d
        next_bout_ptrs[cur:] = N
        
        total_bouts += len(bouts)
        sessions.append({
            "session_idx": sess_idx,
            "session_id": f"sess_{sess_idx:02d}",
            "frame_sylls": frame_sylls,
            "cum_frame_sylls": cum_frame_sylls,
            "cum_bout_sylls": cum_bout_sylls,
            "bigram_frame_idx": bigram_frame_idx,
            "cum_spatial": cum_spatial,
            "cum_spatial_sq": cum_spatial_sq,
            "next_bout_ptrs": next_bout_ptrs,
            "n_frames": N
        })
        
    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s ({total_bouts:,} bouts, {sum(s['n_frames'] for s in sessions)/FPS/3600:.2f} hrs).")
    return sessions


def run_2x2_experiment():
    log("=" * 80)
    log("LAUNCHING 2x2 FACTORIAL CROSS-PREDICTION EXPERIMENT (H = 120.0s, W = 60.0s)")
    log("=" * 80)
    
    sessions = load_and_preprocess_sessions()
    w_frames = PAST_WINDOW_FRAMES
    h_frames = HORIZON_FRAMES
    
    conditions = [
        ("frame_input", "frame_target", "1. Frame In -> Frame Tgt (Time Allocation)"),
        ("bout_input",  "frame_target", "2. Bout In  -> Frame Tgt (Cross: Bout In -> Time Tgt)"),
        ("frame_input", "bout_target",  "3. Frame In -> Bout Tgt  (Cross: Frame In -> Count Tgt)"),
        ("bout_input",  "bout_target",  "4. Bout In  -> Bout Tgt  (Action Count Frequency)")
    ]
    
    results = []
    
    for in_type, tgt_type, label in conditions:
        t0 = time.time()
        log("-" * 60)
        log(f"Evaluating: {label}")
        
        all_time_feat = []
        all_spatial_feat = []
        all_curr_onehot = []
        all_curr_syll = []
        all_past_dist = []
        all_past_trans = []
        all_targets = []
        all_sess_idx = []
        
        for s in sessions:
            sess_idx = s["session_idx"]
            N = s["n_frames"]
            if N < w_frames + h_frames + 200: continue
            
            t_anchors = np.arange(w_frames, N - h_frames, STRIDE_FRAMES)
            t_targets_start = s["next_bout_ptrs"][t_anchors]
            valid_mask = (t_targets_start + h_frames < N)
            t_anchors = t_anchors[valid_mask]
            t_targets_start = t_targets_start[valid_mask]
            M = len(t_anchors)
            if M == 0: continue
            
            curr_syll = s["frame_sylls"][t_anchors]
            
            # 1. Time Features
            t_norm = t_anchors / float(N)
            time_feat = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
            
            # 2. Spatial Features via O(1) Prefix Sum
            sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - w_frames]
            sp_mean = sp_sum / float(w_frames)
            sp_sq_sum = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - w_frames]
            sp_var = np.maximum(sp_sq_sum / float(w_frames) - sp_mean**2, 1e-6)
            sp_std = np.sqrt(sp_var)
            spatial_feat = np.hstack([sp_mean, sp_std]).astype(np.float32)
            
            # 3. Past Unigram Distribution
            if in_type == "frame_input":
                p_counts = s["cum_frame_sylls"][t_anchors] - s["cum_frame_sylls"][t_anchors - w_frames]
            else:
                p_counts = s["cum_bout_sylls"][t_anchors] - s["cum_bout_sylls"][t_anchors - w_frames]
            past_dist = (p_counts + 1e-4) / (np.sum(p_counts, axis=1, keepdims=True) + 1e-4 * NUM_SYLLABLES)
            
            # 4. Past Transition Matrix
            bigrams = s["bigram_frame_idx"]
            trans_prob = np.zeros((M, NUM_SYLLABLES * NUM_SYLLABLES), dtype=np.float32)
            for i, t in enumerate(t_anchors):
                seg = bigrams[t - w_frames : t - 1]
                b_counts = np.bincount(seg, minlength=NUM_SYLLABLES * NUM_SYLLABLES).reshape(NUM_SYLLABLES, NUM_SYLLABLES)
                row_sums = b_counts.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                trans_prob[i] = ((b_counts + 1e-4) / (row_sums + 1e-4 * NUM_SYLLABLES)).flatten()
            
            # 5. Current One-Hot
            curr_oh = np.zeros((M, NUM_SYLLABLES), dtype=np.float32)
            curr_oh[np.arange(M), curr_syll] = 1.0
            
            # 6. Future Action Budget Target
            if tgt_type == "frame_target":
                f_counts = s["cum_frame_sylls"][t_targets_start + h_frames] - s["cum_frame_sylls"][t_targets_start]
            else:
                f_counts = s["cum_bout_sylls"][t_targets_start + h_frames] - s["cum_bout_sylls"][t_targets_start]
            target = (f_counts + 1e-4) / (np.sum(f_counts, axis=1, keepdims=True) + 1e-4 * NUM_SYLLABLES)
            
            all_time_feat.append(time_feat)
            all_spatial_feat.append(spatial_feat)
            all_curr_onehot.append(curr_oh)
            all_curr_syll.append(curr_syll)
            all_past_dist.append(past_dist)
            all_past_trans.append(trans_prob)
            all_targets.append(target)
            all_sess_idx.append(np.full(M, sess_idx, dtype=int))
            
        X_time = np.vstack(all_time_feat)
        X_spatial = np.vstack(all_spatial_feat)
        X_curr_oh = np.vstack(all_curr_onehot)
        y_curr_syll = np.concatenate(all_curr_syll)
        X_past_dist = np.vstack(all_past_dist)
        X_past_trans = np.vstack(all_past_trans)
        Y_targets = np.vstack(all_targets)
        groups = np.concatenate(all_sess_idx)
        
        # SVD + KMeans K=16
        svd = TruncatedSVD(n_components=24, random_state=42)
        latent_24d = svd.fit_transform(X_past_trans)
        latent_norm = normalize(latent_24d, norm="l2")
        strat_k16 = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(latent_norm)
        strat_k16_oh = np.zeros((len(strat_k16), 16), dtype=np.float32)
        strat_k16_oh[np.arange(len(strat_k16)), strat_k16] = 1.0
        
        # 5-Fold GroupKFold Cross-Validation
        gkf = GroupKFold(n_splits=5)
        fold_g_markov, fold_g_p1, fold_g_p2, fold_g_p3, fold_g_p4, fold_g_p5 = [], [], [], [], [], []
        
        for tr_idx, te_idx in gkf.split(X_time, groups=groups):
            Y_tr, Y_te = Y_targets[tr_idx], Y_targets[te_idx]
            
            p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
            p_glob /= np.sum(p_glob, axis=1, keepdims=True)
            js_blind = js_vec(Y_te, p_glob)
            
            # Standalone Markov S_t
            c_tr, c_te = y_curr_syll[tr_idx], y_curr_syll[te_idx]
            sm = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES), dtype=np.float32)
            for s_id in range(NUM_SYLLABLES):
                mask = (c_tr == s_id)
                sm[s_id] = np.mean(Y_tr[mask], axis=0) if np.sum(mask) > 0 else p_glob[0]
                sm[s_id] /= np.sum(sm[s_id])
            p_m = sm[c_te]
            fold_g_markov.append(((js_blind - js_vec(Y_te, p_m)) / js_blind) * 100.0)
            
            # P2: Macro Context
            X_tr_p2 = np.hstack([X_time[tr_idx], X_spatial[tr_idx]])
            X_te_p2 = np.hstack([X_time[te_idx], X_spatial[te_idx]])
            p_p2 = np.clip(Ridge(10.0).fit(X_tr_p2, Y_tr).predict(X_te_p2), 1e-4, 1.0)
            p_p2 /= np.sum(p_p2, axis=1, keepdims=True)
            fold_g_p2.append(((js_blind - js_vec(Y_te, p_p2)) / js_blind) * 100.0)
            
            # P3: P2 + Posture
            X_tr_p3 = np.hstack([X_tr_p2, X_curr_oh[tr_idx]])
            X_te_p3 = np.hstack([X_te_p2, X_curr_oh[te_idx]])
            
            # P4: Strategy K=16
            X_tr_p4 = np.hstack([X_tr_p3, strat_k16_oh[tr_idx]])
            X_te_p4 = np.hstack([X_te_p3, strat_k16_oh[te_idx]])
            p_p4 = np.clip(Ridge(5.0).fit(X_tr_p4, Y_tr).predict(X_te_p4), 1e-4, 1.0)
            p_p4 /= np.sum(p_p4, axis=1, keepdims=True)
            fold_g_p4.append(((js_blind - js_vec(Y_te, p_p4)) / js_blind) * 100.0)
            
            # P5: Empirical Bound P_max
            X_tr_p5 = np.hstack([X_tr_p3, X_past_dist[tr_idx], latent_24d[tr_idx]])
            X_te_p5 = np.hstack([X_te_p3, X_past_dist[te_idx], latent_24d[te_idx]])
            p_p5 = np.clip(Ridge(1.0).fit(X_tr_p5, Y_tr).predict(X_te_p5), 1e-4, 1.0)
            p_p5 /= np.sum(p_p5, axis=1, keepdims=True)
            fold_g_p5.append(((js_blind - js_vec(Y_te, p_p5)) / js_blind) * 100.0)
            
        m_markov = float(np.mean(fold_g_markov))
        ci_markov = float(1.96 * np.std(fold_g_markov) / np.sqrt(5))
        m_p2 = float(np.mean(fold_g_p2))
        ci_p2 = float(1.96 * np.std(fold_g_p2) / np.sqrt(5))
        m_p4 = float(np.mean(fold_g_p4))
        ci_p4 = float(1.96 * np.std(fold_g_p4) / np.sqrt(5))
        m_p5 = float(np.mean(fold_g_p5))
        ci_p5 = float(1.96 * np.std(fold_g_p5) / np.sqrt(5))
        ret_pct = (m_p4 / m_p5) * 100.0 if m_p5 > 0 else 0.0
        elapsed = time.time() - t0
        
        log(f"-> Markov(S_t): +{m_markov:5.2f}% | Macro(P2): +{m_p2:5.2f}% | Strat(K=16): +{m_p4:5.2f}% | P_max: +{m_p5:5.2f}% (Ret: {ret_pct:4.1f}%) | [{elapsed:.2f}s]")
        
        results.append({
            "condition": label,
            "input_type": in_type,
            "target_type": tgt_type,
            "gain_markov_mean": m_markov,
            "gain_markov_ci": ci_markov,
            "gain_p2_macro_mean": m_p2,
            "gain_p2_macro_ci": ci_p2,
            "gain_p4_strat16_mean": m_p4,
            "gain_p4_strat16_ci": ci_p4,
            "gain_p5_pmax_mean": m_p5,
            "gain_p5_pmax_ci": ci_p5,
            "retention_k16_pct": ret_pct
        })
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(TAB_DIR, "2x2_cross_prediction_results.csv")
    df_results.to_csv(csv_path, index=False)
    log(f"\nSaved 2x2 Factorial Results to: {csv_path}")
    
    # -------------------------------------------------------------------------
    # RENDER 2x2 PUBLICATION BAR CHART / HEATMAP
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    
    labels = ["1. Frame In → Frame Tgt\n(Time Allocation)", 
              "2. Bout In → Frame Tgt\n(Syntax In → Time Tgt)",
              "3. Frame In → Bout Tgt\n(Time In → Event Tgt)",
              "4. Bout In → Bout Tgt\n(Action Frequency)"]
    
    pmax_vals = df_results["gain_p5_pmax_mean"].values
    pmax_cis = df_results["gain_p5_pmax_ci"].values
    strat_vals = df_results["gain_p4_strat16_mean"].values
    strat_cis = df_results["gain_p4_strat16_ci"].values
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, pmax_vals, width, yerr=pmax_cis, capsize=4, label='$\\mathcal{P}_{\\text{max}}$ Ceiling', color='#e7298a', alpha=0.85)
    rects2 = ax1.bar(x + width/2, strat_vals, width, yerr=strat_cis, capsize=4, label='Strategy $K=16$ ($\\mathcal{P}_4$)', color='#1f77b4', alpha=0.85)
    
    ax1.set_ylabel('Predictive Gain at $H=120$s (% $D_{\\text{JS}}$ Red.)', fontsize=11, fontweight='bold')
    ax1.set_title('A. 2x2 Factorial Test: Input vs. Target Dynamics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.2, fontweight='bold')
    ax1.set_ylim(0, 48)
    ax1.legend(frameon=True, fontsize=9.2, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.2, f'+{h:.1f}%', ha='center', va='bottom', fontsize=8.8, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.2, f'+{h:.1f}%', ha='center', va='bottom', fontsize=8.8, fontweight='bold')
        
    frame_tgt_mean = df_results.loc[df_results["target_type"]=="frame_target", "gain_p5_pmax_mean"].mean()
    bout_tgt_mean = df_results.loc[df_results["target_type"]=="bout_target", "gain_p5_pmax_mean"].mean()
    
    ax2.bar(["Target: Time Allocation\n(Seconds per Syllable)", "Target: Action Frequency\n(Discrete Bout Count)"], 
            [frame_tgt_mean, bout_tgt_mean], color=['#e7298a', '#1f77b4'], width=0.5, alpha=0.85)
    
    ax2.set_ylabel('Average $\\mathcal{P}_{\\text{max}}$ Ceiling (% Gain)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Target Space Invariance: Target Determines Gain', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 48)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    ax2.text(0, frame_tgt_mean + 1.2, f'+{frame_tgt_mean:.1f}%\n(Time Budget)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.text(1, bout_tgt_mean + 1.2, f'+{bout_tgt_mean:.1f}%\n(Event Budget)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "2x2_cross_prediction_matrix.png")
    fig.savefig(fig_path)
    plt.close(fig)
    log(f"Saved 2x2 Factorial Figure to: {fig_path}")
    
    log("=" * 80)
    log("2x2 EXPERIMENT COMPLETED SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    run_2x2_experiment()
