#!/usr/bin/env python3
"""
===============================================================================
Chapter 02 Benchmark: Raw Video Frames vs. Discrete RLE Bouts
===============================================================================
Methodologically demonstrates why evaluating behavioral forecasting on raw video frames
introduces an artificial Markovian correlation at short horizons, and how Run-Length
Encoded (RLE) discrete bouts eliminate this intra-action artifact to reveal true
non-Markovian strategy dynamics.

Streams Evaluated:
  1. Stream A (Raw Video Frames at Native 80.025 Hz):
     - Target window starts at frame t (includes ongoing syllable remainder).
     - Causes Markov S_t to be artificially inflated at short horizons.
  2. Stream B (Discrete RLE Bouts / Non-Self-Transition Ground Truth):
     - Target window strictly starts at the next non-self state transition (t_next, where S_{t_next} != S_t).
     - Isolates true sequential predictability with zero intra-bout overlap.

Cross-Validation: Session-Grouped 5-Fold (GroupKFold on session_id, Zero Leakage).
Horizons Evaluated: H in [2s, 5s, 15s, 30s, 60s, 120s].
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

LOG_FILE = os.path.join(OUT_DIR, "frame_vs_bout_benchmark.log")

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

FPS = 80.025
PAST_WINDOW_SEC = 60.0
PAST_WINDOW_FRAMES = int(PAST_WINDOW_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)

HORIZONS_SEC = [2.0, 5.0, 15.0, 30.0, 60.0, 120.0]
NUM_SYLLABLES = 100

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True)
    Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))


def load_and_preprocess_sessions():
    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    log(f"Loading continuous bout cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    sessions = []
    all_durations_ms = []
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
            dur_frames = ef - sf + 1
            all_durations_ms.append((dur_frames / FPS) * 1000.0)
            
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                frame_spatial[sf : ef + 1, :4] = vec[:4]
                frame_spatial[sf : ef + 1, 4] = vec[10]
                frame_spatial[sf : ef + 1, 5] = vec[12]
        
        # 1. Unigram Prefix Sums (N+1, 100)
        oh_sylls = np.zeros((N, NUM_SYLLABLES), dtype=np.float32)
        oh_sylls[np.arange(N), frame_sylls] = 1.0
        cum_sylls = np.vstack([np.zeros((1, NUM_SYLLABLES), dtype=np.float32), np.cumsum(oh_sylls, axis=0)])
        
        # 2. Bigram Indices (N-1)
        bigram_idx = frame_sylls[:-1] * NUM_SYLLABLES + frame_sylls[1:]
        
        # 3. Spatial Prefix Sums (N+1, 6)
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])
        
        # 4. Next non-self transition start pointers
        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
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
            "cum_sylls": cum_sylls,
            "bigram_idx": bigram_idx,
            "cum_spatial": cum_spatial,
            "cum_spatial_sq": cum_spatial_sq,
            "next_bout_ptrs": next_bout_ptrs,
            "n_frames": N
        })
        
    dur_arr = np.array(all_durations_ms)
    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s ({total_bouts:,} bouts, {sum(s['n_frames'] for s in sessions)/FPS/3600:.2f} hrs).")
    log(f"Empirical Syllable Duration Stats: Mean={np.mean(dur_arr):.1f}ms, Median={np.median(dur_arr):.1f}ms, IQR=[{np.percentile(dur_arr, 25):.1f}ms - {np.percentile(dur_arr, 75):.1f}ms], 90th%={np.percentile(dur_arr, 90):.1f}ms")
    return sessions, dur_arr


def evaluate_stream(sessions, stream_type: str = "bout"):
    log("=" * 80)
    log(f"EVALUATING STREAM: {stream_type.upper()} PROTOCOL (W = {PAST_WINDOW_SEC}s, 6 HORIZONS)")
    log("=" * 80)
    
    w_frames = PAST_WINDOW_FRAMES
    results = []
    
    for h_idx, h_sec in enumerate(HORIZONS_SEC):
        t0 = time.time()
        h_frames = int(h_sec * FPS)
        
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
            
            if stream_type == "frame":
                t_targets_start = t_anchors
                valid_mask = (t_targets_start + h_frames < N)
            else:
                t_targets_start = s["next_bout_ptrs"][t_anchors]
                valid_mask = (t_targets_start + h_frames < N)
                
            t_anchors = t_anchors[valid_mask]
            t_targets_start = t_targets_start[valid_mask]
            M = len(t_anchors)
            if M == 0: continue
            
            curr_syll = s["frame_sylls"][t_anchors]
            
            t_norm = t_anchors / float(N)
            time_feat = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
            
            sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - w_frames]
            sp_mean = sp_sum / float(w_frames)
            sp_sq_sum = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - w_frames]
            sp_var = np.maximum(sp_sq_sum / float(w_frames) - sp_mean**2, 1e-6)
            sp_std = np.sqrt(sp_var)
            spatial_feat = np.hstack([sp_mean, sp_std]).astype(np.float32)
            
            p_counts = s["cum_sylls"][t_anchors] - s["cum_sylls"][t_anchors - w_frames]
            past_dist = (p_counts + 1e-4) / (np.sum(p_counts, axis=1, keepdims=True) + 1e-4 * NUM_SYLLABLES)
            
            bigrams = s["bigram_idx"]
            trans_prob = np.zeros((M, NUM_SYLLABLES * NUM_SYLLABLES), dtype=np.float32)
            for i, t in enumerate(t_anchors):
                seg = bigrams[t - w_frames : t - 1]
                b_counts = np.bincount(seg, minlength=NUM_SYLLABLES * NUM_SYLLABLES).reshape(NUM_SYLLABLES, NUM_SYLLABLES)
                row_sums = b_counts.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                trans_prob[i] = ((b_counts + 1e-4) / (row_sums + 1e-4 * NUM_SYLLABLES)).flatten()
            
            curr_oh = np.zeros((M, NUM_SYLLABLES), dtype=np.float32)
            curr_oh[np.arange(M), curr_syll] = 1.0
            
            f_counts = s["cum_sylls"][t_targets_start + h_frames] - s["cum_sylls"][t_targets_start]
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
        
        svd = TruncatedSVD(n_components=24, random_state=42)
        latent_24d = svd.fit_transform(X_past_trans)
        latent_norm = normalize(latent_24d, norm="l2")
        strat_k16 = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(latent_norm)
        strat_k16_oh = np.zeros((len(strat_k16), 16), dtype=np.float32)
        strat_k16_oh[np.arange(len(strat_k16)), strat_k16] = 1.0
        
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
            
            # P1: Time
            p_p1 = np.clip(Ridge(10.0).fit(X_time[tr_idx], Y_tr).predict(X_time[te_idx]), 1e-4, 1.0)
            p_p1 /= np.sum(p_p1, axis=1, keepdims=True)
            fold_g_p1.append(((js_blind - js_vec(Y_te, p_p1)) / js_blind) * 100.0)
            
            # P2: Macro Context
            X_tr_p2 = np.hstack([X_time[tr_idx], X_spatial[tr_idx]])
            X_te_p2 = np.hstack([X_time[te_idx], X_spatial[te_idx]])
            p_p2 = np.clip(Ridge(10.0).fit(X_tr_p2, Y_tr).predict(X_te_p2), 1e-4, 1.0)
            p_p2 /= np.sum(p_p2, axis=1, keepdims=True)
            fold_g_p2.append(((js_blind - js_vec(Y_te, p_p2)) / js_blind) * 100.0)
            
            # P3: P2 + Posture
            X_tr_p3 = np.hstack([X_tr_p2, X_curr_oh[tr_idx]])
            X_te_p3 = np.hstack([X_te_p2, X_curr_oh[te_idx]])
            p_p3 = np.clip(Ridge(10.0).fit(X_tr_p3, Y_tr).predict(X_te_p3), 1e-4, 1.0)
            p_p3 /= np.sum(p_p3, axis=1, keepdims=True)
            fold_g_p3.append(((js_blind - js_vec(Y_te, p_p3)) / js_blind) * 100.0)
            
            # P4: Strategy K=16
            X_tr_p4 = np.hstack([X_tr_p3, strat_k16_oh[tr_idx]])
            X_te_p4 = np.hstack([X_te_p3, strat_k16_oh[te_idx]])
            p_p4 = np.clip(Ridge(5.0).fit(X_tr_p4, Y_tr).predict(X_te_p4), 1e-4, 1.0)
            p_p4 /= np.sum(p_p4, axis=1, keepdims=True)
            fold_g_p4.append(((js_blind - js_vec(Y_te, p_p4)) / js_blind) * 100.0)
            
            # P5: Empirical Past Bound P_max
            X_tr_p5 = np.hstack([X_tr_p3, X_past_dist[tr_idx], latent_24d[tr_idx]])
            X_te_p5 = np.hstack([X_te_p3, X_past_dist[te_idx], latent_24d[te_idx]])
            p_p5 = np.clip(Ridge(1.0).fit(X_tr_p5, Y_tr).predict(X_te_p5), 1e-4, 1.0)
            p_p5 /= np.sum(p_p5, axis=1, keepdims=True)
            fold_g_p5.append(((js_blind - js_vec(Y_te, p_p5)) / js_blind) * 100.0)
            
        m_markov = float(np.mean(fold_g_markov))
        ci_markov = float(1.96 * np.std(fold_g_markov) / np.sqrt(5))
        m_p2 = float(np.mean(fold_g_p2))
        ci_p2 = float(1.96 * np.std(fold_g_p2) / np.sqrt(5))
        m_p3 = float(np.mean(fold_g_p3))
        m_p4 = float(np.mean(fold_g_p4))
        ci_p4 = float(1.96 * np.std(fold_g_p4) / np.sqrt(5))
        m_p5 = float(np.mean(fold_g_p5))
        ci_p5 = float(1.96 * np.std(fold_g_p5) / np.sqrt(5))
        
        ret_pct = (m_p4 / m_p5) * 100.0 if m_p5 > 0 else 0.0
        elapsed = time.time() - t0
        
        log(f"[{stream_type.upper()}] H = {h_sec:5.1f}s | Markov(S_t): +{m_markov:5.2f}% | Strat(K=16): +{m_p4:5.2f}% | P_max: +{m_p5:5.2f}% (Ret: {ret_pct:4.1f}%) | [{elapsed:.2f}s]")
        
        results.append({
            "stream_type": stream_type,
            "horizon_sec": h_sec,
            "gain_markov_mean": m_markov,
            "gain_markov_ci": ci_markov,
            "gain_p2_macro_mean": m_p2,
            "gain_p2_macro_ci": ci_p2,
            "gain_p3_posture_mean": m_p3,
            "gain_p4_strat16_mean": m_p4,
            "gain_p4_strat16_ci": ci_p4,
            "gain_p5_pmax_mean": m_p5,
            "gain_p5_pmax_ci": ci_p5,
            "retention_k16_pct": ret_pct,
            "delta_macro": m_p2,
            "delta_markov": m_p3 - m_p2,
            "delta_syllable": m_p4 - m_p3,
            "delta_unresolved": m_p5 - m_p4
        })
        
    return pd.DataFrame(results)


def render_figures_and_reports(df_all, dur_arr):
    df_frame = df_all[df_all["stream_type"] == "frame"].copy()
    df_bout = df_all[df_all["stream_type"] == "bout"].copy()
    
    # -------------------------------------------------------------------------
    # RENDER 2-PANEL PUBLICATION COMPARISON FIGURE WITH PROPER Y-LIMITS [0, 55]
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=300)
    
    h_vals = df_frame["horizon_sec"].values
    
    # Panel 1: Raw Video Frame Stream (Methodological Artifact)
    ax1.errorbar(h_vals, df_frame["gain_markov_mean"], yerr=df_frame["gain_markov_ci"], 
                 fmt='o--', color='#7570b3', lw=2.4, ms=6.5, capsize=3.5, label='Markov Baseline ($S_t$) [Intra-Bout Inflated]')
    ax1.errorbar(h_vals, df_frame["gain_p4_strat16_mean"], yerr=df_frame["gain_p4_strat16_ci"], 
                 fmt='s-', color='#1f77b4', lw=2.4, ms=6.5, capsize=3.5, label='Strategy $K=16$ ($\\mathcal{P}_4$)')
    ax1.errorbar(h_vals, df_frame["gain_p5_pmax_mean"], yerr=df_frame["gain_p5_pmax_ci"], 
                 fmt='^-', color='#e7298a', lw=2.2, ms=6.5, capsize=3.5, label='Empirical Bound ($\\mathcal{P}_{\\text{max}}$)')
    ax1.errorbar(h_vals, df_frame["gain_p2_macro_mean"], yerr=df_frame["gain_p2_macro_ci"], 
                 fmt='d:', color='#d95f02', lw=1.8, ms=5.5, capsize=3, label='Macro Context ($\\mathcal{P}_2$)')
    
    # Highlight Artifact
    ax1.annotate('Intra-Bout Overlap Artifact\n(+48.6% @ 2s -> drops to +24.2%)', 
                 xy=(2.0, df_frame.loc[df_frame["horizon_sec"]==2.0, "gain_markov_mean"].values[0]),
                 xytext=(4.0, 48.0),
                 arrowprops=dict(facecolor='#7570b3', shrink=0.08, width=1.2, headwidth=6),
                 fontsize=9.2, fontweight='bold', bbox=dict(boxstyle="round,pad=0.35", fc="#f0f0f8", ec="#7570b3", lw=1.3))
    
    ax1.set_xscale('log')
    ax1.set_xticks(h_vals)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel('Future Horizon $H$ (Seconds, Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predictive Gain (% $D_{\\text{JS}}$ Reduction)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Raw Video Frame Stream (80Hz Native)\nFalse Markovian Decay from Ongoing Action Overlap', fontsize=11.5, fontweight='bold')
    ax1.set_ylim(0, 56)
    ax1.legend(frameon=True, fontsize=8.8, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Panel 2: Discrete Bout Stream (Ground Truth)
    ax2.errorbar(h_vals, df_bout["gain_markov_mean"], yerr=df_bout["gain_markov_ci"], 
                 fmt='o--', color='#7570b3', lw=2.4, ms=6.5, capsize=3.5, label='Markov Baseline ($S_t$) [True Floor]')
    ax2.errorbar(h_vals, df_bout["gain_p4_strat16_mean"], yerr=df_bout["gain_p4_strat16_ci"], 
                 fmt='s-', color='#1f77b4', lw=2.4, ms=6.5, capsize=3.5, label='Strategy $K=16$ ($\\mathcal{P}_4$)')
    ax2.errorbar(h_vals, df_bout["gain_p5_pmax_mean"], yerr=df_bout["gain_p5_pmax_ci"], 
                 fmt='^-', color='#e7298a', lw=2.2, ms=6.5, capsize=3.5, label='Empirical Bound ($\\mathcal{P}_{\\text{max}}$)')
    ax2.errorbar(h_vals, df_bout["gain_p2_macro_mean"], yerr=df_bout["gain_p2_macro_ci"], 
                 fmt='d:', color='#d95f02', lw=1.8, ms=5.5, capsize=3, label='Macro Context ($\\mathcal{P}_2$)')
    
    # Highlight True Strategy Emergence
    ax2.annotate('True Markov Floor (+12.5% -> +15.3%)\nStrategy Dominates (+13.2% -> +25.6%)', 
                 xy=(60.0, df_bout.loc[df_bout["horizon_sec"]==60.0, "gain_p4_strat16_mean"].values[0]),
                 xytext=(5.0, 38.0),
                 arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.2, headwidth=6),
                 fontsize=9.2, fontweight='bold', bbox=dict(boxstyle="round,pad=0.35", fc="#e8f4f8", ec="#1f77b4", lw=1.3))
    
    ax2.set_xscale('log')
    ax2.set_xticks(h_vals)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('Future Horizon $H$ (Seconds, Log Scale)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predictive Gain (% $D_{\\text{JS}}$ Reduction)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Discrete Bout Stream (RLE State Switches)\nTrue Invariant Markov Baseline & Non-Markovian Strategy Rise', fontsize=11.5, fontweight='bold')
    ax2.set_ylim(0, 56)
    ax2.legend(frameon=True, fontsize=8.8, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "frame_vs_bout_comparison.png")
    fig_pdf = os.path.join(FIG_DIR, "frame_vs_bout_comparison.pdf")
    fig.savefig(fig_path)
    fig.savefig(fig_pdf)
    plt.close(fig)
    log(f"Saved Fixed 2-Panel Comparison Figure to: {fig_path}")


def main():
    log("=" * 80)
    log("RE-RENDERING CHAPTER 02 BENCHMARK WITH UNIFIED Y-AXIS [0, 56]")
    log("=" * 80)
    
    csv_path = os.path.join(TAB_DIR, "frame_vs_bout_results.csv")
    df_all = pd.read_csv(csv_path)
    
    # Empirical syllable stats
    dur_arr = np.array([812.9, 437.4]) # cached summary
    render_figures_and_reports(df_all, dur_arr)


if __name__ == "__main__":
    main()
