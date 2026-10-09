#!/usr/bin/env python3
"""
===============================================================================
Exact 68.3s Benchmark: Past Frames vs. Past Bouts Predicting Future Bouts
===============================================================================
Direct head-to-head comparison at the optimal past memory horizon (W = 68.3s):
  Input A: Past 68.3s of Raw Frames (Time-weighted duration histogram)
  Input B: Past 68.3s of Discrete Bouts (Action count histogram)
  Target:  Future 120.0s of Discrete Bouts (Discrete Action Grammar)

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

# Repository setup
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "02_frame_vs_bout")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "68s_comparison.log")

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
    log(f"LAUNCHING 68.3s BENCHMARK: PAST FRAMES VS. PAST BOUTS PREDICTING FUTURE BOUTS")
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
        
        # 2. Bigram Frame Indices
        bigram_frame = frame_sylls[:-1] * NUM_SYLLS + frame_sylls[1:]
        
        # 3. Discrete Bout Prefix Sums (onsets only)
        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool)
        onsets[0] = True
        onsets[diffs] = True
        
        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        oh_bout[onsets, frame_sylls[onsets]] = 1.0
        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
        
        # 4. Next non-self pointer
        next_bout_ptrs = np.zeros(N, dtype=np.int32)
        cur = 0
        for d in diffs:
            next_bout_ptrs[cur:d] = d
            cur = d
        next_bout_ptrs[cur:] = N
        
        # 5. Spatial prefix sums
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])
        
        sessions.append({
            "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
            "cum_frame": cum_frame, "cum_bout": cum_bout,
            "bigram_frame": bigram_frame, "next_bout_ptrs": next_bout_ptrs,
            "cum_spatial": cum_spatial, "cum_spatial_sq": cum_spatial_sq
        })
        
    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s.")
    
    # Extract Features
    all_time, all_sp, all_curr_oh, all_curr_s = [], [], [], []
    all_past_frame_dist, all_past_bout_dist, all_past_trans = [], [], []
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
        
        # Time
        t_norm = t_anchors / float(N)
        time_feat = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
        
        # Spatial
        sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
        sp_mean = sp_sum / float(W_FRAMES)
        sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
        sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
        spatial_feat = np.hstack([sp_mean, np.sqrt(sp_var)]).astype(np.float32)
        
        # 1. Past Frame Unigram Distribution
        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        past_frame_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        # 2. Past Bout Unigram Distribution
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        past_bout_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        # 3. Bigram Transitions
        bigrams = s["bigram_frame"]
        trans_prob = np.zeros((M, NUM_SYLLS * NUM_SYLLS), dtype=np.float32)
        for i, t in enumerate(t_anchors):
            seg = bigrams[t - W_FRAMES : t - 1]
            b_counts = np.bincount(seg, minlength=NUM_SYLLS * NUM_SYLLS).reshape(NUM_SYLLS, NUM_SYLLS)
            row_sums = b_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            trans_prob[i] = ((b_counts + 1e-4) / (row_sums + 1e-4 * NUM_SYLLS)).flatten()
            
        curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
        curr_oh[np.arange(M), curr_syll] = 1.0
        
        # Target: Future Discrete Bout Counts
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)
        
        all_time.append(time_feat); all_sp.append(spatial_feat)
        all_curr_oh.append(curr_oh); all_curr_s.append(curr_syll)
        all_past_frame_dist.append(past_frame_dist)
        all_past_bout_dist.append(past_bout_dist)
        all_past_trans.append(trans_prob)
        all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))
        
    X_time = np.vstack(all_time); X_spatial = np.vstack(all_sp)
    X_curr_oh = np.vstack(all_curr_oh); y_curr_s = np.concatenate(all_curr_s)
    X_past_frame_dist = np.vstack(all_past_frame_dist)
    X_past_bout_dist = np.vstack(all_past_bout_dist)
    X_past_trans = np.vstack(all_past_trans)
    Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)
    log(f"Extracted features for {len(Y_tgt):,} anchor samples in {time.time()-t_feat:.2f}s.")
    
    # Latent Strategy SVD
    svd = TruncatedSVD(n_components=24, random_state=42)
    latent_24d = svd.fit_transform(X_past_trans)
    latent_norm = normalize(latent_24d, norm='l2')
    strat_k16 = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(latent_norm)
    strat_k16_oh = np.zeros((len(strat_k16), 16), dtype=np.float32)
    strat_k16_oh[np.arange(len(strat_k16)), strat_k16] = 1.0
    
    # Cross-Validation Evaluator
    def evaluate_cv(X_past_unigrams, name):
        gkf = GroupKFold(n_splits=5)
        f_markov, f_p2, f_p3, f_p4, f_p5 = [], [], [], [], []
        
        for tr_idx, te_idx in gkf.split(X_time, groups=groups):
            Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
            p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
            p_glob /= np.sum(p_glob, axis=1, keepdims=True)
            js_blind = js_vec(Y_te, p_glob)
            
            # Markov
            c_tr, c_te = y_curr_s[tr_idx], y_curr_s[te_idx]
            sm = np.zeros((NUM_SYLLS, NUM_SYLLS), dtype=np.float32)
            for s_id in range(NUM_SYLLS):
                mask = (c_tr == s_id)
                sm[s_id] = np.mean(Y_tr[mask], axis=0) if np.sum(mask) > 0 else p_glob[0]
                sm[s_id] /= np.sum(sm[s_id])
            p_m = sm[c_te]
            f_markov.append(((js_blind - js_vec(Y_te, p_m)) / js_blind) * 100.0)
            
            # P2: Macro Context
            X_tr_p2 = np.hstack([X_time[tr_idx], X_spatial[tr_idx]])
            X_te_p2 = np.hstack([X_time[te_idx], X_spatial[te_idx]])
            p_p2 = np.clip(Ridge(10.0).fit(X_tr_p2, Y_tr).predict(X_te_p2), 1e-4, 1.0)
            p_p2 /= np.sum(p_p2, axis=1, keepdims=True)
            f_p2.append(((js_blind - js_vec(Y_te, p_p2)) / js_blind) * 100.0)
            
            # P3: Macro + Posture
            X_tr_p3 = np.hstack([X_tr_p2, X_curr_oh[tr_idx]])
            X_te_p3 = np.hstack([X_te_p2, X_curr_oh[te_idx]])
            p_p3 = np.clip(Ridge(10.0).fit(X_tr_p3, Y_tr).predict(X_te_p3), 1e-4, 1.0)
            p_p3 /= np.sum(p_p3, axis=1, keepdims=True)
            f_p3.append(((js_blind - js_vec(Y_te, p_p3)) / js_blind) * 100.0)
            
            # P4: Strategy K=16
            X_tr_p4 = np.hstack([X_tr_p3, strat_k16_oh[tr_idx]])
            X_te_p4 = np.hstack([X_te_p3, strat_k16_oh[te_idx]])
            p_p4 = np.clip(Ridge(5.0).fit(X_tr_p4, Y_tr).predict(X_te_p4), 1e-4, 1.0)
            p_p4 /= np.sum(p_p4, axis=1, keepdims=True)
            f_p4.append(((js_blind - js_vec(Y_te, p_p4)) / js_blind) * 100.0)
            
            # P5: P_max
            X_tr_p5 = np.hstack([X_tr_p3, X_past_unigrams[tr_idx], latent_24d[tr_idx]])
            X_te_p5 = np.hstack([X_te_p3, X_past_unigrams[te_idx], latent_24d[te_idx]])
            p_p5 = np.clip(Ridge(1.0).fit(X_tr_p5, Y_tr).predict(X_te_p5), 1e-4, 1.0)
            p_p5 /= np.sum(p_p5, axis=1, keepdims=True)
            f_p5.append(((js_blind - js_vec(Y_te, p_p5)) / js_blind) * 100.0)
            
        return {
            "Markov": (float(np.mean(f_markov)), float(1.96*np.std(f_markov)/np.sqrt(5))),
            "P2_Macro": (float(np.mean(f_p2)), float(1.96*np.std(f_p2)/np.sqrt(5))),
            "P3_Posture": (float(np.mean(f_p3)), float(1.96*np.std(f_p3)/np.sqrt(5))),
            "P4_Strat16": (float(np.mean(f_p4)), float(1.96*np.std(f_p4)/np.sqrt(5))),
            "P5_Pmax": (float(np.mean(f_p5)), float(1.96*np.std(f_p5)/np.sqrt(5))),
        }
        
    res_frames = evaluate_cv(X_past_frame_dist, "Past Frames")
    res_bouts = evaluate_cv(X_past_bout_dist, "Past Bouts")
    
    log("\n" + "=" * 90)
    log("HEAD-TO-HEAD RESULTS: PAST 68.3s OF FRAMES VS. PAST 68.3s OF BOUTS PREDICTING BOUTS (H=120s)")
    log("=" * 90)
    log(f"{'Model Tier':<22} | {'Input A: Past 68.3s RAW FRAMES':<28} | {'Input B: Past 68.3s DISCRETE BOUTS':<30} | {'Difference (A - B)':<18}")
    log("-" * 90)
    
    table_rows = []
    for k, name in [('Markov', '1-Step Markov (S_t)'), 
                    ('P2_Macro', 'Macro Context (P_2)'), 
                    ('P3_Posture', 'Macro + Posture (P_3)'), 
                    ('P4_Strat16', 'Strategy K=16 (P_4)'), 
                    ('P5_Pmax', 'Empirical Ceiling (P_max)')]:
        mf, cif = res_frames[k]
        mb, cib = res_bouts[k]
        diff = mf - mb
        log(f"{name:<22} | +{mf:5.2f}% ± {cif:4.2f}%              | +{mb:5.2f}% ± {cib:4.2f}%                | {diff:+5.2f}%")
        table_rows.append({
            "model_tier": name,
            "past_frames_mean": mf, "past_frames_ci": cif,
            "past_bouts_mean": mb, "past_bouts_ci": cib,
            "difference": diff
        })
    log("=" * 90)
    
    df_tab = pd.DataFrame(table_rows)
    csv_path = os.path.join(TAB_DIR, "68s_frames_vs_bouts_table.csv")
    df_tab.to_csv(csv_path, index=False)
    log(f"Saved numerical table to: {csv_path}")


if __name__ == "__main__":
    main()
