#!/usr/bin/env python3
"""
Test Syllable 0 / Outlier Impact on the P_n Information Ladder
==============================================================
Evaluates:
1. Exact frequency and distribution of Syllable <= 0 in the dataset.
2. Direct comparison of the P_n ladder:
   - Baseline (100-D, where index 0 is kept as a feature/target channel).
   - Strict 1..99 Re-indexed (99-D, where syllable labels 1..99 map to 0..98 and 0 is removed).
   - Syllable 1 Excluded (What happens if the lowest label is masked).
"""

import sys
import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)
sb = cache["session_bouts"]

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

def js_vec(P, Q, eps=1e-12):
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True); Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))

def run_pn_ladder(exclude_syll_zero=False):
    K_dim = 99 if exclude_syll_zero else 100
    sessions_data = []
    
    for sess_idx, bouts in enumerate(sb):
        if len(bouts) < 100: continue
        N = bouts[-1]["end_frame"] + 1
        fs = np.zeros(N, dtype=np.int32)
        sp = np.zeros((N, 6), dtype=np.float32)
        pose = np.zeros((N, 30), dtype=np.float32)
        
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_raw = b["syllable"]
            
            if exclude_syll_zero:
                # Syllables are 1..99 -> map 1..99 to 0..98. Syllable <=0 is ignored / mapped to -1.
                s_id = (s_raw - 1) if (s_raw >= 1 and s_raw <= 99) else -1
            else:
                s_id = min(max(s_raw, 0), 99)
                
            fs[sf:ef+1] = s_id
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                sp[sf:ef+1, :4] = vec[:4]; sp[sf:ef+1, 4] = vec[10]; sp[sf:ef+1, 5] = vec[12]
                if len(vec) >= 30: pose[sf:ef+1] = vec[:30]
                
        diffs = np.where(fs[1:] != fs[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool); onsets[0] = True; onsets[diffs] = True
        
        oh_bout = np.zeros((N, K_dim), dtype=np.float32)
        valid_onsets = onsets & (fs >= 0) & (fs < K_dim)
        oh_bout[valid_onsets, fs[valid_onsets]] = 1.0
        cum_bout = np.vstack([np.zeros((1, K_dim), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
        
        next_bout_ptrs = np.zeros(N, dtype=np.int32); cur = 0
        for d in diffs: next_bout_ptrs[cur:d] = d; cur = d
        next_bout_ptrs[cur:] = N
        
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp**2, axis=0)])
        
        sessions_data.append({
            "sess_idx": sess_idx, "N": N, "fs": fs, "pose": pose, "cum_bout": cum_bout,
            "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
            "cum_spatial_sq": cum_spatial_sq, "inj_frame": session_meta[sess_idx]["inj_frame"]
        })

    all_p1, all_p2, all_p25, all_p3, all_p4, all_tgt, all_sess = [], [], [], [], [], [], []

    for s in sessions_data:
        N = s["N"]
        if N < W_FRAMES + H_FRAMES + 200: continue
        inj_f = s["inj_frame"]
        
        t_anchors = np.arange(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES)
        t_tgt_start = s["next_bout_ptrs"][t_anchors]
        valid = (t_tgt_start + H_FRAMES < N)
        t_anchors = t_anchors[valid]
        t_tgt_start = t_tgt_start[valid]
        
        # Clean non-straddling
        past_start = t_anchors - W_FRAMES
        future_end = t_tgt_start + H_FRAMES
        no_straddle = (future_end < inj_f) | (past_start > inj_f)
        t_anchors = t_anchors[no_straddle]
        t_tgt_start = t_tgt_start[no_straddle]
        M = len(t_anchors)
        if M == 0: continue
        
        # P1
        t_norm = t_anchors / float(N)
        p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
        
        # P2
        sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
        sp_mean = sp_sum / float(W_FRAMES)
        sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
        sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
        p2 = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)
        
        # P2.5
        p25 = np.hstack([p2, s["pose"][t_anchors]]).astype(np.float32)
        
        # P3
        curr_syll = s["fs"][t_anchors]
        curr_oh = np.zeros((M, K_dim), dtype=np.float32)
        valid_syll = (curr_syll >= 0) & (curr_syll < K_dim)
        curr_oh[np.arange(M)[valid_syll], curr_syll[valid_syll]] = 1.0
        p3 = np.hstack([p2, curr_oh]).astype(np.float32)
        
        # P4 (Repertoire)
        b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        p_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * K_dim)
        p4 = np.hstack([p3, p_rep]).astype(np.float32)
        
        # Target
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * K_dim)
        
        all_p1.append(p1); all_p2.append(p2); all_p25.append(p25)
        all_p3.append(p3); all_p4.append(p4); all_tgt.append(target)
        all_sess.append(np.full(M, s["sess_idx"], dtype=int))
        
    X_p1, X_p2, X_p25 = np.vstack(all_p1), np.vstack(all_p2), np.vstack(all_p25)
    X_p3, X_p4, Y_tgt = np.vstack(all_p3), np.vstack(all_p4), np.vstack(all_tgt)
    groups = np.concatenate(all_sess)
    
    gkf = GroupKFold(n_splits=5)
    g1, g2, g25, g3, g4 = [], [], [], [], []
    
    for tr, te in gkf.split(X_p1, groups=groups):
        Y_tr, Y_te = Y_tgt[tr], Y_tgt[te]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        jb = js_vec(Y_te, p_glob)
        
        # P1
        p_1 = np.clip(Ridge(5.0).fit(X_p1[tr], Y_tr).predict(X_p1[te]), 1e-4, 1.0)
        g1.append(((jb - js_vec(Y_te, p_1 / np.sum(p_1, axis=1, keepdims=True))) / jb) * 100.0)
        
        # P2
        p_2 = np.clip(Ridge(5.0).fit(X_p2[tr], Y_tr).predict(X_p2[te]), 1e-4, 1.0)
        g2.append(((jb - js_vec(Y_te, p_2 / np.sum(p_2, axis=1, keepdims=True))) / jb) * 100.0)
        
        # P2.5
        p_25 = np.clip(Ridge(5.0).fit(X_p25[tr], Y_tr).predict(X_p25[te]), 1e-4, 1.0)
        g25.append(((jb - js_vec(Y_te, p_25 / np.sum(p_25, axis=1, keepdims=True))) / jb) * 100.0)
        
        # P3
        p_3 = np.clip(Ridge(5.0).fit(X_p3[tr], Y_tr).predict(X_p3[te]), 1e-4, 1.0)
        g3.append(((jb - js_vec(Y_te, p_3 / np.sum(p_3, axis=1, keepdims=True))) / jb) * 100.0)
        
        # P4
        p_4 = np.clip(Ridge(1.0).fit(X_p4[tr], Y_tr).predict(X_p4[te]), 1e-4, 1.0)
        g4.append(((jb - js_vec(Y_te, p_4 / np.sum(p_4, axis=1, keepdims=True))) / jb) * 100.0)
            
    ci = lambda arr: float(1.96 * np.std(arr) / np.sqrt(5))
    return {
        "P1_Time": (np.mean(g1), ci(g1), g1),
        "P2_Space": (np.mean(g2), ci(g2), g2),
        "P2.5_Pose": (np.mean(g25), ci(g25), g25),
        "P3_Syllable": (np.mean(g3), ci(g3), g3),
        "P4_Repertoire": (np.mean(g4), ci(g4), g4),
    }

if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATING IMPACT OF SYLLABLE 0 ON THE P_n INFORMATION LADDER")
    print("=" * 80)
    print("Running baseline P_n ladder (100-D: Syllables 0..99)...")
    res_100 = run_pn_ladder(exclude_syll_zero=False)
    
    print("\nRunning clean 99-D P_n ladder (Excluding Syllable 0: Syllables 1..99 mapped to 0..98)...")
    res_99 = run_pn_ladder(exclude_syll_zero=True)
    
    print("\n" + "=" * 80)
    print(f"{'Level / Feature Tier':<25} | {'Original (100-D)':<18} | {'Exclude Syll 0 (99-D)':<20} | {'Delta':<8}")
    print("=" * 80)
    for k in ["P1_Time", "P2_Space", "P2.5_Pose", "P3_Syllable", "P4_Repertoire"]:
        m1, c1, _ = res_100[k]
        m2, c2, _ = res_99[k]
        d = m2 - m1
        print(f"{k:<25} | +{m1:5.2f}% ± {c1:4.2f}%    | +{m2:5.2f}% ± {c2:4.2f}%        | {d:+5.2f} pp")
    print("=" * 80)
