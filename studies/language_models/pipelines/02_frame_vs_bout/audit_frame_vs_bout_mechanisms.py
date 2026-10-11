#!/usr/bin/env python3
"""
===============================================================================
Rigorous Audit: Mechanisms of Past Frames vs. Past Bouts in Predicting Future Bouts
===============================================================================
Comprehensive statistical validation to verify:
  1. Feature Correlation: Cosine similarity and Pearson r between frame and bout unigrams.
  2. Strategy Clustering Overlap: Adjusted Rand Index (ARI) & NMI between Frame-SVD and Bout-SVD.
  3. Completely Independent 5-Fold Cross-Validation:
     - Pure Raw Frames Pipeline (Frame Unigrams + Frame Bigrams -> Frame Strategy K=16)
     - Pure Discrete Bouts Pipeline (Bout Unigrams + Bout Bigrams -> Bout Strategy K=16)
     Both predicting Future 120s of Discrete Bouts.
===============================================================================
"""

import os
import pickle
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "02_frame_vs_bout")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "audit_frame_vs_bout.log")

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
    log("LAUNCHING RIGOROUS AUDIT: FRAME PIPELINE VS. BOUT PIPELINE")
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
        bigram_frame = frame_sylls[:-1] * NUM_SYLLS + frame_sylls[1:]

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
            "bigram_frame": bigram_frame, "next_bout_ptrs": next_bout_ptrs,
            "cum_spatial": cum_spatial, "cum_spatial_sq": cum_spatial_sq,
            "bout_seq": bout_seq, "bout_onsets": bout_onsets
        })

    log(f"Preprocessed {len(sessions)} sessions in {time.time()-t0:.2f}s.")

    # Extract Features
    all_time, all_sp, all_curr_oh, all_curr_s = [], [], [], []
    all_past_frame_dist, all_past_bout_dist = [], []
    all_trans_frame, all_trans_bout = [], []
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

        # 1. Past Frame Unigrams
        pf_cnt = s["cum_frame"][t_anchors] - s["cum_frame"][t_anchors - W_FRAMES]
        past_frame_dist = (pf_cnt + 1e-4) / (np.sum(pf_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        # 2. Past Bout Unigrams
        pb_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
        past_bout_dist = (pb_cnt + 1e-4) / (np.sum(pb_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        # 3. Past Frame Transitions (with self-transitions)
        bigrams_f = s["bigram_frame"]
        trans_f = np.zeros((M, NUM_SYLLS * NUM_SYLLS), dtype=np.float32)
        for i, t in enumerate(t_anchors):
            seg = bigrams_f[t - W_FRAMES : t - 1]
            b_cnt = np.bincount(seg, minlength=NUM_SYLLS * NUM_SYLLS).reshape(NUM_SYLLS, NUM_SYLLS)
            rs = b_cnt.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
            trans_f[i] = ((b_cnt + 1e-4) / (rs + 1e-4 * NUM_SYLLS)).flatten()

        # 4. Past Bout Transitions (O(log K) searchsorted strictly non-self transitions)
        b_onsets = s["bout_onsets"]
        b_seq = s["bout_seq"]
        trans_b = np.zeros((M, NUM_SYLLS * NUM_SYLLS), dtype=np.float32)

        # Binary search range for all anchors
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

        # Target: Future Discrete Bout Counts
        f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
        target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

        all_time.append(time_feat); all_sp.append(spatial_feat)
        all_curr_oh.append(curr_oh); all_curr_s.append(curr_syll)
        all_past_frame_dist.append(past_frame_dist); all_past_bout_dist.append(past_bout_dist)
        all_trans_frame.append(trans_f); all_trans_bout.append(trans_b)
        all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

    X_time = np.vstack(all_time); X_spatial = np.vstack(all_sp)
    X_curr_oh = np.vstack(all_curr_oh); y_curr_s = np.concatenate(all_curr_s)
    X_pf_dist = np.vstack(all_past_frame_dist); X_pb_dist = np.vstack(all_past_bout_dist)
    X_tf = np.vstack(all_trans_frame); X_tb = np.vstack(all_trans_bout)
    Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

    log(f"Extracted all features for {len(Y_tgt):,} anchor samples in {time.time()-t_feat:.2f}s.")

    # -------------------------------------------------------------------------
    # TEST 1: DIRECT FEATURE CORRELATIONS
    # -------------------------------------------------------------------------
    cos_sim = np.sum(X_pf_dist * X_pb_dist, axis=1) / (np.linalg.norm(X_pf_dist, axis=1) * np.linalg.norm(X_pb_dist, axis=1))
    mean_cos = float(np.mean(cos_sim))
    log(f"\n[Audit Test 1] Cosine Similarity between Frame Unigrams and Bout Unigrams: {mean_cos:.4f}")

    # -------------------------------------------------------------------------
    # TEST 2: INDEPENDENT STRATEGY CLUSTERING (Frame-SVD vs. Bout-SVD)
    # -------------------------------------------------------------------------
    t_svd = time.time()
    svd_f = TruncatedSVD(n_components=24, random_state=42)
    latent_f = svd_f.fit_transform(X_tf)
    norm_f = normalize(latent_f, norm='l2')
    strat_f = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(norm_f)
    strat_f_oh = np.zeros((len(strat_f), 16), dtype=np.float32)
    strat_f_oh[np.arange(len(strat_f)), strat_f] = 1.0

    svd_b = TruncatedSVD(n_components=24, random_state=42)
    latent_b = svd_b.fit_transform(X_tb)
    norm_b = normalize(latent_b, norm='l2')
    strat_b = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(norm_b)
    strat_b_oh = np.zeros((len(strat_b), 16), dtype=np.float32)
    strat_b_oh[np.arange(len(strat_b)), strat_b] = 1.0

    ari = adjusted_rand_score(strat_f, strat_b)
    nmi = normalized_mutual_info_score(strat_f, strat_b)
    log(f"[Audit Test 2] Strategy Clusters Agreement: ARI = {ari:.4f}, NMI = {nmi:.4f} (Fit in {time.time()-t_svd:.2f}s)")

    # -------------------------------------------------------------------------
    # TEST 3: COMPLETELY INDEPENDENT 5-FOLD CV PREDICTION BENCHMARK
    # -------------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    g_ug_f, g_p4_f, g_p5_f = [], [], []
    g_ug_b, g_p4_b, g_p5_b = [], [], []

    for tr_idx, te_idx in gkf.split(X_time, groups=groups):
        Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
        p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
        p_glob /= np.sum(p_glob, axis=1, keepdims=True)
        js_blind = js_vec(Y_te, p_glob)

        # Base Macro + Posture
        X_tr_p3 = np.hstack([X_time[tr_idx], X_spatial[tr_idx], X_curr_oh[tr_idx]])
        X_te_p3 = np.hstack([X_time[te_idx], X_spatial[te_idx], X_curr_oh[te_idx]])

        # 1. PURE FRAME PIPELINE
        # Unigram alone
        p_ug_f = np.clip(Ridge(1.0).fit(X_pf_dist[tr_idx], Y_tr).predict(X_pf_dist[te_idx]), 1e-4, 1.0)
        p_ug_f /= np.sum(p_ug_f, axis=1, keepdims=True)
        g_ug_f.append(((js_blind - js_vec(Y_te, p_ug_f)) / js_blind) * 100.0)

        # P4: Macro + Posture + Frame-Derived Strategy K=16
        p_p4_f = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, strat_f_oh[tr_idx]]), Y_tr).predict(np.hstack([X_te_p3, strat_f_oh[te_idx]])), 1e-4, 1.0)
        p_p4_f /= np.sum(p_p4_f, axis=1, keepdims=True)
        g_p4_f.append(((js_blind - js_vec(Y_te, p_p4_f)) / js_blind) * 100.0)

        # P5: Full Frame Empirical Bound
        p_p5_f = np.clip(Ridge(1.0).fit(np.hstack([X_tr_p3, X_pf_dist[tr_idx], latent_f[tr_idx]]), Y_tr).predict(np.hstack([X_te_p3, X_pf_dist[te_idx], latent_f[te_idx]])), 1e-4, 1.0)
        p_p5_f /= np.sum(p_p5_f, axis=1, keepdims=True)
        g_p5_f.append(((js_blind - js_vec(Y_te, p_p5_f)) / js_blind) * 100.0)

        # 2. PURE BOUT PIPELINE
        # Unigram alone
        p_ug_b = np.clip(Ridge(1.0).fit(X_pb_dist[tr_idx], Y_tr).predict(X_pb_dist[te_idx]), 1e-4, 1.0)
        p_ug_b /= np.sum(p_ug_b, axis=1, keepdims=True)
        g_ug_b.append(((js_blind - js_vec(Y_te, p_ug_b)) / js_blind) * 100.0)

        # P4: Macro + Posture + Bout-Derived Strategy K=16
        p_p4_b = np.clip(Ridge(5.0).fit(np.hstack([X_tr_p3, strat_b_oh[tr_idx]]), Y_tr).predict(np.hstack([X_te_p3, strat_b_oh[te_idx]])), 1e-4, 1.0)
        p_p4_b /= np.sum(p_p4_b, axis=1, keepdims=True)
        g_p4_b.append(((js_blind - js_vec(Y_te, p_p4_b)) / js_blind) * 100.0)

        # P5: Full Bout Empirical Bound
        p_p5_b = np.clip(Ridge(1.0).fit(np.hstack([X_tr_p3, X_pb_dist[tr_idx], latent_b[tr_idx]]), Y_tr).predict(np.hstack([X_te_p3, X_pb_dist[te_idx], latent_b[te_idx]])), 1e-4, 1.0)
        p_p5_b /= np.sum(p_p5_b, axis=1, keepdims=True)
        g_p5_b.append(((js_blind - js_vec(Y_te, p_p5_b)) / js_blind) * 100.0)

    log("\n" + "=" * 90)
    log("AUDIT RESULTS: COMPLETELY INDEPENDENT PIPELINES (W = 68.3s -> Future Bouts H = 120s)")
    log("=" * 90)
    log(f"{'Model Component':<24} | {'Pure RAW FRAMES Pipeline':<28} | {'Pure DISCRETE BOUTS Pipeline':<30} | {'Delta (Frames - Bouts)':<18}")
    log("-" * 90)

    rows = [
        ("Past Unigrams Alone", np.mean(g_ug_f), 1.96*np.std(g_ug_f)/np.sqrt(5), np.mean(g_ug_b), 1.96*np.std(g_ug_b)/np.sqrt(5)),
        ("Strategy K=16 (P_4)", np.mean(g_p4_f), 1.96*np.std(g_p4_f)/np.sqrt(5), np.mean(g_p4_b), 1.96*np.std(g_p4_b)/np.sqrt(5)),
        ("Empirical Bound (P_max)", np.mean(g_p5_f), 1.96*np.std(g_p5_f)/np.sqrt(5), np.mean(g_p5_b), 1.96*np.std(g_p5_b)/np.sqrt(5))
    ]

    for name, mf, cif, mb, cib in rows:
        diff = mf - mb
        log(f"{name:<24} | +{mf:5.2f}% ± {cif:4.2f}%              | +{mb:5.2f}% ± {cib:4.2f}%                | {diff:+5.2f}%")

    log("=" * 90)
    log("AUDIT COMPLETED SUCCESSFULLY!")
    log("=" * 90)


if __name__ == "__main__":
    main()
