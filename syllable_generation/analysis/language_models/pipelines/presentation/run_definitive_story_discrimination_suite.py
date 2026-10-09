#!/usr/bin/env python3
"""
run_definitive_story_discrimination_suite.py
=============================================================================
FOUR DEFINITIVE TESTS DISCRIMINATING THE GENOTYPE & TEMPORAL SCALE NARRATIVES:

1. TEST 1: Direct Genotype x Coordinate-System Interaction Statistics
   - Evaluates per-mouse contrasts from the 2x2 matched factorial:
     * C_boundary = Time Window (15s) - Bout Window (20b)
     * C_weight   = Event Frequency - Duration Dwell
   - Formal statistical tests: Mann-Whitney U, Welch's t-test, 10,000 label permutations.

2. TEST 2: True Local Markov-Order Experiment (Next-Bout Prediction S_{t+1})
   - Evaluates discrete Markov sequence memory across orders m in [1, 2, 3, 4]
   - Uses hierarchical Jelinek-Mercer linear backoff smoothing under LOAO-CV.
   - Tests the Two-Scale Hypothesis: Is KI shallower locally while deeper in macro-regimes?

3. TEST 3: Local Drift-Preserving Temporal Null Control
   - Local surrogate pairing history with future within the same 3-6 minute neighborhood
     (|delta| in [180s, 360s]), preserving local habituation, arousal, and drug phase.

4. TEST 4: Strict Zero-Leakage Fold-Wise Shared NMF
   - Balanced Shared NMF (K=12) fit strictly inside every LOAO training fold.
   - Out-of-fold evaluation of Enriched vs. Suppressed subspace predictive gains.

Deliverables:
- Tables:
  * table_genotype_coordinate_interaction.csv
  * table_local_markov_order_suite.csv
  * table_local_drift_temporal_null.csv
  * table_strict_foldwise_shared_nmf.csv
- Figure: fig_definitive_story_discrimination_suite.png / .pdf
=============================================================================
"""

import os
import sys
import pickle
import time
import shutil
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------------
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FPS = 80.025
NUM_SYLLS = 100
W_MAX_SEC = 60.0
W_MAX_FRAMES = int(W_MAX_SEC * FPS)
N_MAX_BOUTS = 50
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
ANCHOR_STRIDE = 2
LAMBDA = 10.0

COHORT_LABELS = {"WT_Saline": "WT Saline", "KI_Saline": "KI Saline"}
COLORS = {"WT_Saline": "#1f77b4", "KI_Saline": "#2ca02c"}

# ---------------------------------------------------------------------------
# UTILITIES: JS DIVERGENCE
# ---------------------------------------------------------------------------
def js_divergence(P, Q):
    P = np.clip(P, 1e-12, 1.0)
    Q = np.clip(Q, 1e-12, 1.0)
    P = P / np.sum(P, axis=-1, keepdims=True)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(np.where(P > 0, P * np.log2(P / np.maximum(M, 1e-12)), 0.0), axis=-1)
    kl_qm = np.sum(np.where(Q > 0, Q * np.log2(Q / np.maximum(M, 1e-12)), 0.0), axis=-1)
    return 0.5 * (kl_pm + kl_qm)

# ---------------------------------------------------------------------------
# DATA LOADING & MATCHED ANCHORS
# ---------------------------------------------------------------------------
def build_matched_dataset():
    print("Loading bout cache and raw metadata...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data
    
    with open(RAW_PATH, "rb") as f:
        raw_sessions = pickle.load(f)
        
    sessions = []
    total_anchors = 0
    
    for i, bouts in enumerate(session_bouts):
        if len(bouts) < 150:
            continue
        raw_s = raw_sessions[i]
        meta = getattr(raw_s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        if cohort not in ["WT_Saline", "KI_Saline"]:
            continue
        inj_f = meta.get("injection_frame", 0)
        sid = meta.get("animal_id", f"mouse_{i}")
        
        post_bouts = [b for b in bouts if b["start_frame"] > inj_f]
        if len(post_bouts) < N_MAX_BOUTS + 50:
            continue
            
        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int32)
        starts = np.array([b["start_frame"] for b in post_bouts], dtype=np.int64)
        ends = np.array([b["end_frame"] for b in post_bouts], dtype=np.int64)
        max_f = ends[-1]
        M = len(post_bouts)
        
        # Cumulative event counts over bouts
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0
            
        # Cumulative durations over bouts
        cum_durations = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_durations[idx_b + 1] = cum_durations[idx_b]
            dur_sec = (ends[idx_b] - starts[idx_b]) / FPS
            cum_durations[idx_b + 1, sylls[idx_b]] += dur_sec
            
        # Continuous frame array for future target Y (h = 120s) and clock-time history
        N_frames = max_f + 1
        cum_frame_sylls = np.zeros((N_frames, NUM_SYLLS), dtype=np.float32)
        for b in post_bouts:
            dur = b["end_frame"] - b["start_frame"]
            if dur > 0:
                cum_frame_sylls[b["start_frame"]:b["end_frame"], b["syllable"]] = 1.0
        cum_frame_sylls = np.cumsum(cum_frame_sylls, axis=0)
        
        valid_anchors = []
        min_start = starts[0]
        for anc in range(N_MAX_BOUTS, M, ANCHOR_STRIDE):
            t_st = starts[anc]
            t_ed = ends[anc]
            if (t_st - W_MAX_FRAMES >= min_start) and (t_ed + H_FRAMES < N_frames):
                valid_anchors.append(anc)
                
        if len(valid_anchors) < 50:
            continue
            
        valid_anchors = np.array(valid_anchors, dtype=np.int32)
        K_obs = len(valid_anchors)
        total_anchors += K_obs
        
        t_ends = ends[valid_anchors]
        t_starts = starts[valid_anchors]
        
        # Target Y: Future h=120s Syllable distribution (K_obs, 100)
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)
        
        # Current syllable one-hot
        s_curr = np.zeros((K_obs, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(K_obs), sylls[valid_anchors]] = 1.0
        
        # 4 Factorial History Representations (strictly simplex-normalized, sum to 1.0)
        # 1. BE: 20 Bouts, Event Frequency
        raw_be = cum_events[valid_anchors] - cum_events[valid_anchors - 20]
        X_be = raw_be / 20.0
        
        # 2. BD: 20 Bouts, Duration Dwell
        raw_bd = cum_durations[valid_anchors] - cum_durations[valid_anchors - 20]
        X_bd = raw_bd / (np.sum(raw_bd, axis=1, keepdims=True) + 1e-6)
        
        # 3. TE: 15 Seconds, Event Frequency
        W15_frames = int(15.0 * FPS)
        raw_te = np.zeros((K_obs, NUM_SYLLS), dtype=np.float32)
        raw_td = np.zeros((K_obs, NUM_SYLLS), dtype=np.float32)
        for idx_k, anc in enumerate(valid_anchors):
            t_st = t_starts[idx_k]
            t_w_st = t_st - W15_frames
            # Find bouts within [t_w_st, t_st]
            b_indices = np.where((starts <= t_st) & (ends >= t_w_st))[0]
            for bi in b_indices:
                s_id = sylls[bi]
                raw_te[idx_k, s_id] += 1.0
                ovlp = min(ends[bi], t_st) - max(starts[bi], t_w_st)
                if ovlp > 0:
                    raw_td[idx_k, s_id] += (ovlp / FPS)
        X_te = raw_te / (np.sum(raw_te, axis=1, keepdims=True) + 1e-6)
        X_td = raw_td / (np.sum(raw_td, axis=1, keepdims=True) + 1e-6)
        
        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "n_samples": K_obs,
            "raw_sylls": sylls,
            "valid_anchors": valid_anchors,
            "anchor_frames": t_ends,
            "Y_dist": Y_dist,
            "s_curr": s_curr,
            "X_be": X_be,
            "X_bd": X_bd,
            "X_te": X_te,
            "X_td": X_td
        })
        
    print(f"Loaded {len(sessions)} valid Saline sessions ({total_anchors} anchors).")
    return sessions

# ---------------------------------------------------------------------------
# MODULE 1: GENOTYPE x COORDINATE-SYSTEM INTERACTION STATISTICS
# ---------------------------------------------------------------------------
def run_test1_genotype_coordinate_interaction(sessions):
    print("\n" + "="*80)
    print("TEST 1: GENOTYPE x COORDINATE-SYSTEM INTERACTION STATISTICS")
    print("="*80)
    
    animal_records = []
    
    for c in ["WT_Saline", "KI_Saline"]:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        
        for test_idx in range(n_mice):
            te = c_sess[test_idx]
            tr = [c_sess[j] for j in range(n_mice) if j != test_idx]
            
            Y_tr = np.vstack([s["Y_dist"] for s in tr])
            Y_te = te["Y_dist"]
            p_prior = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_prior))
            
            X0_tr = np.vstack([s["s_curr"] for s in tr])
            X0_te = te["s_curr"]
            beta0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
            Y0_pr = np.clip(X0_te @ beta0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0
            
            # Evaluate 4 factorial representations
            gains = {}
            for k_rep in ["X_be", "X_bd", "X_te", "X_td"]:
                H_tr_raw = np.vstack([s[k_rep] for s in tr])
                H_te_raw = te[k_rep]
                mu_h = np.mean(H_tr_raw, axis=0, keepdims=True)
                sig_h = np.std(H_tr_raw, axis=0, keepdims=True) + 1e-6
                H_tr = (H_tr_raw - mu_h) / sig_h
                H_te = (H_te_raw - mu_h) / sig_h
                
                XH_tr = np.hstack([X0_tr, H_tr])
                XH_te = np.hstack([X0_te, H_te])
                betaH = np.linalg.solve(XH_tr.T @ XH_tr + LAMBDA * np.eye(200, dtype=np.float32), XH_tr.T @ Y_tr)
                YH_pr = np.clip(XH_te @ betaH, 1e-12, None); YH_pr /= YH_pr.sum(axis=-1, keepdims=True)
                d_h = np.mean(js_divergence(Y_te, YH_pr))
                g_h = (d_prior - d_h) / d_prior * 100.0
                gains[k_rep] = g_h - g_m0
                
            # Contrasts:
            # C_boundary = Time preference (15s) - Bout preference (20b)
            c_boundary = 0.5 * (gains["X_te"] + gains["X_td"]) - 0.5 * (gains["X_be"] + gains["X_bd"])
            # C_weight = Event preference - Dwell preference
            c_weight = 0.5 * (gains["X_be"] + gains["X_te"]) - 0.5 * (gains["X_bd"] + gains["X_td"])
            
            animal_records.append({
                "cohort": c,
                "sid": te["sid"],
                "gain_be": gains["X_be"],
                "gain_bd": gains["X_bd"],
                "gain_te": gains["X_te"],
                "gain_td": gains["X_td"],
                "c_boundary": c_boundary,
                "c_weight": c_weight
            })
            
    df_animals = pd.DataFrame(animal_records)
    csv_path = os.path.join(TAB_DIR, "table_genotype_coordinate_interaction.csv")
    df_animals.to_csv(csv_path, index=False)
    
    # Statistical tests: WT vs KI
    wt_c_bound = df_animals[df_animals["cohort"] == "WT_Saline"]["c_boundary"].values
    ki_c_bound = df_animals[df_animals["cohort"] == "KI_Saline"]["c_boundary"].values
    wt_c_weight = df_animals[df_animals["cohort"] == "WT_Saline"]["c_weight"].values
    ki_c_weight = df_animals[df_animals["cohort"] == "KI_Saline"]["c_weight"].values
    
    # Mann-Whitney U
    u_bound, p_bound_u = stats.mannwhitneyu(wt_c_bound, ki_c_bound, alternative="two-sided")
    u_weight, p_weight_u = stats.mannwhitneyu(ki_c_weight, wt_c_weight, alternative="two-sided")
    
    # Welch's t-test
    t_bound, p_bound_t = stats.ttest_ind(wt_c_bound, ki_c_bound, equal_var=False)
    t_weight, p_weight_t = stats.ttest_ind(ki_c_weight, wt_c_weight, equal_var=False)
    
    # Monte Carlo Permutation Test (10,000 resamples)
    np.random.seed(42)
    N_PERM = 10000
    obs_diff_bound = np.mean(wt_c_bound) - np.mean(ki_c_bound)
    obs_diff_weight = np.mean(ki_c_weight) - np.mean(wt_c_weight)
    
    all_bound = np.concatenate([wt_c_bound, ki_c_bound])
    all_weight = np.concatenate([wt_c_weight, ki_c_weight])
    n_wt = len(wt_c_bound)
    
    perm_diffs_bound = []
    perm_diffs_weight = []
    for _ in range(N_PERM):
        p_idx = np.random.permutation(len(all_bound))
        perm_diffs_bound.append(np.mean(all_bound[p_idx[:n_wt]]) - np.mean(all_bound[p_idx[n_wt:]]))
        perm_diffs_weight.append(np.mean(all_weight[p_idx[n_wt:]]) - np.mean(all_weight[p_idx[:n_wt]]))
        
    p_bound_perm = np.mean(np.abs(perm_diffs_bound) >= np.abs(obs_diff_bound))
    p_weight_perm = np.mean(np.abs(perm_diffs_weight) >= np.abs(obs_diff_weight))
    
    print("\n--- STATISTICAL RESULTS: GENOTYPE x COORDINATE-SYSTEM ---")
    print(f"1. Boundary Contrast C_boundary (Physical Time Preference: Time - Bout):")
    print(f"   * WT Saline: {np.mean(wt_c_bound):+5.2f}% ± {stats.sem(wt_c_bound):4.2f}%")
    print(f"   * KI Saline: {np.mean(ki_c_bound):+5.2f}% ± {stats.sem(ki_c_bound):4.2f}%")
    print(f"   * Observed Difference (WT - KI): {obs_diff_bound:+5.2f}%")
    print(f"   * Mann-Whitney U p = {p_bound_u:.4f} | Welch t p = {p_bound_t:.4f} | Permutation p = {p_bound_perm:.4f}")
    
    print(f"\n2. Weighting Contrast C_weight (Event-Count Preference: Event - Dwell):")
    print(f"   * WT Saline: {np.mean(wt_c_weight):+5.2f}% ± {stats.sem(wt_c_weight):4.2f}%")
    print(f"   * KI Saline: {np.mean(ki_c_weight):+5.2f}% ± {stats.sem(ki_c_weight):4.2f}%")
    print(f"   * Observed Difference (KI - WT): {obs_diff_weight:+5.2f}%")
    print(f"   * Mann-Whitney U p = {p_weight_u:.4f} | Welch t p = {p_weight_t:.4f} | Permutation p = {p_weight_perm:.4f}")
    
    return df_animals, {
        "obs_diff_bound": obs_diff_bound, "p_bound_perm": p_bound_perm,
        "obs_diff_weight": obs_diff_weight, "p_weight_perm": p_weight_perm
    }

# ---------------------------------------------------------------------------
# MODULE 2: TRUE LOCAL MARKOV-ORDER EXPERIMENT (S_{t+1})
# ---------------------------------------------------------------------------
def run_test2_local_markov_order(sessions):
    print("\n" + "="*80)
    print("TEST 2: TRUE LOCAL MARKOV-ORDER EXPERIMENT (S_{t+1} PREDICTION)")
    print("="*80)
    
    orders = [0, 1, 2, 3, 4]
    results = []
    
    for c in ["WT_Saline", "KI_Saline"]:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        
        mouse_nlls = {m: [] for m in orders}
        
        for test_idx in range(n_mice):
            te_seq = c_sess[test_idx]["raw_sylls"]
            tr_seqs = [c_sess[j]["raw_sylls"] for j in range(n_mice) if j != test_idx]
            
            # Count n-grams on training sequences
            counts = {m: {} for m in orders}
            tot_unigrams = 0
            for seq in tr_seqs:
                for idx in range(len(seq)):
                    tot_unigrams += 1
                    s_curr = seq[idx]
                    counts[0][s_curr] = counts[0].get(s_curr, 0) + 1
                    for m in [1, 2, 3, 4]:
                        if idx >= m:
                            ctx = tuple(seq[idx - m : idx])
                            if ctx not in counts[m]:
                                counts[m][ctx] = {}
                            counts[m][ctx][s_curr] = counts[m][ctx].get(s_curr, 0) + 1
                            
            # Compute log loss on held-out test sequence for each order
            # Using Jelinek-Mercer hierarchical interpolation:
            # P_m(s | ctx) = lambda * P_m_MLE(s | ctx) + (1 - lambda) * P_{m-1}(s | shorter_ctx)
            # lambda = count(ctx) / (count(ctx) + alpha)
            ALPHA = 5.0
            
            for m in orders:
                nll_sum = 0.0
                eval_count = 0
                
                # Start evaluating from index 4 so all orders evaluate on identical tokens!
                for idx in range(4, len(te_seq)):
                    target_s = te_seq[idx]
                    
                    # Compute interpolated probability hierarchically
                    # Base unigram
                    p_curr = (counts[0].get(target_s, 0) + 0.1) / (tot_unigrams + 10.0)
                    
                    for cur_m in range(1, m + 1):
                        ctx = tuple(te_seq[idx - cur_m : idx])
                        ctx_dict = counts[cur_m].get(ctx, {})
                        c_ctx = sum(ctx_dict.values())
                        if c_ctx > 0:
                            p_mle = ctx_dict.get(target_s, 0) / float(c_ctx)
                            lam = c_ctx / (c_ctx + ALPHA)
                            p_curr = lam * p_mle + (1.0 - lam) * p_curr
                        else:
                            p_curr = p_curr # fallback to lower order
                            
                    p_curr = max(p_curr, 1e-10)
                    nll_sum += -np.log2(p_curr)
                    eval_count += 1
                    
                mean_nll = nll_sum / eval_count
                mouse_nlls[m].append(mean_nll)
                
        # Aggregate across mice
        for m in orders:
            vals = mouse_nlls[m]
            gain_over_m0 = [mouse_nlls[0][i] - mouse_nlls[m][i] for i in range(n_mice)]
            results.append({
                "cohort": c,
                "order": m,
                "mean_nll": np.mean(vals),
                "sem_nll": float(stats.sem(vals)),
                "gain_over_m0": np.mean(gain_over_m0),
                "sem_gain": float(stats.sem(gain_over_m0))
            })
            print(f"Cohort {c:10s} | Order {m}: NLL = {np.mean(vals):.3f} bits | Gain = {np.mean(gain_over_m0):+.3f} ± {stats.sem(gain_over_m0):.3f} bits/bout")
            
    df_markov = pd.DataFrame(results)
    csv_markov_path = os.path.join(TAB_DIR, "table_local_markov_order_suite.csv")
    df_markov.to_csv(csv_markov_path, index=False)
    return df_markov

# ---------------------------------------------------------------------------
# MODULE 3: LOCAL DRIFT-PRESERVING TEMPORAL NULL
# ---------------------------------------------------------------------------
def run_test3_local_drift_null(sessions):
    print("\n" + "="*80)
    print("TEST 3: LOCAL DRIFT-PRESERVING TEMPORAL NULL CONTROL")
    print("="*80)
    
    # We compare:
    # 1. Real History Gain (Delta_real)
    # 2. Local Drift-Preserving Null: paired within same 3-6 min neighborhood (|delta_t| in [180s, 360s])
    # 3. Global Circular Null (half session shift)
    
    results = []
    
    for c in ["WT_Saline", "KI_Saline"]:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        
        real_gains = []
        local_null_gains = []
        global_null_gains = []
        
        for test_idx in range(n_mice):
            te = c_sess[test_idx]
            tr = [c_sess[j] for j in range(n_mice) if j != test_idx]
            
            K_obs = te["n_samples"]
            t_frames = te["anchor_frames"]
            
            # Construct Local Null Target:
            # For each anchor k, find an anchor in [180s, 360s] neighborhood
            # (enforcing separation > 120s to break history-future alignment, but < 360s to preserve slow state)
            local_idx = np.zeros(K_obs, dtype=np.int32)
            np.random.seed(test_idx * 100 + 42)
            for k in range(K_obs):
                t_k = t_frames[k]
                # Candidates:
                cand = np.where((np.abs(t_frames - t_k) >= int(180.0 * FPS)) & 
                                (np.abs(t_frames - t_k) <= int(360.0 * FPS)))[0]
                if len(cand) > 0:
                    local_idx[k] = np.random.choice(cand)
                else:
                    # Fallback to nearest anchor outside guard band
                    cand_fb = np.where(np.abs(t_frames - t_k) >= int(180.0 * FPS))[0]
                    local_idx[k] = cand_fb[0] if len(cand_fb) > 0 else (k + K_obs // 2) % K_obs
                    
            Y_te_local_null = te["Y_dist"][local_idx]
            
            # Global circular shift
            circ_idx = (np.arange(K_obs) + K_obs // 2) % K_obs
            H_te_circ_null = te["X_be"][circ_idx]
            
            # Fitting LOAO
            Y_tr = np.vstack([s["Y_dist"] for s in tr])
            Y_te = te["Y_dist"]
            p_prior = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_prior))
            d_prior_local = np.mean(js_divergence(Y_te_local_null, p_prior))
            
            X0_tr = np.vstack([s["s_curr"] for s in tr])
            X0_te = te["s_curr"]
            beta0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
            
            # M0
            Y0_pr = np.clip(X0_te @ beta0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
            d_m0_local = np.mean(js_divergence(Y_te_local_null, Y0_pr))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0
            g_m0_local = (d_prior_local - d_m0_local) / d_prior_local * 100.0
            
            # Full Real History
            H_tr_raw = np.vstack([s["X_be"] for s in tr])
            H_te_raw = te["X_be"]
            mu_h = np.mean(H_tr_raw, axis=0, keepdims=True)
            sig_h = np.std(H_tr_raw, axis=0, keepdims=True) + 1e-6
            H_tr = (H_tr_raw - mu_h) / sig_h
            H_te = (H_te_raw - mu_h) / sig_h
            
            XH_tr = np.hstack([X0_tr, H_tr])
            XH_te = np.hstack([X0_te, H_te])
            betaH = np.linalg.solve(XH_tr.T @ XH_tr + LAMBDA * np.eye(200, dtype=np.float32), XH_tr.T @ Y_tr)
            
            YH_pr = np.clip(XH_te @ betaH, 1e-12, None); YH_pr /= YH_pr.sum(axis=-1, keepdims=True)
            d_real = np.mean(js_divergence(Y_te, YH_pr))
            g_real = (d_prior - d_real) / d_prior * 100.0
            real_gains.append(g_real - g_m0)
            
            # Local Drift Null Evaluation
            d_loc = np.mean(js_divergence(Y_te_local_null, YH_pr))
            g_loc = (d_prior_local - d_loc) / d_prior_local * 100.0
            local_null_gains.append(g_loc - g_m0_local)
            
            # Global Circ Null Evaluation
            H_te_circ = (H_te_circ_null - mu_h) / sig_h
            XH_te_circ = np.hstack([X0_te, H_te_circ])
            Ycirc_pr = np.clip(XH_te_circ @ betaH, 1e-12, None); Ycirc_pr /= Ycirc_pr.sum(axis=-1, keepdims=True)
            d_circ = np.mean(js_divergence(Y_te, Ycirc_pr))
            g_circ = (d_prior - d_circ) / d_prior * 100.0
            global_null_gains.append(g_circ - g_m0)
            
        w_stat, p_loc = stats.wilcoxon(np.array(real_gains) - np.array(local_null_gains))
        
        results.append({
            "cohort": c,
            "mean_real": np.mean(real_gains),
            "sem_real": float(stats.sem(real_gains)),
            "mean_local_null": np.mean(local_null_gains),
            "sem_local_null": float(stats.sem(local_null_gains)),
            "mean_global_null": np.mean(global_null_gains),
            "sem_global_null": float(stats.sem(global_null_gains)),
            "wilcoxon_p": p_loc
        })
        
        print(f"Cohort {c:10s}:")
        print(f"  * Real History Gain      : {np.mean(real_gains):+5.2f}% ± {stats.sem(real_gains):4.2f}%")
        print(f"  * Local Drift Null (3-6m): {np.mean(local_null_gains):+5.2f}% ± {stats.sem(local_null_gains):4.2f}% (p = {p_loc:.4f})")
        print(f"  * Global Circ Null       : {np.mean(global_null_gains):+5.2f}% ± {stats.sem(global_null_gains):4.2f}%")
        
    df_null = pd.DataFrame(results)
    csv_null_path = os.path.join(TAB_DIR, "table_local_drift_temporal_null.csv")
    df_null.to_csv(csv_null_path, index=False)
    return df_null

# ---------------------------------------------------------------------------
# MODULE 4: STRICT ZERO-LEAKAGE FOLD-WISE SHARED NMF
# ---------------------------------------------------------------------------
def run_test4_strict_foldwise_shared_nmf(sessions):
    print("\n" + "="*80)
    print("TEST 4: STRICT ZERO-LEAKAGE FOLD-WISE SHARED NMF (K=12)")
    print("="*80)
    
    wt_sal = [s for s in sessions if s["cohort"] == "WT_Saline"]
    ki_sal = [s for s in sessions if s["cohort"] == "KI_Saline"]
    
    K_SHARED = 12
    records = []
    
    for c_name, c_sess, other_sess in [("KI_Saline", ki_sal, wt_sal), ("WT_Saline", wt_sal, ki_sal)]:
        n_mice = len(c_sess)
        
        for test_idx in range(n_mice):
            te = c_sess[test_idx]
            tr_same = [c_sess[j] for j in range(n_mice) if j != test_idx]
            tr_other = other_sess
            
            # Balanced pooling strictly on TRAINING mice:
            H_same_pool = np.vstack([s["X_be"] for s in tr_same])
            H_other_pool = np.vstack([s["X_be"] for s in tr_other])
            min_r = min(len(H_same_pool), len(H_other_pool))
            
            np.random.seed(test_idx * 100 + 42)
            idx_s = np.random.choice(len(H_same_pool), min_r, replace=False)
            idx_o = np.random.choice(len(H_other_pool), min_r, replace=False)
            H_tr_balanced = np.vstack([H_same_pool[idx_s], H_other_pool[idx_o]])
            
            # Fit NMF strictly on training mice!
            nmf_fold = NMF(n_components=K_SHARED, init="nndsvda", max_iter=400, random_state=42)
            nmf_fold.fit(H_tr_balanced)
            
            # Project training mice to compute component occupancy delta_k
            Z_same_tr = nmf_fold.transform(H_same_pool)
            Z_other_tr = nmf_fold.transform(H_other_pool)
            
            p_same = np.sum(Z_same_tr, axis=0) / (np.sum(Z_same_tr) + 1e-12)
            p_other = np.sum(Z_other_tr, axis=0) / (np.sum(Z_other_tr) + 1e-12)
            
            # Enrichment relative to other genotype in training data:
            delta_k = p_same - p_other
            idx_enr = np.where(delta_k > 0)[0]
            idx_sup = np.where(delta_k < 0)[0]
            
            # Transform held-out mouse strictly with fixed dictionary
            Z_tr_same_all = nmf_fold.transform(H_same_pool)
            Z_te = nmf_fold.transform(te["X_be"])
            
            # Standardize on training fold
            mu_z = np.mean(Z_tr_same_all, axis=0, keepdims=True)
            sig_z = np.std(Z_tr_same_all, axis=0, keepdims=True) + 1e-6
            Z_tr_std = (Z_tr_same_all - mu_z) / sig_z
            Z_te_std = (Z_te - mu_z) / sig_z
            
            # LOAO Prediction
            Y_tr = np.vstack([s["Y_dist"] for s in tr_same])
            Y_te = te["Y_dist"]
            p_prior = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_prior))
            
            X0_tr = np.vstack([s["s_curr"] for s in tr_same])
            X0_te = te["s_curr"]
            beta0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
            Y0_pr = np.clip(X0_te @ beta0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0
            
            # M_enriched
            XE_tr = np.hstack([X0_tr, Z_tr_std[:, idx_enr]])
            XE_te = np.hstack([X0_te, Z_te_std[:, idx_enr]])
            betaE = np.linalg.solve(XE_tr.T @ XE_tr + LAMBDA * np.eye(XE_tr.shape[1], dtype=np.float32), XE_tr.T @ Y_tr)
            YE_pr = np.clip(XE_te @ betaE, 1e-12, None); YE_pr /= YE_pr.sum(axis=-1, keepdims=True)
            d_enr = np.mean(js_divergence(Y_te, YE_pr))
            g_enr = (d_prior - d_enr) / d_prior * 100.0
            
            # M_suppressed
            XS_tr = np.hstack([X0_tr, Z_tr_std[:, idx_sup]])
            XS_te = np.hstack([X0_te, Z_te_std[:, idx_sup]])
            betaS = np.linalg.solve(XS_tr.T @ XS_tr + LAMBDA * np.eye(XS_tr.shape[1], dtype=np.float32), XS_tr.T @ Y_tr)
            YS_pr = np.clip(XS_te @ betaS, 1e-12, None); YS_pr /= YS_pr.sum(axis=-1, keepdims=True)
            d_sup = np.mean(js_divergence(Y_te, YS_pr))
            g_sup = (d_prior - d_sup) / d_prior * 100.0
            
            records.append({
                "cohort": c_name,
                "sid": te["sid"],
                "g_m0": g_m0,
                "gain_enriched": g_enr - g_m0,
                "gain_suppressed": g_sup - g_m0,
                "n_enriched": len(idx_enr),
                "n_suppressed": len(idx_sup)
            })
            
    df_strict = pd.DataFrame(records)
    csv_strict_path = os.path.join(TAB_DIR, "table_strict_foldwise_shared_nmf.csv")
    df_strict.to_csv(csv_strict_path, index=False)
    
    print("\n--- STRICT ZERO-LEAKAGE SHARED NMF RESULTS ---")
    for c in ["KI_Saline", "WT_Saline"]:
        sub = df_strict[df_strict["cohort"] == c]
        ge = sub["gain_enriched"].values
        gs = sub["gain_suppressed"].values
        w_s, p_v = stats.wilcoxon(ge - gs)
        print(f"Cohort {c:10s}:")
        print(f"  * Enriched Subspace Gain  : {np.mean(ge):+5.2f}% ± {stats.sem(ge):4.2f}%")
        print(f"  * Suppressed Subspace Gain: {np.mean(gs):+5.2f}% ± {stats.sem(gs):4.2f}%")
        print(f"  * Difference (Enr - Sup)  : {np.mean(ge) - np.mean(gs):+5.2f}% (Wilcoxon p = {p_v:.4f})")
        
    return df_strict

# ---------------------------------------------------------------------------
# PUBLICATION FIGURE GENERATION
# ---------------------------------------------------------------------------
def render_publication_figure(df_animals, stat_res, df_markov, df_null, df_strict):
    print("\nRendering Publication Figure: fig_definitive_story_discrimination_suite.png...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), dpi=300)
    
    # PANEL A: Direct Genotype x Coordinate Interaction
    ax1 = axes[0]
    wt_data = df_animals[df_animals["cohort"] == "WT_Saline"]
    ki_data = df_animals[df_animals["cohort"] == "KI_Saline"]
    
    # Contrasts to show:
    # 1. C_weight (Event Preference: Event - Dwell)
    # 2. C_boundary (Time Preference: Time - Bout)
    pos = [1, 2, 4, 5]
    bp = ax1.boxplot([wt_data["c_weight"], ki_data["c_weight"], wt_data["c_boundary"], ki_data["c_boundary"]],
                     positions=pos, widths=0.55, patch_artist=True,
                     boxprops=dict(facecolor="#e0e0e0", alpha=0.7),
                     medianprops=dict(color="black", linewidth=2.0))
                     
    # Overlay jittered animal points
    np.random.seed(42)
    ax1.scatter(np.random.normal(1, 0.05, len(wt_data)), wt_data["c_weight"], color=COLORS["WT_Saline"], s=55, alpha=0.9, zorder=4)
    ax1.scatter(np.random.normal(2, 0.05, len(ki_data)), ki_data["c_weight"], color=COLORS["KI_Saline"], s=55, alpha=0.9, zorder=4)
    ax1.scatter(np.random.normal(4, 0.05, len(wt_data)), wt_data["c_boundary"], color=COLORS["WT_Saline"], s=55, alpha=0.9, zorder=4)
    ax1.scatter(np.random.normal(5, 0.05, len(ki_data)), ki_data["c_boundary"], color=COLORS["KI_Saline"], s=55, alpha=0.9, zorder=4)
    
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_xticks([1.5, 4.5])
    ax1.set_xticklabels(["Weighting Contrast\n($C_{\\text{weight}}$: Event $-$ Dwell)",
                         "Boundary Contrast\n($C_{\\text{boundary}}$: Time $-$ Bout)"],
                        fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Representation Contrast (% Gain)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Genotype $\\times$ Coordinate Interaction\n(Direct Animal-Level Contrasts)", fontsize=12, fontweight="bold")
    
    # Annotate p-values
    p_w = stat_res["p_weight_perm"]
    p_b = stat_res["p_bound_perm"]
    ax1.annotate(f"Genotype Interaction\n$\\Delta = {stat_res['obs_diff_weight']:+.2f}\\%$\n(Perm p = {p_w:.4f})",
                 xy=(1.5, 4.2), xytext=(1.5, 5.0), ha="center", fontweight="bold", fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="green", alpha=0.85))
    ax1.annotate(f"Genotype Interaction\n$\\Delta = {stat_res['obs_diff_bound']:+.2f}\\%$\n(Perm p = {p_b:.4f})",
                 xy=(4.5, 1.8), xytext=(4.5, 3.2), ha="center", fontweight="bold", fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#cce5ff", ec="blue", alpha=0.85))
    ax1.grid(True, linestyle="--", alpha=0.35)
    
    # PANEL B: Local Markov Order Memory (S_{t+1})
    ax2 = axes[1]
    for c in ["WT_Saline", "KI_Saline"]:
        sub = df_markov[df_markov["cohort"] == c]
        # Gain over unigram
        ax2.plot(sub["order"], sub["gain_over_m0"], marker="o", color=COLORS[c], linewidth=2.4,
                 label=f"{COHORT_LABELS[c]} ($S_{{t+1}}$ gain)")
        ax2.fill_between(sub["order"], sub["gain_over_m0"] - sub["sem_gain"],
                         sub["gain_over_m0"] + sub["sem_gain"], color=COLORS[c], alpha=0.18)
                         
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xlabel("Markov Memory Order $m$ ($S_t, S_{t-1}, \\dots$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Local Next-Bout Gain (bits / bout)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Local Transition Memory ($S_{t+1}$)\n(Two-Scale Test: Local Syntax vs. Macro)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.35)
    
    # PANEL C: Local Drift-Preserving Temporal Null
    ax3 = axes[2]
    x_c = np.arange(2)
    w_bar = 0.26
    
    real_vals = df_null["mean_real"].values
    real_err = df_null["sem_real"].values
    loc_vals = df_null["mean_local_null"].values
    loc_err = df_null["sem_local_null"].values
    glob_vals = df_null["mean_global_null"].values
    glob_err = df_null["sem_global_null"].values
    
    ax3.bar(x_c - w_bar, real_vals, w_bar, yerr=real_err, label="Real History", color=[COLORS[c] for c in ["WT_Saline", "KI_Saline"]], alpha=0.9, edgecolor="black", capsize=3.5)
    ax3.bar(x_c, loc_vals, w_bar, yerr=loc_err, label="Local Null (3-6m gap)", color="#f0ad4e", alpha=0.85, edgecolor="black", capsize=3.5)
    ax3.bar(x_c + w_bar, glob_vals, w_bar, yerr=glob_err, label="Global Circ Null", color="#777777", hatch="//", alpha=0.75, edgecolor="black", capsize=3.5)
    
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(["WT Saline", "KI Saline"], fontsize=11, fontweight="bold")
    ax3.set_ylabel("Predictive Gain (% above $S_t$)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Local Drift-Preserving Null\n(Controls for Arousal & Session Phase)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9.0, framealpha=0.92)
    ax3.grid(True, linestyle="--", alpha=0.35)
    
    p_loc_ki = df_null[df_null["cohort"] == "KI_Saline"]["wilcoxon_p"].values[0]
    ax3.annotate(f"KI Real > Local Null\n(p = {p_loc_ki:.4f})", xy=(1, loc_vals[1]), xytext=(1 - 0.2, 5.2),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2)
                 
    # PANEL D: Strict Fold-Wise Shared NMF
    ax4 = axes[3]
    pos_d = [1, 2, 4, 5]
    wt_str = df_strict[df_strict["cohort"] == "WT_Saline"]
    ki_str = df_strict[df_strict["cohort"] == "KI_Saline"]
    
    ax4.boxplot([wt_str["gain_enriched"], wt_str["gain_suppressed"],
                 ki_str["gain_enriched"], ki_str["gain_suppressed"]],
                positions=pos_d, widths=0.55, patch_artist=True,
                boxprops=dict(facecolor="#e0e0e0", alpha=0.7),
                medianprops=dict(color="black", linewidth=2.0))
                
    np.random.seed(42)
    ax4.scatter(np.random.normal(1, 0.05, len(wt_str)), wt_str["gain_enriched"], color=COLORS["WT_Saline"], s=55, alpha=0.9, zorder=4)
    ax4.scatter(np.random.normal(2, 0.05, len(wt_str)), wt_str["gain_suppressed"], color="#999999", s=55, alpha=0.9, zorder=4)
    ax4.scatter(np.random.normal(4, 0.05, len(ki_str)), ki_str["gain_enriched"], color=COLORS["KI_Saline"], s=55, alpha=0.9, zorder=4)
    ax4.scatter(np.random.normal(5, 0.05, len(ki_str)), ki_str["gain_suppressed"], color="#999999", s=55, alpha=0.9, zorder=4)
    
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax4.set_xticks([1.5, 4.5])
    ax4.set_xticklabels(["WT Saline\n(Enriched vs. Suppressed)", "KI Saline\n(Enriched vs. Suppressed)"],
                        fontsize=10.5, fontweight="bold")
    ax4.set_ylabel("Out-of-Fold Predictive Gain (%)", fontsize=11, fontweight="bold")
    ax4.set_title("D. Strict Zero-Leakage Shared NMF\n(Enriched vs. Suppressed Subspaces)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.35)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_definitive_story_discrimination_suite.png")
    pdf_path = os.path.join(OUT_DIR, "fig_definitive_story_discrimination_suite.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_definitive_story_discrimination_suite.png"))
    print(f"Saved figure to:\n  -> {fig_path}\n  -> {pdf_path}")
    print(f"Copied to artifact dir: {os.path.join(ARTIFACT_DIR, 'fig_definitive_story_discrimination_suite.png')}")

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    sessions = build_matched_dataset()
    
    df_animals, stat_res = run_test1_genotype_coordinate_interaction(sessions)
    df_markov = run_test2_local_markov_order(sessions)
    df_null = run_test3_local_drift_null(sessions)
    df_strict = run_test4_strict_foldwise_shared_nmf(sessions)
    
    render_publication_figure(df_animals, stat_res, df_markov, df_null, df_strict)
    print(f"\nExecution completed successfully in {time.time() - t0:.1f}s.")
