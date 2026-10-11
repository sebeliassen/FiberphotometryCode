#!/usr/bin/env python3
"""
run_definitive_coordinate_system_suite.py
=============================================================================
DEFINITIVE BEHAVIORAL HISTORY COORDINATE SYSTEM SUITE:
Dissecting Physical Clock Time, Event Composition, Dwell Time, and Sequence Order

Evaluates on PERFECTLY MATCHED ANCHORS across all 30 mice (4 cohorts)
under strict within-cohort Leave-One-Animal-Out (LOAO) cross-validation:

1. MODULE 1: Matched Continuous History Curves
   - Clock Time: W in [2, 5, 10, 15, 20, 30, 60] seconds
   - Event Depth: n in [1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50] bouts
   Evaluated on the exact same anchors!

2. MODULE 2: The 2x2 Factorial Dissection
   - Window Boundary: Fixed 20 Bouts vs. Fixed 15 Seconds
   - Internal Weighting: Event Proportions vs. Duration Proportions
   All 4 cells strictly normalized on the probability simplex Delta^99.

3. MODULE 3: Shuffled-Order Surrogate Control
   - Chronological Recency Order (decay gamma=0.85) vs.
   - Shuffled Order Surrogate (permuted order pi over 20 permutations)
   Identical 100-D feature dimensionality, identical scale and ridge penalty!

Deliverables:
- Tables:
  * table_definitive_w_vs_n_curves.csv
  * table_definitive_2x2_factorial.csv
  * table_definitive_order_vs_shuffled.csv
- Figure: fig_definitive_coordinate_system_suite.png / .pdf
=============================================================================
"""

import os
import pickle
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
W_MAX_FRAMES = int(W_MAX_SEC * FPS) # ~4801 frames
N_MAX_BOUTS = 50
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS) # ~9603 frames
ANCHOR_STRIDE = 2     # Stride over bouts to balance speed & sample size
LAMBDA = 10.0         # Benchmark regularizer

W_GRID = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
N_GRID = [1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50]

COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "WT_Amphetamine": "WT Amphetamine",
    "KI_Saline": "KI Saline",
    "KI_Amphetamine": "KI Amphetamine"
}
COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#ff7f0e",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#9467bd"
}

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
# BUILD STRICTLY MATCHED OBSERVATION SESSIONS
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
        if cohort not in COHORT_ORDER:
            continue
        inj_f = meta.get("injection_frame", 0)
        sid = meta.get("subject_id", f"mouse_{i}")

        # Post-injection bouts only
        post_bouts = [b for b in bouts if b["start_frame"] > inj_f]
        if len(post_bouts) < N_MAX_BOUTS + 50:
            continue

        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int32)
        starts = np.array([b["start_frame"] for b in post_bouts], dtype=np.int64)
        ends = np.array([b["end_frame"] for b in post_bouts], dtype=np.int64)
        durs = ends - starts
        max_f = ends[-1]
        M = len(post_bouts)

        # Cumulative event counts over bouts
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0

        # Cumulative duration-weighted event counts over bouts
        cum_dur_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_dur_events[idx_b + 1] = cum_dur_events[idx_b]
            cum_dur_events[idx_b + 1, sylls[idx_b]] += float(durs[idx_b])

        # Cumulative frame counts over continuous time
        N_frames = max_f + 1
        cum_frame_sylls = np.zeros((N_frames, NUM_SYLLS), dtype=np.float32)
        for b in post_bouts:
            dur = b["end_frame"] - b["start_frame"]
            if dur > 0:
                cum_frame_sylls[b["start_frame"]:b["end_frame"], b["syllable"]] = 1.0
        cum_frame_sylls = np.cumsum(cum_frame_sylls, axis=0)

        # Find STRICT MATCHED ANCHORS
        # Every anchor satisfies:
        # - anc >= N_MAX_BOUTS (50 bouts)
        # - starts[anc] - W_MAX_FRAMES >= starts[0] (60 seconds)
        # - ends[anc] + H_FRAMES < N_frames (120 seconds future)
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
        K = len(valid_anchors)
        total_anchors += K

        # Target Y: future 120s syllable distribution (K, 100)
        t_ends = ends[valid_anchors]
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)

        # Baseline S_t (K, 100)
        s_curr = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(K), sylls[valid_anchors]] = 1.0

        # Precompute features for Module 1: Clock Time W_GRID
        clock_features = {}
        t_starts = starts[valid_anchors]
        for w in W_GRID:
            w_f = int(w * FPS)
            cnt = cum_frame_sylls[t_starts] - cum_frame_sylls[t_starts - w_f]
            clock_features[w] = cnt / (np.sum(cnt, axis=1, keepdims=True) + 1e-6)

        # Precompute features for Module 1: Event Depth N_GRID
        event_features = {}
        for n in N_GRID:
            cnt = cum_events[valid_anchors] - cum_events[valid_anchors - n]
            event_features[n] = cnt / float(n)

        # Precompute features for Module 2: 2x2 Factorial Matrix
        # Cell 1: 20 bouts, Event-Weighted
        cnt_b20_ev = cum_events[valid_anchors] - cum_events[valid_anchors - 20]
        feat_b20_ev = cnt_b20_ev / 20.0

        # Cell 2: 20 bouts, Duration-Weighted
        cnt_b20_dur = cum_dur_events[valid_anchors] - cum_dur_events[valid_anchors - 20]
        feat_b20_dur = cnt_b20_dur / (np.sum(cnt_b20_dur, axis=1, keepdims=True) + 1e-6)

        # Cell 3: 15 seconds, Event-Weighted (counts of bouts initiated in past 15s)
        w15_f = int(15.0 * FPS)
        feat_t15_ev = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        for k_idx, anc in enumerate(valid_anchors):
            t_win_st = starts[anc] - w15_f
            # bouts whose start_frame is in [t_win_st, starts[anc]]
            b_in_win = [post_bouts[j] for j in range(anc-1, -1, -1) if post_bouts[j]["start_frame"] >= t_win_st]
            if len(b_in_win) > 0:
                for b_w in b_in_win:
                    feat_t15_ev[k_idx, b_w["syllable"]] += 1.0
                feat_t15_ev[k_idx] /= float(len(b_in_win))
            else:
                feat_t15_ev[k_idx, sylls[anc]] = 1.0

        # Cell 4: 15 seconds, Duration-Weighted (continuous frame dwell in past 15s)
        cnt_t15_dur = cum_frame_sylls[t_starts] - cum_frame_sylls[t_starts - w15_f]
        feat_t15_dur = cnt_t15_dur / (np.sum(cnt_t15_dur, axis=1, keepdims=True) + 1e-6)

        # Precompute features for Module 3: Shuffled-Order vs. Chronological Recency
        # Chronological decay gamma = 0.85 over the 20 bouts
        gamma = 0.85
        weights_chrono = gamma ** np.arange(19, -1, -1) # oldest=0, newest=19
        weights_chrono /= np.sum(weights_chrono)

        feat_chrono_decay = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        for k_idx, anc in enumerate(valid_anchors):
            syll_seq = sylls[anc - 20:anc]
            for pos in range(20):
                feat_chrono_decay[k_idx, syll_seq[pos]] += weights_chrono[pos]

        # Shuffled surrogate: average over 20 random permutations of weights
        np.random.seed(42)
        feat_shuffled_decay = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        N_PERMS = 20
        perm_weights_list = [np.random.permutation(weights_chrono) for _ in range(N_PERMS)]
        for k_idx, anc in enumerate(valid_anchors):
            syll_seq = sylls[anc - 20:anc]
            for p_idx in range(N_PERMS):
                w_perm = perm_weights_list[p_idx]
                for pos in range(20):
                    feat_shuffled_decay[k_idx, syll_seq[pos]] += w_perm[pos] / float(N_PERMS)

        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "n_samples": K,
            "Y_dist": Y_dist,
            "s_curr": s_curr,
            "clock_features": clock_features,
            "event_features": event_features,
            "feat_b20_ev": feat_b20_ev,
            "feat_b20_dur": feat_b20_dur,
            "feat_t15_ev": feat_t15_ev,
            "feat_t15_dur": feat_t15_dur,
            "feat_chrono_decay": feat_chrono_decay,
            "feat_shuffled_decay": feat_shuffled_decay
        })

    print(f"Built {len(sessions)} valid sessions with {total_anchors} strictly matched observations.")
    return sessions

# ---------------------------------------------------------------------------
# RUN THE COMPLETE DEFINITIVE EXPERIMENT SUITE
# ---------------------------------------------------------------------------
def run_suite():
    sessions = build_matched_dataset()

    print("\n" + "="*80)
    print("EXECUTING DEFINITIVE COORDINATE SYSTEM SUITE (LOAO-CV)")
    print("="*80)

    # -----------------------------------------------------------------------
    # MODULE 1: Continuous Curves W vs n
    # -----------------------------------------------------------------------
    print("\n[MODULE 1] Computing Matched Curves: Delta_clock(W) vs. Delta_event(n)...")
    curve_records = []
    animal_curve_records = []

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)

        # 1. Baseline M0 (S_t only)
        m0_scores = {}
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
            m0_scores[test_idx] = {"d_prior": d_prior, "g_m0": g_m0}

        m0_mean = np.mean([m0_scores[i]["g_m0"] for i in range(n_mice)])

        # Sweep W (Clock Time)
        for w in W_GRID:
            deltas = []
            for test_idx in range(n_mice):
                te = c_sess[test_idx]
                tr = [c_sess[j] for j in range(n_mice) if j != test_idx]
                Y_tr = np.vstack([s["Y_dist"] for s in tr])
                Y_te = te["Y_dist"]
                d_prior = m0_scores[test_idx]["d_prior"]
                g_m0 = m0_scores[test_idx]["g_m0"]

                F_tr_raw = np.vstack([s["clock_features"][w] for s in tr])
                F_te_raw = te["clock_features"][w]
                mu = np.mean(F_tr_raw, axis=0, keepdims=True)
                sig = np.std(F_tr_raw, axis=0, keepdims=True) + 1e-6
                F_tr = (F_tr_raw - mu) / sig
                F_te = (F_te_raw - mu) / sig

                X_tr = np.hstack([np.vstack([s["s_curr"] for s in tr]), F_tr])
                X_te = np.hstack([te["s_curr"], F_te])

                beta = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(200, dtype=np.float32), X_tr.T @ Y_tr)
                Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_m = np.mean(js_divergence(Y_te, Y_pr))
                g_tot = (d_prior - d_m) / d_prior * 100.0
                delta = g_tot - g_m0
                deltas.append(delta)

                animal_curve_records.append({
                    "cohort": c, "sid": te["sid"], "domain": "Clock_Time_W",
                    "param_val": w, "delta_history": delta
                })

            curve_records.append({
                "cohort": c, "domain": "Clock_Time_W", "param_val": w,
                "mean_delta": float(np.mean(deltas)), "sem_delta": float(stats.sem(deltas))
            })

        # Sweep n (Event Depth)
        for n in N_GRID:
            deltas = []
            for test_idx in range(n_mice):
                te = c_sess[test_idx]
                tr = [c_sess[j] for j in range(n_mice) if j != test_idx]
                Y_tr = np.vstack([s["Y_dist"] for s in tr])
                Y_te = te["Y_dist"]
                d_prior = m0_scores[test_idx]["d_prior"]
                g_m0 = m0_scores[test_idx]["g_m0"]

                F_tr_raw = np.vstack([s["event_features"][n] for s in tr])
                F_te_raw = te["event_features"][n]
                mu = np.mean(F_tr_raw, axis=0, keepdims=True)
                sig = np.std(F_tr_raw, axis=0, keepdims=True) + 1e-6
                F_tr = (F_tr_raw - mu) / sig
                F_te = (F_te_raw - mu) / sig

                X_tr = np.hstack([np.vstack([s["s_curr"] for s in tr]), F_tr])
                X_te = np.hstack([te["s_curr"], F_te])

                beta = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(200, dtype=np.float32), X_tr.T @ Y_tr)
                Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_m = np.mean(js_divergence(Y_te, Y_pr))
                g_tot = (d_prior - d_m) / d_prior * 100.0
                delta = g_tot - g_m0
                deltas.append(delta)

                animal_curve_records.append({
                    "cohort": c, "sid": te["sid"], "domain": "Event_Depth_n",
                    "param_val": n, "delta_history": delta
                })

            curve_records.append({
                "cohort": c, "domain": "Event_Depth_n", "param_val": n,
                "mean_delta": float(np.mean(deltas)), "sem_delta": float(stats.sem(deltas))
            })

    df_curves = pd.DataFrame(curve_records)
    csv_curves_path = os.path.join(TAB_DIR, "table_definitive_w_vs_n_curves.csv")
    df_curves.to_csv(csv_curves_path, index=False)
    print(f"Saved: {csv_curves_path}")

    # -----------------------------------------------------------------------
    # MODULE 2: The 2x2 Factorial Matrix
    # -----------------------------------------------------------------------
    print("\n[MODULE 2] Computing 2x2 Factorial Matrix (Boundary x Weighting)...")
    factorial_cells = ["B20_Event", "B20_Duration", "T15_Event", "T15_Duration"]
    fact_records = []
    animal_fact_records = []

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)

        for cell in factorial_cells:
            deltas = []
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

                if cell == "B20_Event":
                    F_tr_raw = np.vstack([s["feat_b20_ev"] for s in tr])
                    F_te_raw = te["feat_b20_ev"]
                elif cell == "B20_Duration":
                    F_tr_raw = np.vstack([s["feat_b20_dur"] for s in tr])
                    F_te_raw = te["feat_b20_dur"]
                elif cell == "T15_Event":
                    F_tr_raw = np.vstack([s["feat_t15_ev"] for s in tr])
                    F_te_raw = te["feat_t15_ev"]
                elif cell == "T15_Duration":
                    F_tr_raw = np.vstack([s["feat_t15_dur"] for s in tr])
                    F_te_raw = te["feat_t15_dur"]

                mu = np.mean(F_tr_raw, axis=0, keepdims=True)
                sig = np.std(F_tr_raw, axis=0, keepdims=True) + 1e-6
                F_tr = (F_tr_raw - mu) / sig
                F_te = (F_te_raw - mu) / sig

                X_tr = np.hstack([X0_tr, F_tr])
                X_te = np.hstack([X0_te, F_te])

                beta = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(200, dtype=np.float32), X_tr.T @ Y_tr)
                Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_m = np.mean(js_divergence(Y_te, Y_pr))
                g_tot = (d_prior - d_m) / d_prior * 100.0
                delta = g_tot - g_m0
                deltas.append(delta)

                animal_fact_records.append({
                    "cohort": c, "sid": te["sid"], "cell": cell, "delta_history": delta
                })

            fact_records.append({
                "cohort": c, "cell": cell,
                "mean_delta": float(np.mean(deltas)), "sem_delta": float(stats.sem(deltas))
            })

    df_fact = pd.DataFrame(fact_records)
    csv_fact_path = os.path.join(TAB_DIR, "table_definitive_2x2_factorial.csv")
    df_fact.to_csv(csv_fact_path, index=False)
    print(f"Saved: {csv_fact_path}")

    # -----------------------------------------------------------------------
    # MODULE 3: Shuffled-Order Surrogate Control
    # -----------------------------------------------------------------------
    print("\n[MODULE 3] Computing Shuffled-Order Surrogate Control...")
    order_cells = ["Chronological_Recency", "Shuffled_Order_Surrogate"]
    order_records = []
    animal_order_records = []

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)

        for cell in order_cells:
            deltas = []
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

                if cell == "Chronological_Recency":
                    F_tr_raw = np.vstack([s["feat_chrono_decay"] for s in tr])
                    F_te_raw = te["feat_chrono_decay"]
                else:
                    F_tr_raw = np.vstack([s["feat_shuffled_decay"] for s in tr])
                    F_te_raw = te["feat_shuffled_decay"]

                mu = np.mean(F_tr_raw, axis=0, keepdims=True)
                sig = np.std(F_tr_raw, axis=0, keepdims=True) + 1e-6
                F_tr = (F_tr_raw - mu) / sig
                F_te = (F_te_raw - mu) / sig

                X_tr = np.hstack([X0_tr, F_tr])
                X_te = np.hstack([X0_te, F_te])

                beta = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(200, dtype=np.float32), X_tr.T @ Y_tr)
                Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_m = np.mean(js_divergence(Y_te, Y_pr))
                g_tot = (d_prior - d_m) / d_prior * 100.0
                delta = g_tot - g_m0
                deltas.append(delta)

                animal_order_records.append({
                    "cohort": c, "sid": te["sid"], "order_type": cell, "delta_history": delta
                })

            order_records.append({
                "cohort": c, "order_type": cell,
                "mean_delta": float(np.mean(deltas)), "sem_delta": float(stats.sem(deltas))
            })

    df_order = pd.DataFrame(order_records)
    csv_order_path = os.path.join(TAB_DIR, "table_definitive_order_vs_shuffled.csv")
    df_order.to_csv(csv_order_path, index=False)
    print(f"Saved: {csv_order_path}")

    # -----------------------------------------------------------------------
    # PRINT SUMMARY FINDINGS
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODULE 2: 2x2 FACTORIAL DISSECTION SUMMARY (MEAN ± SEM %)")
    print("="*80)
    print(df_fact.pivot(index="cohort", columns="cell", values="mean_delta").round(2))

    print("\n" + "="*80)
    print("MODULE 3: CHRONOLOGICAL VS. SHUFFLED SURROGATE SUMMARY (MEAN ± SEM %)")
    print("="*80)
    print(df_order.pivot(index="cohort", columns="order_type", values="mean_delta").round(2))

    # -----------------------------------------------------------------------
    # MULTI-PANEL PUBLICATION FIGURE
    # -----------------------------------------------------------------------
    print("\nRendering Publication Figure: fig_definitive_coordinate_system_suite.png...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # PANEL A: Matched Continuous Curves W vs n
    ax1 = axes[0]
    ax1_twin = ax1.twiny()

    for c in COHORT_ORDER:
        # Clock time curve W
        sub_w = df_curves[(df_curves["cohort"] == c) & (df_curves["domain"] == "Clock_Time_W")]
        ax1.plot(sub_w["param_val"], sub_w["mean_delta"], color=COLORS[c], linestyle="-",
                 marker="o", linewidth=2.0, label=f"{COHORT_LABELS[c]} (Clock $W$)")
        ax1.fill_between(sub_w["param_val"], sub_w["mean_delta"] - sub_w["sem_delta"],
                         sub_w["mean_delta"] + sub_w["sem_delta"], color=COLORS[c], alpha=0.12)

        # Event depth curve n
        sub_n = df_curves[(df_curves["cohort"] == c) & (df_curves["domain"] == "Event_Depth_n")]
        ax1_twin.plot(sub_n["param_val"], sub_n["mean_delta"], color=COLORS[c], linestyle="--",
                      marker="s", linewidth=2.0, alpha=0.85, label=f"{COHORT_LABELS[c]} (Event $n$)")

    ax1.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("Physical Clock-Time History $W$ (seconds) [Solid Lines]", fontsize=11, fontweight="bold")
    ax1_twin.set_xlabel("Behavioral Event-Depth History $n$ (bouts) [Dashed Lines]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Incremental History Gain $\\Delta_{\\text{hist}}$ (% above $S_t$)", fontsize=11.5, fontweight="bold")
    ax1.set_title("A. Continuous History Trajectories\n(Strictly Matched Observation Anchors)", fontsize=12.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Custom legend for Panel A
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", lw=2, label="Clock Time $W$ (s)"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", lw=2, label="Event Depth $n$ (bouts)"),
        Line2D([0], [0], color=COLORS["WT_Saline"], lw=3, label="WT Saline"),
        Line2D([0], [0], color=COLORS["WT_Amphetamine"], lw=3, label="WT Amphetamine"),
        Line2D([0], [0], color=COLORS["KI_Saline"], lw=3, label="KI Saline"),
        Line2D([0], [0], color=COLORS["KI_Amphetamine"], lw=3, label="KI Amphetamine"),
    ]
    ax1.legend(handles=custom_lines, loc="lower left", fontsize=8.8, framealpha=0.92)

    # PANEL B: 2x2 Factorial Dissection per Cohort
    ax2 = axes[1]
    cell_order = ["B20_Event", "B20_Duration", "T15_Event", "T15_Duration"]
    cell_names = ["20b (Event)", "20b (Dwell)", "15s (Event)", "15s (Dwell)"]
    cell_colors = ["#2ca02c", "#e377c2", "#17becf", "#1f77b4"]

    bar_width = 0.18
    n_cells = len(cell_order)
    for i, cell in enumerate(cell_order):
        vals = []
        errs = []
        for c in COHORT_ORDER:
            row = df_fact[(df_fact["cohort"] == c) & (df_fact["cell"] == cell)]
            vals.append(row["mean_delta"].values[0])
            errs.append(row["sem_delta"].values[0])

        x_pos = np.arange(len(COHORT_ORDER)) + (i - (n_cells - 1) / 2.0) * bar_width
        ax2.bar(x_pos, vals, bar_width, yerr=errs, capsize=3, label=cell_names[i],
                color=cell_colors[i], alpha=0.88, edgecolor="black", linewidth=0.8)

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xticks(range(len(COHORT_ORDER)))
    ax2.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Incremental History Gain $\\Delta_{\\text{hist}}$ (%)", fontsize=11.5, fontweight="bold")
    ax2.set_title("B. $2 \\times 2$ Factorial Dissection\n(Window Boundary $\\times$ Internal Weighting)", fontsize=12.5, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9.0, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # PANEL C: Shuffled-Order Surrogate Control
    ax3 = axes[2]
    bar_w = 0.32
    for i, o_type in enumerate(order_cells):
        vals = []
        errs = []
        for c in COHORT_ORDER:
            row = df_order[(df_order["cohort"] == c) & (df_order["order_type"] == o_type)]
            vals.append(row["mean_delta"].values[0])
            errs.append(row["sem_delta"].values[0])

        x_pos = np.arange(len(COHORT_ORDER)) + (i - 0.5) * bar_w
        lbl = "Chronological Recency" if i == 0 else "Shuffled-Order Surrogate"
        col = "#333333" if i == 0 else "#aaaaaa"
        ax3.bar(x_pos, vals, bar_w, yerr=errs, capsize=4, label=lbl,
                color=col, alpha=0.85, edgecolor="black", linewidth=0.9)

    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax3.set_xticks(range(len(COHORT_ORDER)))
    ax3.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=11, fontweight="bold")
    ax3.set_ylabel("Incremental History Gain $\\Delta_{\\text{hist}}$ (%)", fontsize=11.5, fontweight="bold")
    ax3.set_title("C. Sequence Order vs. Shuffled Surrogate\n(Identical 100-D Representation & Scale)", fontsize=12.5, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9.2, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_definitive_coordinate_system_suite.png")
    pdf_path = os.path.join(OUT_DIR, "fig_definitive_coordinate_system_suite.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_definitive_coordinate_system_suite.png"))
    print(f"Saved figure to:\n  -> {fig_path}\n  -> {pdf_path}")
    print(f"Copied to artifact dir: {os.path.join(ARTIFACT_DIR, 'fig_definitive_coordinate_system_suite.png')}")

if __name__ == "__main__":
    t0 = time.time()
    run_suite()
    print(f"\nExecution completed in {time.time() - t0:.1f}s.")
