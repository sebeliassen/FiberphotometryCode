#!/usr/bin/env python3
"""
===============================================================================
DIAGNOSTIC: Why does WT Saline gain appear to rise monotonically to +50.9%?
===============================================================================
Two issues:
  1. Case A trains on ALL WT Saline and tests on the SAME WT Saline data (train=test).
     This creates overfitting leakage — the model memorises session-specific noise.
  2. The within-cohort CV (Case B) used Ridge alpha=500, which may be
     inappropriate for the feature dimensionality.

This script runs a proper leave-one-session-out CV for EACH cohort at every W,
matching the methodology of the original presentation figure (which used 
GroupKFold n_splits=5 across ALL 30 sessions mixed together, with Ridge alpha=1-10).

We report:
  - Case 1: Train=Test (leaky, explains the +50.9%)
  - Case 2: Leave-One-Session-Out within cohort (proper, should match original ~30%)
  - Case 3: Pooled all-session GroupKFold(5) (the original presentation methodology)
===============================================================================
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)

FPS = 80.025
NUM_SYLLS = 100
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_FRAMES = int(5.0 * FPS)
MIN_BOUT_FRAMES = 3

W_VALUES = [2.0, 6.0, 10.0, 20.0, 30.0, 40.0, 68.3, 90.0, 120.0, 160.0, 200.0, 240.0, 300.0, 360.0]

def js_divergence(P, Q, eps=1e-12):
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)
    P = P / P.sum(axis=-1, keepdims=True); Q = Q / Q.sum(axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(P * np.log2(P / M), axis=-1) + 0.5 * np.sum(Q * np.log2(Q / M), axis=-1)

print("=" * 80)
print("DIAGNOSTIC: Overfitting Analysis — Why does WT Saline gain rise to +50.9%?")
print("=" * 80)

# ── Load data ──
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)
session_bouts = cache["session_bouts"] if isinstance(cache, dict) and "session_bouts" in cache else cache

rows = []
for s_idx, bouts in enumerate(session_bouts):
    s = raw_sessions[s_idx]
    meta = getattr(s, "metadata", {})
    gt, drug = meta.get("genotype", "WT"), meta.get("drug", "Saline")
    cohort = f"{gt}_{drug}"
    inj_f = meta.get("injection_frame", 0)

    filtered_bouts = [b for b in bouts if (b["end_frame"] - b["start_frame"] + 1) >= MIN_BOUT_FRAMES]
    if len(filtered_bouts) < 100: continue

    max_frame = filtered_bouts[-1]["end_frame"]
    N = max_frame + 1
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in filtered_bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf:ef+1] = s_id
        oh_bout[sf, s_id] = 1.0

    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

    for t in range(int(360 * FPS), N - H_FRAMES, STRIDE_FRAMES):
        is_post = (t >= inj_f + int(360 * FPS))
        if not is_post: continue

        curr_oh = np.zeros(NUM_SYLLS, dtype=np.float32)
        curr_oh[frame_sylls[t]] = 1.0

        past_w = {}
        for w_sec in W_VALUES:
            w_fr = int(w_sec * FPS)
            if t >= w_fr:
                c = cum_bout[t] - cum_bout[t - w_fr]
                s = c.sum()
                past_w[w_sec] = c / s if s > 0 else c
            else:
                past_w[w_sec] = np.zeros(NUM_SYLLS, dtype=np.float32)

        future_counts = cum_bout[min(t + H_FRAMES, N)] - cum_bout[t]
        fs = future_counts.sum()
        y_dist = future_counts / fs if fs > 0 else future_counts

        rows.append({"session_idx": s_idx, "cohort": cohort, "curr_syll_oh": curr_oh,
                      "past_w": past_w, "y_dist": y_dist})

df = pd.DataFrame(rows)
print(f"\nTotal post-injection evaluation windows: {len(df)}")
for c in ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
    n = len(df[df["cohort"] == c])
    nsess = df[df["cohort"] == c]["session_idx"].nunique()
    print(f"  {c}: {n} windows ({nsess} sessions)")

# ── Ridge via normal equations ──
def ridge_predict(X_train, Y_train, X_test, alpha=10.0):
    d = X_train.shape[1]
    I = np.eye(d, dtype=np.float32)
    beta = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
    return X_test @ beta

# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: WT Saline Only — Train=Test vs Leave-One-Session-Out
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST 1: WT Saline — Train=Test (LEAKY) vs Leave-One-Session-Out (PROPER)")
print("=" * 80)

wt_sal = df[df["cohort"] == "WT_Saline"]
Y_all = np.array(wt_sal["y_dist"].tolist())
sess_ids = wt_sal["session_idx"].values

print(f"\n{'W_sec':>6}  {'TrainEqTest':>12}  {'LOSO_CV':>12}  {'Overfit_Δ':>10}")
print("-" * 50)

for w_sec in W_VALUES:
    X = np.hstack([np.array(wt_sal["curr_syll_oh"].tolist()),
                   np.array([w[w_sec] for w in wt_sal["past_w"]])])

    # --- Train=Test (leaky) ---
    p0_all = np.mean(Y_all, axis=0, keepdims=True)
    d_null_all = np.mean(js_divergence(Y_all, p0_all))
    Y_pr_leaky = np.clip(ridge_predict(X, Y_all, X, alpha=500.0), 1e-12, None)
    Y_pr_leaky /= Y_pr_leaky.sum(axis=-1, keepdims=True)
    d_pr_leaky = np.mean(js_divergence(Y_all, Y_pr_leaky))
    gain_leaky = (d_null_all - d_pr_leaky) / d_null_all * 100.0

    # --- Leave-One-Session-Out (proper) ---
    unique_sess = np.unique(sess_ids)
    fold_gains = []
    for test_s in unique_sess:
        mask_te = (sess_ids == test_s)
        mask_tr = ~mask_te
        X_tr, Y_tr = X[mask_tr], Y_all[mask_tr]
        X_te, Y_te = X[mask_te], Y_all[mask_te]

        p0_te = np.mean(Y_te, axis=0, keepdims=True)
        d_null_te = np.mean(js_divergence(Y_te, p0_te))

        Y_pr = np.clip(ridge_predict(X_tr, Y_tr, X_te, alpha=10.0), 1e-12, None)
        Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
        d_pr = np.mean(js_divergence(Y_te, Y_pr))
        fold_gains.append((d_null_te - d_pr) / d_null_te * 100.0)

    gain_cv = np.mean(fold_gains)
    print(f"{w_sec:6.1f}  {gain_leaky:+12.2f}%  {gain_cv:+12.2f}%  {gain_leaky - gain_cv:+10.2f}%")

# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: Proper LOSO CV for ALL 4 cohorts
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST 2: Proper Leave-One-Session-Out CV for ALL 4 Cohorts (alpha=10)")
print("=" * 80)

cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

results = []
for w_sec in W_VALUES:
    row = {"W_sec": w_sec}
    for c in cohort_order:
        c_df = df[df["cohort"] == c]
        c_sess = c_df["session_idx"].values
        Y_c = np.array(c_df["y_dist"].tolist())
        X_c = np.hstack([np.array(c_df["curr_syll_oh"].tolist()),
                         np.array([w[w_sec] for w in c_df["past_w"]])])

        unique_sess = np.unique(c_sess)
        fold_gains = []
        for test_s in unique_sess:
            mask_te = (c_sess == test_s)
            mask_tr = ~mask_te
            X_tr, Y_tr = X_c[mask_tr], Y_c[mask_tr]
            X_te, Y_te = X_c[mask_te], Y_c[mask_te]
            if len(X_te) == 0 or len(X_tr) == 0: continue

            p0 = np.mean(Y_te, axis=0, keepdims=True)
            d_null = np.mean(js_divergence(Y_te, p0))
            if d_null < 1e-12: continue

            Y_pr = np.clip(ridge_predict(X_tr, Y_tr, X_te, alpha=10.0), 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))
            fold_gains.append((d_null - d_pr) / d_null * 100.0)

        row[c] = np.mean(fold_gains) if fold_gains else float('nan')
    results.append(row)

df_cv = pd.DataFrame(results)
print("\nPredictive Gain (% ΔD_JS) — Leave-One-Session-Out CV (alpha=10):")
print(df_cv.to_string(index=False, float_format="%.2f"))

# ── Find where each cohort peaks and tapers ──
print("\n" + "=" * 80)
print("PEAK & TAPER ANALYSIS (Proper LOSO CV)")
print("=" * 80)

for c in cohort_order:
    vals = df_cv[c].values
    w_vals = df_cv["W_sec"].values
    peak_idx = np.argmax(vals)
    peak_w = w_vals[peak_idx]
    peak_g = vals[peak_idx]

    # Find where gain first drops below 90% of peak
    taper_90 = None
    for i in range(peak_idx, len(vals)):
        if vals[i] < 0.9 * peak_g:
            taper_90 = w_vals[i]
            break

    # Find where gain crosses zero (if it does)
    zero_cross = None
    for i in range(len(vals) - 1):
        if vals[i] > 0 and vals[i+1] <= 0:
            # Linear interpolation
            zero_cross = w_vals[i] + (0 - vals[i]) / (vals[i+1] - vals[i]) * (w_vals[i+1] - w_vals[i])
            break

    print(f"\n{c}:")
    print(f"  Peak Gain: +{peak_g:.2f}% at W* = {peak_w:.1f}s")
    if taper_90 is not None:
        print(f"  Drops below 90% of peak at W = {taper_90:.1f}s")
    else:
        print(f"  Does NOT drop below 90% of peak within W ≤ 360s")
    if zero_cross is not None:
        print(f"  Crosses zero gain at W ≈ {zero_cross:.1f}s")
    else:
        print(f"  Does NOT cross zero within W ≤ 360s (final gain at 360s: {vals[-1]:+.2f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# TEST 3: Pooled All-30-Sessions GroupKFold(5) — Matches Original Presentation
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST 3: Pooled All-30-Sessions GroupKFold(5) — Original Presentation Method")
print("=" * 80)

Y_all_global = np.array(df["y_dist"].tolist())
sess_all_global = df["session_idx"].values
gkf = GroupKFold(n_splits=5)

pool_results = []
for w_sec in W_VALUES:
    X_all = np.hstack([np.array(df["curr_syll_oh"].tolist()),
                       np.array([w[w_sec] for w in df["past_w"]])])

    fold_gains = []
    for tr_idx, te_idx in gkf.split(X_all, groups=sess_all_global):
        X_tr, Y_tr = X_all[tr_idx], Y_all_global[tr_idx]
        X_te, Y_te = X_all[te_idx], Y_all_global[te_idx]

        p0 = np.mean(Y_te, axis=0, keepdims=True)
        d_null = np.mean(js_divergence(Y_te, p0))

        Y_pr = np.clip(ridge_predict(X_tr, Y_tr, X_te, alpha=10.0), 1e-12, None)
        Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
        d_pr = np.mean(js_divergence(Y_te, Y_pr))
        fold_gains.append((d_null - d_pr) / d_null * 100.0)

    pool_results.append({"W_sec": w_sec, "Pooled_All30": np.mean(fold_gains)})

df_pool = pd.DataFrame(pool_results)
print("\nPredictive Gain (Pooled All 30 Sessions, GroupKFold=5):")
for _, r in df_pool.iterrows():
    print(f"  W = {r['W_sec']:6.1f}s   Gain = {r['Pooled_All30']:+.2f}%")

peak_idx = df_pool["Pooled_All30"].idxmax()
print(f"\n  >>> Pooled peak: +{df_pool.loc[peak_idx, 'Pooled_All30']:.2f}% at W = {df_pool.loc[peak_idx, 'W_sec']:.1f}s")
print(f"  >>> Compare to original slide 4 CSV peak: +30.05% at W = 90.0s")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
