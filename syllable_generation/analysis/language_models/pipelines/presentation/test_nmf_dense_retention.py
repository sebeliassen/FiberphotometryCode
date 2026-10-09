#!/usr/bin/env python3
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)

from pipelines.presentation.render_4_cohort_figures import js_divergence, load_data

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")

def get_post_injection_data():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached post-injection dataframe from {CACHE_PATH}...")
        t0 = time.time()
        with open(CACHE_PATH, "rb") as f:
            df_post = pickle.load(f)
        print(f"Loaded in {time.time()-t0:.2f}s, shape: {df_post.shape}")
        return df_post

    print("Extracting post-injection data via load_data()...")
    df = load_data()
    df_post = df[df["is_post"]].copy()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(df_post, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(df_post)} post-injection rows to {CACHE_PATH}")
    return df_post

def run_retention_sweep():
    df_post = get_post_injection_data()
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

    wt_df = df_post[df_post["cohort"] == "WT_Saline"]
    wt_sess = wt_df["session_idx"].unique()
    wt_sess_arr = wt_df["session_idx"].values
    wt_X_past = np.array(wt_df["past_counts"].tolist(), dtype=np.float32)
    wt_curr = np.array(wt_df["curr_syll_oh"].tolist(), dtype=np.float32)
    wt_Y = np.array(wt_df["y_dist"].tolist(), dtype=np.float32)
    p_wt_prior = np.mean(wt_Y, axis=0, keepdims=True)

    # Pre-extract other cohort matrices
    other_cohorts = {}
    for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        c_df = df_post[df_post["cohort"] == c]
        other_cohorts[c] = {
            "past": np.array(c_df["past_counts"].tolist(), dtype=np.float32),
            "curr": np.array(c_df["curr_syll_oh"].tolist(), dtype=np.float32),
            "Y": np.array(c_df["y_dist"].tolist(), dtype=np.float32),
            "sess_arr": c_df["session_idx"].values,
            "sess_uniq": c_df["session_idx"].unique()
        }

    # 1. First, compute the UNCOMPRESSED 100-D Ceiling for all cohorts
    print("\n--- Computing Uncompressed 100-D Past Ceiling ---")
    # WT Saline LOSO uncompressed
    gains_wt_100d = []
    X_wt_100d = np.hstack([wt_curr, wt_X_past])
    for ts in wt_sess:
        m_te = (wt_sess_arr == ts)
        m_tr = ~m_te
        X_tr, Y_tr = X_wt_100d[m_tr], wt_Y[m_tr]
        X_te, Y_te = X_wt_100d[m_te], wt_Y[m_te]
        d = X_tr.shape[1]
        I = np.eye(d, dtype=np.float32)
        beta = np.linalg.solve(X_tr.T @ X_tr + 500.0 * I, X_tr.T @ Y_tr)
        Y_pr = np.clip(X_te @ beta, 1e-12, None)
        Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
        d_pr = np.mean(js_divergence(Y_te, Y_pr))
        p_tr = np.mean(Y_tr, axis=0, keepdims=True)
        d_prior = np.mean(js_divergence(Y_te, p_tr))
        gains_wt_100d.append((d_prior - d_pr) / d_prior * 100.0)

    ceil_100d = {"WT_Saline": np.mean(gains_wt_100d)}

    # Full WT model transfer to other cohorts
    d = X_wt_100d.shape[1]
    I = np.eye(d, dtype=np.float32)
    beta_full_100d = np.linalg.solve(X_wt_100d.T @ X_wt_100d + 500.0 * I, X_wt_100d.T @ wt_Y)

    for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        X_c = np.hstack([other_cohorts[c]["curr"], other_cohorts[c]["past"]])
        Y_c = other_cohorts[c]["Y"]
        Y_pr = np.clip(X_c @ beta_full_100d, 1e-12, None)
        Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
        d_pr = np.mean(js_divergence(Y_c, Y_pr))
        d_prior = np.mean(js_divergence(Y_c, p_wt_prior))
        ceil_100d[c] = (d_prior - d_pr) / d_prior * 100.0

    print("Uncompressed 100-D Ceilings (% gain vs prior):")
    for c in cohort_order:
        print(f"  {c:15s}: {ceil_100d[c]:.2f}%")

    # 2. Dense K-sweep: K = 1, 2, 3, ..., 30
    K_GRID = list(range(1, 31))
    print(f"\n--- Running Dense NMF K-Sweep for K=1..30 ({len(K_GRID)} values) ---")

    sweep_results = []
    t_sweep_start = time.time()

    for K in K_GRID:
        t0 = time.time()
        nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=400)

        # SPEEDUP 1: Fit and transform once per dataset!
        Z_wt = nmf.fit_transform(wt_X_past)
        Z_others = {c: nmf.transform(other_cohorts[c]["past"]) for c in other_cohorts}

        # WT Saline LOSO CV
        gains_wt = []
        X_wt_all = np.hstack([wt_curr, Z_wt])
        for ts in wt_sess:
            m_te = (wt_sess_arr == ts)
            m_tr = ~m_te
            X_tr, Y_tr = X_wt_all[m_tr], wt_Y[m_tr]
            X_te, Y_te = X_wt_all[m_te], wt_Y[m_te]

            d_feat = X_tr.shape[1]
            I = np.eye(d_feat, dtype=np.float32)
            beta = np.linalg.solve(X_tr.T @ X_tr + 500.0 * I, X_tr.T @ Y_tr)

            Y_pr = np.clip(X_te @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))

            p_tr = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_tr))
            gains_wt.append((d_prior - d_pr) / d_prior * 100.0)

        wt_gain = np.mean(gains_wt)

        # Full WT model transfer to other cohorts
        d_feat = X_wt_all.shape[1]
        I = np.eye(d_feat, dtype=np.float32)
        beta_full = np.linalg.solve(X_wt_all.T @ X_wt_all + 500.0 * I, X_wt_all.T @ wt_Y)

        row = {
            "K": K,
            "WT_Saline": wt_gain,
            "WT_Saline_pct_100d": wt_gain / ceil_100d["WT_Saline"] * 100.0
        }

        for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
            X_c = np.hstack([other_cohorts[c]["curr"], Z_others[c]])
            Y_c = other_cohorts[c]["Y"]
            Y_pr = np.clip(X_c @ beta_full, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_c, Y_pr))
            d_prior = np.mean(js_divergence(Y_c, p_wt_prior))
            c_gain = (d_prior - d_pr) / d_prior * 100.0
            row[c] = c_gain
            row[f"{c}_pct_100d"] = c_gain / ceil_100d[c] * 100.0

        sweep_results.append(row)
        dt = time.time() - t0
        print(f"  K={K:2d} done in {dt:.2f}s | WT_Sal: {wt_gain:.2f}% ({row['WT_Saline_pct_100d']:.1f}%) | "
              f"WT_Amph: {row['WT_Amphetamine']:.2f}% ({row['WT_Amphetamine_pct_100d']:.1f}%) | "
              f"KI_Sal: {row['KI_Saline']:.2f}% ({row['KI_Saline_pct_100d']:.1f}%) | "
              f"KI_Amph: {row['KI_Amphetamine']:.2f}% ({row['KI_Amphetamine_pct_100d']:.1f}%)")

    total_time = time.time() - t_sweep_start
    print(f"\nTotal sweep of 30 components took {total_time:.2f}s ({total_time/len(K_GRID):.2f}s/K)!")

    df_sweep = pd.DataFrame(sweep_results)

    # Also compute relative to K=30 ceiling
    for c in cohort_order:
        k30_val = df_sweep.loc[df_sweep["K"] == 30, c].values[0]
        df_sweep[f"{c}_pct_k30"] = df_sweep[c] / k30_val * 100.0

    out_csv = os.path.join(LM_DIR, "output", "presentation", "tables", "cohort_4group_dense_nmf_retention.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_sweep.to_csv(out_csv, index=False)
    print(f"Saved dense NMF results to: {out_csv}")

    return df_sweep, ceil_100d

if __name__ == "__main__":
    run_retention_sweep()
