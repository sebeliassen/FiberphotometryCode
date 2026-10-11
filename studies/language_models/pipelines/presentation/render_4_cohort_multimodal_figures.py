#!/usr/bin/env python3
"""
===============================================================================
4-Cohort Multimodal Timescale Curves & Benchmark (Slide 4 Identical Features)
===============================================================================
Confined strictly to POST-INJECTION and evaluated per group.
Uses the EXACT same multimodal feature hierarchy as Slide 4 (P_max):
  1. Time Regressors (t_norm, t_norm^2) -> 2 features
  2. Spatial Kinematics (Mean + Std of arena x, y, speed, elevation) -> 12 features
  3. Current Syllable One-Hot (S_t) -> 100 features
  4. Past Syllable Distribution BoW(W) -> 100 features
  5. 24-D TruncatedSVD Bigram Transition Syntax -> 24 features
Total: 238 Multimodal Features -> Future Syllable Budget H = 120s

Evaluation:
  - WT Saline: Leave-One-Session-Out Cross-Validation (Zero Data Leakage)
  - Transfer: Trained on WT Saline Post-Injection -> Evaluated on WT Amph, KI Sal, KI Amph
  - Metric: % Delta D_JS relative to blind marginal prior
===============================================================================
"""

import os
import pickle
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/acfd96c6-639a-4c71-be4a-46cc2b7a0386"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CURRENT_ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/86fe4323-a753-4391-895c-3fb29aaa4e41"
os.makedirs(CURRENT_ARTIFACT_DIR, exist_ok=True)

FPS = 80.025
NUM_SYLLS = 100
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
MIN_BOUT_FRAMES = 3

W_GRID = [2.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 68.3, 90.0, 120.0, 160.0, 200.0, 240.0, 300.0, 360.0]

COLORS = {
    "WT_Saline": "#1f77b4",        # Deep Blue
    "WT_Amphetamine": "#d95f02",   # Burnt Orange
    "KI_Saline": "#2ca02c",        # Green
    "KI_Amphetamine": "#7570b3"    # Purple
}
LABELS = {
    "WT_Saline": "WT + Saline (LOSO CV, N=8)",
    "WT_Amphetamine": "WT + Amphetamine (N=8)",
    "KI_Saline": "KI + Saline (N=7)",
    "KI_Amphetamine": "KI + Amphetamine (N=7)"
}

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.1

def js_divergence(P, Q, eps=1e-12):
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)
    P = P / P.sum(axis=-1, keepdims=True); Q = Q / Q.sum(axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(P * np.log2(P / M), axis=-1) + 0.5 * np.sum(Q * np.log2(Q / M), axis=-1)

def extract_multimodal_dataset():
    print("Loading sessions and continuous bout cache...")
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    session_bouts = cache["session_bouts"] if isinstance(cache, dict) and "session_bouts" in cache else cache

    print("Extracting multimodal features (Time + Spatial + Bigram Grammar + Syllables)...")
    samples = []
    w_max_fr = int(360.0 * FPS)

    for s_idx, bouts in enumerate(session_bouts):
        s = raw_sessions[s_idx]
        meta = getattr(s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        inj_f = meta.get("injection_frame", 0)

        filtered_bouts = [b for b in bouts if (b["end_frame"] - b["start_frame"] + 1) >= MIN_BOUT_FRAMES]
        if len(filtered_bouts) < 100: continue

        max_frame = filtered_bouts[-1]["end_frame"]
        N = max_frame + 1

        frame_sylls = np.zeros(N, dtype=np.int32)
        frame_spatial = np.zeros((N, 6), dtype=np.float32)
        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)

        for b in filtered_bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLS - 1)
            frame_sylls[sf : ef + 1] = s_id
            oh_bout[sf, s_id] = 1.0
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                frame_spatial[sf : ef + 1, :4] = vec[:4]
                frame_spatial[sf : ef + 1, 4] = vec[10]
                frame_spatial[sf : ef + 1, 5] = vec[12]

        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
        cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
        cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])

        # Confined strictly to post-injection
        for t in range(w_max_fr, N - H_FRAMES, STRIDE_FRAMES):
            if t < inj_f + w_max_fr: continue

            # Time feature: normalized post-injection clock
            t_norm = (t - inj_f) / float(N - inj_f)
            time_feat = np.array([t_norm, t_norm**2], dtype=np.float32)

            # Current syllable one-hot
            curr_oh = np.zeros(NUM_SYLLS, dtype=np.float32)
            curr_oh[frame_sylls[t]] = 1.0

            # Precompute past window features for all W
            past_feat_by_w = {}
            for w_sec in W_GRID:
                w_fr = int(w_sec * FPS)

                # Syllable distribution (BoW)
                c = cum_bout[t] - cum_bout[t - w_fr]
                s = c.sum()
                past_dist = c / s if s > 0 else c

                # Spatial kinematics (mean + std)
                sp_sum = cum_spatial[t] - cum_spatial[t - w_fr]
                sp_mean = sp_sum / float(w_fr)
                sp_sq_sum = cum_spatial_sq[t] - cum_spatial_sq[t - w_fr]
                sp_var = np.maximum(sp_sq_sum / float(w_fr) - sp_mean**2, 1e-8)
                sp_std = np.sqrt(sp_var)
                sp_feat = np.hstack([sp_mean, sp_std])

                past_feat_by_w[w_sec] = {
                    "past_dist": past_dist,
                    "spatial_feat": sp_feat
                }

            # Future action budget target
            f_counts = cum_bout[min(t + H_FRAMES, N)] - cum_bout[t]
            fs = f_counts.sum()
            y_dist = f_counts / fs if fs > 0 else f_counts

            # Transition bigram counts over 68.3s for SVD syntax
            w_syntax_fr = int(68.3 * FPS)
            seg_sylls = frame_sylls[t - w_syntax_fr : t]
            src = seg_sylls[:-1]; dst = seg_sylls[1:]
            trans_mat = np.zeros((NUM_SYLLS, NUM_SYLLS), dtype=np.float32)
            np.add.at(trans_mat, (src, dst), 1.0)
            r_sum = trans_mat.sum(axis=1, keepdims=True); r_sum[r_sum == 0] = 1.0
            trans_flat = (trans_mat / r_sum).flatten()

            samples.append({
                "session_idx": s_idx, "cohort": cohort, "t_frame": t,
                "time_feat": time_feat, "curr_oh": curr_oh, "trans_flat": trans_flat,
                "past_by_w": past_feat_by_w, "y_dist": y_dist
            })

    print(f"Total post-injection samples extracted: {len(samples)}")

    # 24-D TruncatedSVD for bigram syntax (Identical to Slide 4)
    print("Fitting 24-D TruncatedSVD on bigram syntax...")
    X_trans = np.array([s["trans_flat"] for s in samples], dtype=np.float32)
    svd = TruncatedSVD(n_components=24, random_state=42)
    latent_24d = svd.fit_transform(X_trans)
    latent_24d = normalize(latent_24d, norm="l2")

    for i, s in enumerate(samples):
        s["latent_24d"] = latent_24d[i]
        del s["trans_flat"]

    return samples

def run_multimodal_evaluation(samples):
    print("\nRunning Multimodal Predictive Capacity across 4 Cohorts + Pooled Baseline...")
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

    all_sess_arr = np.array([s["session_idx"] for s in samples])
    all_cohort_arr = np.array([s["cohort"] for s in samples])
    all_y_arr = np.array([s["y_dist"] for s in samples])

    results = []

    for w_sec in W_GRID:
        row = {"W_sec": w_sec}

        # Build multimodal feature matrix: [Time(2) + Spatial(12) + CurrSyll(100) + PastBoW(100) + Latent24(24)] = 238-D
        all_X = np.array([
            np.hstack([s["time_feat"], s["past_by_w"][w_sec]["spatial_feat"], s["curr_oh"], s["past_by_w"][w_sec]["past_dist"], s["latent_24d"]])
            for s in samples
        ], dtype=np.float32)

        # ---------------------------------------------------------------------
        # 1. POOLED POST-INJECTION BASELINE (All 30 Sessions, 5-Fold GroupKFold CV)
        # ---------------------------------------------------------------------
        gkf = GroupKFold(n_splits=5)
        gains_pooled = []
        for tr_idx, te_idx in gkf.split(all_X, all_y_arr, groups=all_sess_arr):
            X_tr, Y_tr = all_X[tr_idx], all_y_arr[tr_idx]
            X_te, Y_te = all_X[te_idx], all_y_arr[te_idx]
            d = X_tr.shape[1]
            I = np.eye(d, dtype=np.float32)
            beta = np.linalg.solve(X_tr.T @ X_tr + 10.0 * I, X_tr.T @ Y_tr)
            Y_pr = np.clip(X_te @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))
            p_tr = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_tr))
            gains_pooled.append((d_prior - d_pr) / d_prior * 100.0)

        row["Pooled_Post_Injection"] = np.mean(gains_pooled)

        # ---------------------------------------------------------------------
        # 2. WT Saline Leave-One-Session-Out Cross-Validation (ZERO data leakage)
        # ---------------------------------------------------------------------
        wt_mask = (all_cohort_arr == "WT_Saline")
        wt_X = all_X[wt_mask]
        wt_y_arr = all_y_arr[wt_mask]
        wt_sess_arr = all_sess_arr[wt_mask]
        wt_sess = np.unique(wt_sess_arr)

        gains_wt = []
        for ts in wt_sess:
            m_te = (wt_sess_arr == ts)
            m_tr = ~m_te
            X_tr, Y_tr = wt_X[m_tr], wt_y_arr[m_tr]
            X_te, Y_te = wt_X[m_te], wt_y_arr[m_te]

            d = X_tr.shape[1]
            I = np.eye(d, dtype=np.float32)
            beta = np.linalg.solve(X_tr.T @ X_tr + 10.0 * I, X_tr.T @ Y_tr)

            Y_pr = np.clip(X_te @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))

            p_tr = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_tr))
            gains_wt.append((d_prior - d_pr) / d_prior * 100.0)

        row["WT_Saline"] = np.mean(gains_wt)
        row["WT_Saline_within"] = row["WT_Saline"]

        # ---------------------------------------------------------------------
        # 3. Within-Cohort LOSO for the other 3 cohorts
        # ---------------------------------------------------------------------
        for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
            c_mask = (all_cohort_arr == c)
            c_X = all_X[c_mask]
            c_Y = all_y_arr[c_mask]
            c_sess = all_sess_arr[c_mask]
            uniq_sess = np.unique(c_sess)

            gains_c = []
            for ts in uniq_sess:
                m_te = (c_sess == ts)
                m_tr = ~m_te
                X_tr, Y_tr = c_X[m_tr], c_Y[m_tr]
                X_te, Y_te = c_X[m_te], c_Y[m_te]

                d = X_tr.shape[1]
                I = np.eye(d, dtype=np.float32)
                beta = np.linalg.solve(X_tr.T @ X_tr + 10.0 * I, X_tr.T @ Y_tr)

                Y_pr = np.clip(X_te @ beta, 1e-12, None)
                Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(Y_te, Y_pr))

                p_tr = np.mean(Y_tr, axis=0, keepdims=True)
                d_null = np.mean(js_divergence(Y_te, p_tr))
                gains_c.append((d_null - d_pr) / d_null * 100.0)

            row[f"{c}_within"] = np.mean(gains_c)

        # ---------------------------------------------------------------------
        # 4. Transfer Model trained on full WT Saline -> Evaluated on other 3 cohorts
        # ---------------------------------------------------------------------
        d = wt_X.shape[1]
        I = np.eye(d, dtype=np.float32)
        beta_full = np.linalg.solve(wt_X.T @ wt_X + 10.0 * I, wt_X.T @ wt_y_arr)
        p_wt_prior = np.mean(wt_y_arr, axis=0, keepdims=True)

        for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
            c_mask = (all_cohort_arr == c)
            X_c = all_X[c_mask]
            Y_c = all_y_arr[c_mask]

            Y_pr = np.clip(X_c @ beta_full, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_c, Y_pr))

            d_prior = np.mean(js_divergence(Y_c, p_wt_prior))
            row[f"{c}_transfer"] = (d_prior - d_pr) / d_prior * 100.0
            row[c] = row[f"{c}_transfer"]  # primary column for continuity

        results.append(row)

    df_res = pd.DataFrame(results)
    print("\n=== Multimodal Predictive Capacity Table (% Gain vs Prior) ===")
    display_cols = ["W_sec", "Pooled_Post_Injection", "WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    print(df_res[display_cols].to_string(index=False, float_format="%.2f"))

    # Save table
    csv_path = os.path.join(LM_DIR, "output", "presentation", "tables", "cohort_4group_multimodal_timescale.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved results table to: {csv_path}")

    return df_res

def render_multimodal_figure(df_res):
    print("\nRendering Publication-Grade Multimodal Comparison Figure...")
    w_vals = df_res["W_sec"].values

    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=300)
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    markers = ["o", "s", "^", "D"]

    # 1. Plot Pooled Post-Injection Baseline (Gray dashed reference trace)
    g_pooled = df_res["Pooled_Post_Injection"].values
    ax.plot(w_vals, g_pooled, color="#333333", linestyle="--", linewidth=2.4,
            label="Pooled Baseline (All 30 Sessions, Post-Inj, 5-Fold CV)", zorder=2)
    ax.scatter(w_vals, g_pooled, color="#333333", marker="x", s=45, linewidth=1.5, zorder=3)

    # 2. Plot 4 Cohorts
    for c, m in zip(cohort_order, markers):
        g_vals = df_res[c].values
        ax.plot(w_vals, g_vals, color=COLORS[c], linewidth=2.5, label=LABELS[c], zorder=4)
        ax.scatter(w_vals, g_vals, color=COLORS[c], marker=m, s=55, edgecolors="white", linewidth=0.9, zorder=5)

    # Reference vertical line for operational window
    ax.axvline(68.3, color="#888888", linestyle=":", linewidth=1.5, zorder=2, label="Operational Window ($W^* = 68.3$s)")

    # Peak values
    p_wt_sal = df_res["WT_Saline"].max()
    w_wt_sal = df_res.loc[df_res["WT_Saline"].idxmax(), "W_sec"]
    p_wt_amph = df_res["WT_Amphetamine"].max()
    w_wt_amph = df_res.loc[df_res["WT_Amphetamine"].idxmax(), "W_sec"]
    p_ki_sal = df_res["KI_Saline"].max()
    w_ki_sal = df_res.loc[df_res["KI_Saline"].idxmax(), "W_sec"]
    p_ki_amph = df_res["KI_Amphetamine"].max()
    w_ki_amph = df_res.loc[df_res["KI_Amphetamine"].idxmax(), "W_sec"]
    p_pool = df_res["Pooled_Post_Injection"].max()
    w_pool = df_res.loc[df_res["Pooled_Post_Injection"].idxmax(), "W_sec"]

    # In-bounds clean annotations
    ax.annotate(f"Pooled Baseline ({w_pool:.0f}s, +{p_pool:.1f}%)",
                xy=(w_pool, p_pool), xytext=(210, 35.0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5", edgecolor="#666666", alpha=0.92))

    ax.annotate(f"WT Saline Peak ({w_wt_sal:.0f}s, +{p_wt_sal:.1f}%)\nTapers to +{df_res['WT_Saline'].iloc[-1]:.1f}%",
                xy=(w_wt_sal, p_wt_sal), xytext=(w_wt_sal + 40, p_wt_sal - 4.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.2),
                fontsize=8.5, fontweight="bold", color=COLORS["WT_Saline"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e1f5fe", edgecolor=COLORS["WT_Saline"], alpha=0.92))

    ax.annotate(f"WT Amph Peak ({w_wt_amph:.0f}s, +{p_wt_amph:.1f}%)\nTapers to +{df_res['WT_Amphetamine'].iloc[-1]:.1f}%",
                xy=(w_wt_amph, p_wt_amph), xytext=(w_wt_amph + 30, p_wt_amph - 3.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Amphetamine"], lw=1.2),
                fontsize=8.5, fontweight="bold", color=COLORS["WT_Amphetamine"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor=COLORS["WT_Amphetamine"], alpha=0.92))

    ax.annotate(f"KI Amph Peak ({w_ki_amph:.0f}s, +{p_ki_amph:.1f}%)\nSharp drop to +{df_res['KI_Amphetamine'].iloc[-1]:.1f}%",
                xy=(w_ki_amph, p_ki_amph), xytext=(w_ki_amph + 25, p_ki_amph + 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Amphetamine"], lw=1.2),
                fontsize=8.5, fontweight="bold", color=COLORS["KI_Amphetamine"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e5f5", edgecolor=COLORS["KI_Amphetamine"], alpha=0.92))

    # Add 2x2 Factorial Summary Inset Box in upper right
    table_text = (
        "Timescale Hierarchy (W*):\n"
        "─────────────────────────────\n"
        f" • WT + Saline : {w_wt_sal:4.0f}s (+{p_wt_sal:.1f}%)\n"
        f" • WT + Amph   : {w_wt_amph:4.0f}s (+{p_wt_amph:.1f}%)\n"
        f" • KI + Saline : {w_ki_sal:4.0f}s (+{p_ki_sal:.1f}%)\n"
        f" • KI + Amph   : {w_ki_amph:4.0f}s (+{p_ki_amph:.1f}%)\n"
        f" • Pooled Base : {w_pool:4.0f}s (+{p_pool:.1f}%)"
    )
    ax.text(0.97, 0.73, table_text, transform=ax.transAxes, fontsize=8.8,
            verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa", edgecolor="#cccccc", alpha=0.95))

    ax.set_xlim(-5, 380)
    all_vals = df_res[cohort_order + ["Pooled_Post_Injection"]]
    y_min = np.floor(all_vals.min().min() - 2)
    y_max = np.ceil(all_vals.max().max() + 4)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Past Memory History Horizon $W$ (Seconds)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$ vs. Reference Prior)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("Behavioral Memory Capacity Curves: 4 Cohorts vs. Pooled Baseline\n(Post-Injection Period, 3-Frame Bout Quality Filter, Slide 4 Multimodal Features)",
                 fontsize=12.5, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d7de", fontsize=8.8)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig2_4cohort_timescale_memory_curves.png")
    pdf_path = os.path.join(OUT_DIR, "fig2_4cohort_timescale_memory_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    # Also save as specific multimodal filename & copy to artifact directories
    shutil.copy(png_path, os.path.join(OUT_DIR, "fig2_4cohort_multimodal_timescale_curves.png"))
    shutil.copy(pdf_path, os.path.join(OUT_DIR, "fig2_4cohort_multimodal_timescale_curves.pdf"))
    shutil.copy(png_path, os.path.join(ARTIFACT_DIR, "fig2_4cohort_timescale_memory_curves.png"))
    shutil.copy(png_path, os.path.join(CURRENT_ARTIFACT_DIR, "fig2_4cohort_timescale_memory_curves.png"))
    print(f"  [SAVED] {png_path}")
    print(f"  [SAVED] {pdf_path}")
    print(f"  [COPIED TO ARTIFACTS] {os.path.join(CURRENT_ARTIFACT_DIR, 'fig2_4cohort_timescale_memory_curves.png')}")

def main():
    samples = extract_multimodal_dataset()
    df_res = run_multimodal_evaluation(samples)
    render_multimodal_figure(df_res)
    print("\n[COMPLETE] 4-cohort multimodal timescale comparison finished successfully.")

if __name__ == "__main__":
    main()
