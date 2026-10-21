#!/usr/bin/env python3
"""
===============================================================================
Extended NMF Dimensional Retention Sweep up to K=100
Evaluates dimensionality saturation to determine exactly where each cohort hits 100%
===============================================================================
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

matplotlib.use('Agg')
import shutil

import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)

from fpcode.statistics.divergence import js_divergence

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TABLES_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

ARTIFACT_DIR_CURRENT = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
os.makedirs(ARTIFACT_DIR_CURRENT, exist_ok=True)

COLORS = {
    "WT_Saline": "#1f77b4",        # Deep Blue
    "WT_Amphetamine": "#d95f02",   # Burnt Orange
    "KI_Saline": "#2ca02c",        # Green
    "KI_Amphetamine": "#7570b3"    # Purple
}
LABELS = {
    "WT_Saline": "WT + Saline (N=8)",
    "WT_Amphetamine": "WT + Amphetamine (N=8)",
    "KI_Saline": "KI + Saline (N=7)",
    "KI_Amphetamine": "KI + Amphetamine (N=7)"
}
MARKERS = {
    "WT_Saline": "o",
    "WT_Amphetamine": "s",
    "KI_Saline": "^",
    "KI_Amphetamine": "D"
}

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.1

def run_extended_sweep():
    print(f"Loading cached post-injection dataframe from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        df_post = pickle.load(f)

    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

    wt_df = df_post[df_post["cohort"] == "WT_Saline"]
    wt_sess = wt_df["session_idx"].unique()
    wt_sess_arr = wt_df["session_idx"].values
    wt_X_past = np.array(wt_df["past_counts"].tolist(), dtype=np.float32)
    wt_curr = np.array(wt_df["curr_syll_oh"].tolist(), dtype=np.float32)
    wt_Y = np.array(wt_df["y_dist"].tolist(), dtype=np.float32)
    p_wt_prior = np.mean(wt_Y, axis=0, keepdims=True)

    other_cohorts = {}
    for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        c_df = df_post[df_post["cohort"] == c]
        other_cohorts[c] = {
            "past": np.array(c_df["past_counts"].tolist(), dtype=np.float32),
            "curr": np.array(c_df["curr_syll_oh"].tolist(), dtype=np.float32),
            "Y": np.array(c_df["y_dist"].tolist(), dtype=np.float32)
        }

    # 1. Uncompressed 100-D Ceiling
    print("Computing uncompressed 100-D past ceiling...")
    X_wt_100d = np.hstack([wt_curr, wt_X_past])
    gains_wt_100d = []
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

    print("Ceilings (Uncompressed 100-D):")
    for c, v in ceil_100d.items():
        print(f"  {c:16s}: {v:.2f}%")

    # Extended K Grid: 1..30 (all ints), then 35..100 (steps of 5)
    K_GRID = list(range(1, 31)) + [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    print(f"\nRunning Extended K-Sweep for {len(K_GRID)} values up to K=100...")

    results = []
    t_start = time.time()

    for K in K_GRID:
        t0 = time.time()
        max_it = 150 if K > 50 else 250
        nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=max_it)

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
            d = X_tr.shape[1]
            I = np.eye(d, dtype=np.float32)
            beta = np.linalg.solve(X_tr.T @ X_tr + 500.0 * I, X_tr.T @ Y_tr)
            Y_pr = np.clip(X_te @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))
            p_tr = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_tr))
            gains_wt.append((d_prior - d_pr) / d_prior * 100.0)

        wt_gain = np.mean(gains_wt)

        # Full WT model transfer to other 3 cohorts
        d = X_wt_all.shape[1]
        I = np.eye(d, dtype=np.float32)
        beta_full = np.linalg.solve(X_wt_all.T @ X_wt_all + 500.0 * I, X_wt_all.T @ wt_Y)

        row = {"K": K, "WT_Saline": wt_gain}

        for c in ["WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
            X_c = np.hstack([other_cohorts[c]["curr"], Z_others[c]])
            Y_c = other_cohorts[c]["Y"]
            Y_pr = np.clip(X_c @ beta_full, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_c, Y_pr))
            d_prior = np.mean(js_divergence(Y_c, p_wt_prior))
            row[c] = (d_prior - d_pr) / d_prior * 100.0

        results.append(row)
        dt = time.time() - t0
        print(f"  K={K:3d} ({dt:4.1f}s) | WT_Sal: {row['WT_Saline']:5.2f}% | WT_Amph: {row['WT_Amphetamine']:5.2f}% | "
              f"KI_Sal: {row['KI_Saline']:5.2f}% | KI_Amph: {row['KI_Amphetamine']:5.2f}%")

    print(f"\nExtended sweep completed in {time.time()-t_start:.1f}s!")

    df_ext = pd.DataFrame(results)

    # 1. Normalization relative to Uncompressed 100-D Ceiling
    for c in cohort_order:
        df_ext[f"{c}_pct_100d"] = df_ext[c] / ceil_100d[c] * 100.0

    # 2. Normalization relative to each cohort's OWN Empirical Maximum across the K sweep
    for c in cohort_order:
        c_max = df_ext[c].max()
        df_ext[f"{c}_pct_own_max"] = df_ext[c] / c_max * 100.0
        print(f"Cohort {c:16s}: Empirical Peak = {c_max:.2f}% (at K={df_ext.loc[df_ext[c].idxmax(), 'K']})")

    csv_path = os.path.join(TABLES_DIR, "cohort_4group_extended_nmf_retention_k100.csv")
    df_ext.to_csv(csv_path, index=False)
    print(f"Saved extended table to: {csv_path}")

    return df_ext, ceil_100d

def render_extended_figures(df, ceil_100d):
    print("\nRendering Extended K=100 Publication Figures...")
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    k_vals = df["K"].values

    # Saturation metrics for both normalizations
    k90_100d = {}
    k95_100d = {}
    k100_100d = {}
    for c in cohort_order:
        pct = df[f"{c}_pct_100d"].values
        k90_100d[c] = k_vals[np.where(pct >= 90.0)[0][0]] if np.any(pct >= 90.0) else ">100"
        k95_100d[c] = k_vals[np.where(pct >= 95.0)[0][0]] if np.any(pct >= 95.0) else ">100"
        k100_100d[c] = k_vals[np.where(pct >= 99.5)[0][0]] if np.any(pct >= 99.5) else ">100"

    k90_own = {}
    k95_own = {}
    k100_own = {}
    for c in cohort_order:
        pct = df[f"{c}_pct_own_max"].values
        k90_own[c] = k_vals[np.where(pct >= 90.0)[0][0]] if np.any(pct >= 90.0) else ">100"
        k95_own[c] = k_vals[np.where(pct >= 95.0)[0][0]] if np.any(pct >= 95.0) else ">100"
        k100_own[c] = k_vals[np.where(pct >= 99.5)[0][0]] if np.any(pct >= 99.5) else ">100"

    # =========================================================================
    # FIGURE 1: Primary Extended Curve (% of Uncompressed 100-D Ceiling)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=300)

    ax.axhline(100.0, color="#555555", linestyle=":", linewidth=1.5, zorder=1, label="Uncompressed 100-D Past Ceiling (100%)")
    ax.axhline(90.0, color="#888888", linestyle="--", linewidth=1.1, zorder=1, label="90% Informational Saturation Threshold")

    enhanced_labels_100d = {
        "WT_Saline": f"WT + Saline (N=8)       [Peak: 89.2% at K=26 | K=4: {df.loc[df['K']==4, 'WT_Saline_pct_100d'].values[0]:.1f}%]",
        "WT_Amphetamine": f"WT + Amph (N=8)         [K₉₀:  14 | Peak: 91.2% | K=4: {df.loc[df['K']==4, 'WT_Amphetamine_pct_100d'].values[0]:.1f}%]",
        "KI_Saline": f"KI + Saline (N=7)       [K₉₀:   2 | Peak: 97.7% | K=4: {df.loc[df['K']==4, 'KI_Saline_pct_100d'].values[0]:.1f}%]",
        "KI_Amphetamine": f"KI + Amph (N=7)         [K₉₀:   1 | Hits 100%: K=25-30 | K=4: {df.loc[df['K']==4, 'KI_Amphetamine_pct_100d'].values[0]:.1f}%]"
    }

    for c in cohort_order:
        pct_vals = df[f"{c}_pct_100d"].values
        ax.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.4, label=enhanced_labels_100d[c], zorder=3)
        ax.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=35, edgecolors="white", linewidth=0.7, zorder=4)

    ax.axvline(4, color="#d95f02", linestyle="--", linewidth=1.6, zorder=2, label="Candidate Macro Scale ($K = 4$)")

    # Annotations at key landmarks
    ax.annotate("KI Amph: Reaches 100% at K=25-30\n(98.5% captured at K=4)",
                xy=(4, 98.51), xytext=(8, 103.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Amphetamine"], lw=1.2),
                fontsize=8.3, fontweight="bold", color=COLORS["KI_Amphetamine"],
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#f3e5f5", edgecolor=COLORS["KI_Amphetamine"], alpha=0.92))

    # Move KI Saline callout below the curve (y=91.5..94) so it doesn't touch the green points
    ax.annotate("KI Saline: 95.2% at K=4\n(Saturates at ~97% by K=25)",
                xy=(4, 95.20), xytext=(8, 92.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Saline"], lw=1.2),
                fontsize=8.3, fontweight="bold", color=COLORS["KI_Saline"],
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#e8f5e9", edgecolor=COLORS["KI_Saline"], alpha=0.92))

    # WT Saline callout positioned cleanly at lower left
    ax.annotate("WT Saline: Plateaus at 89.2%\n(Requires full 100-D basis)",
                xy=(4, 82.57), xytext=(1.5, 75.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.2),
                fontsize=8.2, fontweight="bold", color=COLORS["WT_Saline"],
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#e1f5fe", edgecolor=COLORS["WT_Saline"], alpha=0.92))

    # Overfitting callout at upper-lower right (x=76..98, y=78..81) with plenty of room
    ax.annotate("High-Rank Overfitting (K>85)\n(WT source noise harms KI)",
                xy=(90, 72.0), xytext=(72, 78.5),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
                fontsize=8.2, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#fff9c4", edgecolor="#fbc02d", alpha=0.95))

    ax.set_xlim(-1, 103)
    ax.set_ylim(68, 107)
    ax.set_xticks([1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Retention (% of Full 100-D Past Ceiling)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("Extended NMF Memory Retention to Full Rank ($K = 1 \\dots 100$)\n(Normalized % of Full History Repertoire Across 4 Cohorts)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    # Position legend in lower-center between WT callout and Overfitting callout
    ax.legend(loc="lower center", bbox_to_anchor=(0.43, 0.03), frameon=True, facecolor="white", framealpha=0.96, edgecolor="#d0d7de",
              fontsize=8.0, prop={'family': 'monospace', 'size': 7.7},
              title="Dimensional Saturation Hierarchy:", title_fontsize=8.5)

    plt.tight_layout()
    png_k100 = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_extended_retention_k100.png")
    pdf_k100 = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_extended_retention_k100.pdf")
    plt.savefig(png_k100, dpi=300)
    plt.savefig(pdf_k100)
    plt.close()
    shutil.copy(png_k100, os.path.join(ARTIFACT_DIR_CURRENT, "fig1b_4cohort_nmf_extended_retention_k100.png"))
    print(f"  [SAVED] {png_k100}")

    # =========================================================================
    # FIGURE 2: DUAL COMPARISON: % of 100-D Ceiling vs. % of Cohort's OWN Peak
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 6.4), dpi=300)

    # Panel A: Relative to Uncompressed 100-D Past Repertoire
    ax1.axhline(100.0, color="#555555", linestyle=":", linewidth=1.4, zorder=1, label="100-D Ceiling (100%)")
    ax1.axhline(90.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1, label="90% Threshold")

    for c in cohort_order:
        pct_vals = df[f"{c}_pct_100d"].values
        ax1.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.3, label=enhanced_labels_100d[c], zorder=3)
        ax1.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=30, edgecolors="white", linewidth=0.6, zorder=4)

    ax1.axvline(4, color="#d95f02", linestyle="--", linewidth=1.5, zorder=2, label="Macro Scale ($K=4$)")
    ax1.set_xlim(-1, 103)
    ax1.set_ylim(68, 107)
    ax1.set_xticks([1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax1.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Predictive Retention (% of Full 100-D Past Ceiling)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax1.set_title("A. Retention vs. Full 100-D History Ceiling", fontsize=12.5, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d0d7de",
               fontsize=7.8, prop={'family': 'monospace', 'size': 7.5},
               title="Saturation vs 100-D Ceiling:", title_fontsize=8.2)

    # Panel B: Relative to each cohort's OWN Empirical NMF Maximum (Hits 100% exactly at its own peak)
    ax2.axhline(100.0, color="#555555", linestyle=":", linewidth=1.4, zorder=1, label="Cohort's Own NMF Peak (100%)")
    ax2.axhline(90.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1, label="90% Saturation")
    ax2.axhline(95.0, color="#aaaaaa", linestyle="-.", linewidth=0.9, zorder=1, label="95% Saturation")

    enhanced_labels_own = {
        "WT_Saline": f"WT + Saline   [Peak: 11.9% at K=26 | K₉₅:  8 | K=4: {df.loc[df['K']==4, 'WT_Saline_pct_own_max'].values[0]:.1f}%]",
        "WT_Amphetamine": f"WT + Amph     [Peak: 16.5% at K=24 | K₉₅:  7 | K=4: {df.loc[df['K']==4, 'WT_Amphetamine_pct_own_max'].values[0]:.1f}%]",
        "KI_Saline": f"KI + Saline   [Peak: 21.3% at K=80 | K₉₅:  4 | K=4: {df.loc[df['K']==4, 'KI_Saline_pct_own_max'].values[0]:.1f}%]",
        "KI_Amphetamine": f"KI + Amph     [Peak: 20.4% at K=75 | K₉₅:  4 | K=4: {df.loc[df['K']==4, 'KI_Amphetamine_pct_own_max'].values[0]:.1f}%]"
    }

    for c in cohort_order:
        pct_vals = df[f"{c}_pct_own_max"].values
        ax2.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.3, label=enhanced_labels_own[c], zorder=3)
        ax2.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=30, edgecolors="white", linewidth=0.6, zorder=4)

    ax2.axvline(4, color="#d95f02", linestyle="--", linewidth=1.5, zorder=2, label="Macro Scale ($K=4$)")
    ax2.set_xlim(-1, 103)
    ax2.set_ylim(68, 103)
    ax2.set_xticks([1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Predictive Retention (% of Group's Own NMF Peak)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax2.set_title("B. Retention vs. Group's Own Empirical Peak ($100\\% \\equiv \\max_K$)", fontsize=12.5, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d0d7de",
               fontsize=7.8, prop={'family': 'monospace', 'size': 7.5},
               title="Saturation vs Own Peak:", title_fontsize=8.2)

    plt.tight_layout()
    dual_k100_png = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_dual_norm_k100.png")
    dual_k100_pdf = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_dual_norm_k100.pdf")
    plt.savefig(dual_k100_png, dpi=300)
    plt.savefig(dual_k100_pdf)
    plt.close()
    shutil.copy(dual_k100_png, os.path.join(ARTIFACT_DIR_CURRENT, "fig1b_4cohort_nmf_dual_norm_k100.png"))
    print(f"  [SAVED] {dual_k100_png}")

def main():
    csv_path = os.path.join(TABLES_DIR, "cohort_4group_extended_nmf_retention_k100.csv")
    if os.path.exists(csv_path):
        print(f"Loading precomputed extended table from {csv_path}...")
        df_ext = pd.read_csv(csv_path)
        ceil_100d = {
            "WT_Saline": 13.2941,
            "WT_Amphetamine": 18.0694,
            "KI_Saline": 21.8407,
            "KI_Amphetamine": 19.8105
        }
    else:
        df_ext, ceil_100d = run_extended_sweep()
    render_extended_figures(df_ext, ceil_100d)
    print("\n[COMPLETE] Extended NMF retention figures updated successfully.")

if __name__ == "__main__":
    main()
