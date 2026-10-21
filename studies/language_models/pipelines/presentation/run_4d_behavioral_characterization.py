#!/usr/bin/env python3
"""
Module 2: The 4-Dimensional Behavioral Characterization Matrix
Computes the 4 distinct behavioral dimensions for each of the N=30 individual mice:
  1. Repertoire Complexity: Model-Free Participation Ratio (PR) rank & Entropy rank
  2. Predictability: Total predictive gain (% Delta D_JS vs. within-cohort prior)
  3. Memory Dependence: Incremental gain of 68s history strictly beyond current syllable (Delta_hist = M_raw - M_0)
  4. Timescale: Autocorrelation half-life tau (in seconds) of syllable transitions

Evaluated strictly within-cohort under Leave-One-Animal-Out cross-validation.
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
from fpcode.statistics.divergence import js_divergence

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")
BOUT_CACHE = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#d95f02",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#7570b3"
}

def compute_4d_matrix():
    print("="*75)
    print("COMPUTING 4-DIMENSIONAL BEHAVIORAL MATRIX (N=30 INDIVIDUAL ANIMALS)")
    print("="*75)

    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)

    cohorts = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    results = []

    LAMBDA = 10.0

    for c in cohorts:
        cdf = df[df["cohort"] == c]
        unique_sessions = cdf["session_idx"].unique()

        # Extract matrices for this cohort
        W_OPT_SEC = 20.0 # Optimal biological memory window (replaces 68.3s pooled artifact)
        S_curr = np.array(cdf["curr_syll_oh"].tolist(), dtype=np.float32)
        X_past = np.array([w[W_OPT_SEC] for w in cdf["past_w"]], dtype=np.float32)
        Y_dist = np.array(cdf["y_dist"].tolist(), dtype=np.float32)
        sess_ids = cdf["session_idx"].values
        sids = cdf["sid"].values

        for s in unique_sessions:
            m_te = (sess_ids == s)
            m_tr = ~m_te
            sid = sids[m_te][0]

            # -------------------------------------------------------------
            # 1. COMPLEXITY: Syllable Repertoire Intrinsic Dimensionality
            # -------------------------------------------------------------
            X_mouse = X_past[m_te]
            C_mouse = np.cov(X_mouse, rowvar=False)
            evals = np.linalg.eigvalsh(C_mouse)
            evals = np.sort(evals)[::-1]
            evals = np.maximum(evals, 1e-10)

            # Participation Ratio Rank
            pr_rank = float((evals.sum())**2 / (evals**2).sum())

            # Entropy Rank
            p = evals / evals.sum()
            ent_rank = float(np.exp(-np.sum(p * np.log(p))))

            # Syllable usage entropy (100-D distribution)
            mean_counts = X_mouse.mean(axis=0)
            p_syll = mean_counts / (mean_counts.sum() + 1e-12)
            p_syll_nz = p_syll[p_syll > 0]
            syll_entropy = float(-np.sum(p_syll_nz * np.log2(p_syll_nz)))

            # -------------------------------------------------------------
            # 2. PREDICTABILITY & 3. MEMORY (Within-Cohort LOAO-CV)
            # -------------------------------------------------------------
            # Cohort-specific prior (fit on other mice in same cohort!)
            p_tr_cohort = np.mean(Y_dist[m_tr], axis=0, keepdims=True)
            d_prior = float(np.mean(js_divergence(Y_dist[m_te], p_tr_cohort)))

            # Standardize past counts using training mice only
            mu_tr = np.mean(X_past[m_tr], axis=0, keepdims=True)
            sig_tr = np.std(X_past[m_tr], axis=0, keepdims=True) + 1e-6
            X_past_tr = (X_past[m_tr] - mu_tr) / sig_tr
            X_past_te = (X_past[m_te] - mu_tr) / sig_tr

            # M_0: Current Syllable only
            X0_tr, X0_te = S_curr[m_tr], S_curr[m_te]
            beta_0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(X0_tr.shape[1], dtype=np.float32), X0_tr.T @ Y_dist[m_tr])
            Y0_pr = np.clip(X0_te @ beta_0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = float(np.mean(js_divergence(Y_dist[m_te], Y0_pr)))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0

            # M_raw: Current Syllable + 20s History
            Xraw_tr = np.hstack([S_curr[m_tr], X_past_tr])
            Xraw_te = np.hstack([S_curr[m_te], X_past_te])
            beta_raw = np.linalg.solve(Xraw_tr.T @ Xraw_tr + LAMBDA * np.eye(Xraw_tr.shape[1], dtype=np.float32), Xraw_tr.T @ Y_dist[m_tr])
            Yraw_pr = np.clip(Xraw_te @ beta_raw, 1e-12, None); Yraw_pr /= Yraw_pr.sum(axis=-1, keepdims=True)
            d_mraw = float(np.mean(js_divergence(Y_dist[m_te], Yraw_pr)))
            g_mraw = (d_prior - d_mraw) / d_prior * 100.0

            delta_history = g_mraw - g_m0

            # -------------------------------------------------------------
            # 4. TIMESCALE: Autocorrelation decay of past counts
            # -------------------------------------------------------------
            # Compute lag-1 to lag-30 autocorrelation of dominant PCA component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            X_pc1 = pca.fit_transform(X_mouse)[:, 0]
            X_pc1 -= np.mean(X_pc1)
            norm_var = np.var(X_pc1) + 1e-12

            lags = np.arange(1, 25)
            autocorrs = []
            for lag in lags:
                if len(X_pc1) > lag:
                    ac = np.mean(X_pc1[:-lag] * X_pc1[lag:]) / norm_var
                    autocorrs.append(ac)
                else:
                    autocorrs.append(0.0)
            autocorrs = np.array(autocorrs)

            # Each stride is 5.0 seconds
            time_lags = lags * 5.0

            # Fit exponential decay: ac(t) = exp(-t / tau)
            def exp_decay(t, tau):
                return np.exp(-t / tau)

            try:
                popt, _ = curve_fit(exp_decay, time_lags, np.maximum(autocorrs, 0.0), p0=[25.0], bounds=(1.0, 300.0))
                tau_sec = float(popt[0])
            except Exception:
                tau_sec = np.nan

            results.append({
                "session_idx": s,
                "sid": sid,
                "cohort": c,
                "pr_rank": pr_rank,
                "entropy_rank": ent_rank,
                "syllable_entropy_bits": syll_entropy,
                "predictability_full_pct": g_mraw,
                "markov_m0_pct": g_m0,
                "delta_history_pct": delta_history,
                "history_to_markov_ratio": delta_history / max(g_m0, 1e-4),
                "autocorr_tau_sec": tau_sec
            })

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(TAB_DIR, "table_4d_behavioral_matrix_animals.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Cohort summary statistics
    summary = df_res.groupby("cohort").agg({
        "pr_rank": ["mean", "sem"],
        "entropy_rank": ["mean", "sem"],
        "syllable_entropy_bits": ["mean", "sem"],
        "predictability_full_pct": ["mean", "sem"],
        "markov_m0_pct": ["mean", "sem"],
        "delta_history_pct": ["mean", "sem"],
        "autocorr_tau_sec": ["mean", "sem"]
    })
    print("\n--- 4-DIMENSIONAL BEHAVIORAL MATRIX SUMMARY (MEAN ± SEM) ---")
    print(summary)

    # Statistical comparison (One-Way ANOVA across 4 cohorts)
    from scipy import stats
    print("\n--- STATISTICAL TESTS (ONE-WAY ANOVA) ---")
    for col in ["pr_rank", "entropy_rank", "syllable_entropy_bits", "predictability_full_pct", "markov_m0_pct", "delta_history_pct", "autocorr_tau_sec"]:
        groups = [df_res[df_res["cohort"] == c][col].dropna().values for c in cohorts]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"{col:25s}: F = {f_stat:6.2f}, p = {p_val:.2e}")

    # Plot 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 5.0), dpi=300)

    metrics = [
        ("pr_rank", "A. Repertoire Complexity\n(Participation Ratio Rank)", "PR Effective Rank"),
        ("predictability_full_pct", "B. Behavioral Predictability\n(Total Gain % $\\Delta D_{\\text{JS}}$)", "Predictability (% vs. Prior)"),
        ("delta_history_pct", "C. Non-Markovian Memory ($W=20\\text{s}$)\n($\\Delta_{\\text{history}} = M_{\\text{raw}} - M_0$)", "History Gain (% above $S_t$)"),
        ("autocorr_tau_sec", "D. Behavioral Timescale\n(Autocorrelation Half-Life $\\tau$)", "Persistence Timescale $\\tau$ (s)")
    ]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for i, c in enumerate(cohorts):
            vals = df_res[df_res["cohort"] == c][col].dropna().values
            x_pos = np.random.normal(i, 0.08, size=len(vals))
            # Boxplot
            ax.boxplot(vals, positions=[i], widths=0.45, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[c], alpha=0.3, edgecolor=COLORS[c], linewidth=1.5),
                       medianprops=dict(color="black", linewidth=2.0),
                       whiskerprops=dict(color=COLORS[c], linewidth=1.2),
                       capprops=dict(color=COLORS[c], linewidth=1.2),
                       showfliers=False)
            # Individual animal points
            ax.scatter(x_pos, vals, color=COLORS[c], s=50, edgecolors="white", linewidth=0.8, zorder=5, alpha=0.9)

        ax.set_xticks(range(4))
        ax.set_xticklabels([c.replace("_", "\n") for c in cohorts], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, axis="y")

    plt.tight_layout()
    fig_png = os.path.join(OUT_DIR, "fig_4d_behavioral_matrix.png")
    fig_pdf = os.path.join(OUT_DIR, "fig_4d_behavioral_matrix.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"\nSaved figures: {fig_png} and {fig_pdf}")
    import shutil
    shutil.copy(fig_png, os.path.join(ARTIFACT_DIR, "fig_4d_behavioral_matrix.png"))

if __name__ == "__main__":
    t0 = time.time()
    compute_4d_matrix()
    print(f"Completed in {time.time() - t0:.1f}s")
