#!/usr/bin/env python3
"""
Module 1: Empirical Dissection of the Group-Identity Confound
Evaluates whether past history in pooled multi-cohort models acts
partially as a static cohort/group classifier rather than dynamic memory.

Models evaluated under Leave-One-Animal-Out Cross-Validation on pooled data (N=30 mice):
  M_0:  Y ~ S_t                         (Instantaneous syllable only)
  M_G:  Y ~ S_t + Cohort_ID             (Posture + static cohort one-hot)
  M_H:  Y ~ S_t + X_past                (Posture + 68s history)
  M_GH: Y ~ S_t + Cohort_ID + X_past    (Posture + cohort + 68s history)
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
from fpcode.statistics.divergence import js_divergence

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

COHORTS = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
COHORT_MAP = {c: i for i, c in enumerate(COHORTS)}

def load_data():
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    return df

def run_group_confound_dissection():
    print("="*75)
    print("RUNNING MODULE 1: GROUP-IDENTITY CONFOUND DISSECTION")
    print("="*75)

    df = load_data()
    unique_sessions = df["session_idx"].unique()
    n_animals = len(unique_sessions)
    print(f"Total samples: {len(df)} across N={n_animals} animals in 4 cohorts.")

    # Pre-extract arrays
    S_curr = np.array(df["curr_syll_oh"].tolist(), dtype=np.float32)
    X_past = np.array(df["past_counts"].tolist(), dtype=np.float32)
    Y_dist = np.array(df["y_dist"].tolist(), dtype=np.float32)
    sess_ids = df["session_idx"].values
    cohort_strs = df["cohort"].values

    # One-hot encoding of cohort (4-D)
    G_onehot = np.zeros((len(df), len(COHORTS)), dtype=np.float32)
    for i, c in enumerate(cohort_strs):
        G_onehot[i, COHORT_MAP[c]] = 1.0

    # Store predictions per animal
    scores = {
        "M0": [],       # S_t only
        "M_G": [],      # S_t + Cohort
        "M_H": [],      # S_t + History
        "M_GH": [],     # S_t + Cohort + History
        "session": [],
        "cohort": []
    }

    LAMBDA = 10.0

    for s in unique_sessions:
        m_te = (sess_ids == s)
        m_tr = ~m_te
        c_test = cohort_strs[m_te][0]

        # Test marginal reference prior (fit on training only)
        p_tr = np.mean(Y_dist[m_tr], axis=0, keepdims=True)
        d_prior = np.mean(js_divergence(Y_dist[m_te], p_tr))

        # Standardize past counts on training split
        mu_tr = np.mean(X_past[m_tr], axis=0, keepdims=True)
        sig_tr = np.std(X_past[m_tr], axis=0, keepdims=True) + 1e-6
        X_past_tr = (X_past[m_tr] - mu_tr) / sig_tr
        X_past_te = (X_past[m_te] - mu_tr) / sig_tr

        # Feature sets
        F_M0_tr, F_M0_te = S_curr[m_tr], S_curr[m_te]
        F_MG_tr, F_MG_te = np.hstack([S_curr[m_tr], G_onehot[m_tr]]), np.hstack([S_curr[m_te], G_onehot[m_te]])
        F_MH_tr, F_MH_te = np.hstack([S_curr[m_tr], X_past_tr]), np.hstack([S_curr[m_te], X_past_te])
        F_MGH_tr, F_MGH_te = np.hstack([S_curr[m_tr], G_onehot[m_tr], X_past_tr]), np.hstack([S_curr[m_te], G_onehot[m_te], X_past_te])

        def solve_and_eval(X_t, X_v):
            d = X_t.shape[1]
            I = np.eye(d, dtype=np.float32)
            beta = np.linalg.solve(X_t.T @ X_t + LAMBDA * I, X_t.T @ Y_dist[m_tr])
            Y_pr = np.clip(X_v @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_dist[m_te], Y_pr))
            return (d_prior - d_pr) / d_prior * 100.0

        g_m0 = solve_and_eval(F_M0_tr, F_M0_te)
        g_mg = solve_and_eval(F_MG_tr, F_MG_te)
        g_mh = solve_and_eval(F_MH_tr, F_MH_te)
        g_mgh = solve_and_eval(F_MGH_tr, F_MGH_te)

        scores["M0"].append(g_m0)
        scores["M_G"].append(g_mg)
        scores["M_H"].append(g_mh)
        scores["M_GH"].append(g_mgh)
        scores["session"].append(s)
        scores["cohort"].append(c_test)

    df_scores = pd.DataFrame(scores)
    df_scores["apparent_hist_gain"] = df_scores["M_H"] - df_scores["M0"]
    df_scores["adjusted_hist_gain"] = df_scores["M_GH"] - df_scores["M_G"]
    df_scores["group_information"] = df_scores["M_G"] - df_scores["M0"]

    # Save table
    tab_path = os.path.join(TAB_DIR, "table_group_confound_dissection.csv")
    df_scores.to_csv(tab_path, index=False)
    print(f"Saved results table: {tab_path}")

    # Print summary
    print("\n--- OVERALL POOLED RESULTS (N=30 ANIMALS, LOAO-CV) ---")
    print(f"M_0 (Current Syllable S_t only):          {df_scores['M0'].mean():.2f}%")
    print(f"M_G (S_t + Ground Truth Cohort ID):       {df_scores['M_G'].mean():.2f}% (Group gain: +{df_scores['group_information'].mean():.2f}%)")
    print(f"M_H (S_t + 68s History):                  {df_scores['M_H'].mean():.2f}% (Apparent history gain: +{df_scores['apparent_hist_gain'].mean():.2f}%)")
    print(f"M_GH (S_t + Cohort ID + 68s History):     {df_scores['M_GH'].mean():.2f}% (Adjusted history gain: +{df_scores['adjusted_hist_gain'].mean():.2f}%)")

    confound_pct = (1.0 - (df_scores['adjusted_hist_gain'].mean() / df_scores['apparent_hist_gain'].mean())) * 100.0
    print(f"\n>> QUANTIFICATION: {confound_pct:.1f}% of apparent history gain in pooled data is explained by static cohort identity!")

    print("\n--- RESULTS BY COHORT ---")
    cohort_summary = df_scores.groupby("cohort")[["M0", "M_G", "M_H", "M_GH", "apparent_hist_gain", "adjusted_hist_gain"]].mean()
    print(cohort_summary)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Panel 1: Bar chart comparing models
    model_labels = ["$M_0$\n($S_t$ only)", "$M_G$\n($S_t$ + Group)", "$M_H$\n($S_t$ + History)", "$M_{GH}$\n($S_t$ + Group + Hist)"]
    means = [df_scores["M0"].mean(), df_scores["M_G"].mean(), df_scores["M_H"].mean(), df_scores["M_GH"].mean()]
    sems = [df_scores[m].sem() for m in ["M0", "M_G", "M_H", "M_GH"]]
    bar_colors = ["#7f7f7f", "#9467bd", "#1f77b4", "#2ca02c"]

    bars = ax1.bar(model_labels, means, yerr=sems, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for b, m in zip(bars, means):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{m:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax1.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$ vs. Prior)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Dissection of Pooled Predictive Gain\n(Leave-One-Animal-Out CV, N=30 Mice)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax1.set_ylim(0, max(means) * 1.25)

    # Panel 2: Apparent vs Adjusted History Gain by Cohort
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    x = np.arange(len(cohort_order))
    width = 0.35

    app_vals = [df_scores[df_scores["cohort"] == c]["apparent_hist_gain"].mean() for c in cohort_order]
    adj_vals = [df_scores[df_scores["cohort"] == c]["adjusted_hist_gain"].mean() for c in cohort_order]

    ax2.bar(x - width/2, app_vals, width, label="Apparent History Gain ($M_H - M_0$)", color="#1f77b4", edgecolor="black", alpha=0.85)
    ax2.bar(x + width/2, adj_vals, width, label="Group-Adjusted Memory Gain ($M_{GH} - M_G$)", color="#2ca02c", edgecolor="black", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in cohort_order], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Predictive Gain Increment (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Apparent vs. True Within-Cohort Memory Gain\n(Removing the Cohort Classification Confound)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax2.legend(loc="upper right", fontsize=9.0)

    plt.tight_layout()
    fig_path_png = os.path.join(OUT_DIR, "fig_group_confound_dissection.png")
    fig_path_pdf = os.path.join(OUT_DIR, "fig_group_confound_dissection.pdf")
    plt.savefig(fig_path_png, dpi=300)
    plt.savefig(fig_path_pdf)
    plt.close()
    print(f"Saved: {fig_path_png} and {fig_path_pdf}")
    import shutil
    shutil.copy(fig_path_png, os.path.join(ARTIFACT_DIR, "fig_group_confound_dissection.png"))

if __name__ == "__main__":
    t0 = time.time()
    run_group_confound_dissection()
    print(f"Completed in {time.time() - t0:.1f}s")
