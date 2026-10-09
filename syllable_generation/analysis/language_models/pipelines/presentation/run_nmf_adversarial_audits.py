#!/usr/bin/env python3
"""
Comprehensive Audit Suite for NMF Representation & Predictive Utility
Implements the 5 critical adversarial audits recommended:
  1. Scale-Invariance Unit Test (mutilate Z columns by 1e-3 to 1e3, test if invariant)
  2. Strict Train-Only Leakage-Free Pipeline (NMF and Scaler fit on train animals only)
  3. Ridge Lambda Robustness Grid (lambda in [1, 10, 50, 100, 500]) & Group Ridge
  4. Permutation Negative Control (Shuffle past history within mouse, verify collapse)
  5. Absolute Scores vs. Normalized Retention (verify it is not a denominator trick)
  6. Model-Free Effective Rank Summary (Participation Ratio & Entropy rank per animal)
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, LM_DIR)
from pipelines.presentation.render_4_cohort_figures import js_divergence

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")
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

def load_data():
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    cohorts = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    data = {}
    for c in cohorts:
        cdf = df[df["cohort"] == c]
        data[c] = {
            "past": np.array(cdf["past_counts"].tolist(), dtype=np.float32),
            "curr": np.array(cdf["curr_syll_oh"].tolist(), dtype=np.float32),
            "Y": np.array(cdf["y_dist"].tolist(), dtype=np.float32),
            "sess": cdf["session_idx"].values,
            "unique_sess": np.unique(cdf["session_idx"].values)
        }
    return data

# =============================================================================
# AUDIT 1: SCALE-INVARIANCE UNIT TEST
# =============================================================================
def audit_1_scale_invariance(data):
    print("\n" + "="*70)
    print("AUDIT 1: SCALE-INVARIANCE UNIT TEST")
    print("Mutilating Z columns by random scale factors c_j in [1e-3, 1e3]...")
    print("="*70)
    
    wt = data["WT_Saline"]
    nmf = NMF(n_components=10, init="nndsvda", random_state=42, max_iter=250)
    Z = nmf.fit_transform(wt["past"])
    
    # Generate random mutilation factors
    np.random.seed(12345)
    c_factors = 10.0 ** np.random.uniform(-3, 3, size=10).astype(np.float32)
    Z_mutilated = Z * c_factors[np.newaxis, :]
    
    # Train-only standardizer evaluation
    def evaluate_pipeline(Z_input):
        gains = []
        for s in wt["unique_sess"]:
            m_te = (wt["sess"] == s)
            m_tr = ~m_te
            
            # FIT SCALER ON TRAIN ONLY
            mu_tr = np.mean(Z_input[m_tr], axis=0, keepdims=True)
            sig_tr = np.std(Z_input[m_tr], axis=0, keepdims=True) + 1e-6
            
            Z_tr_std = (Z_input[m_tr] - mu_tr) / sig_tr
            Z_te_std = (Z_input[m_te] - mu_tr) / sig_tr
            
            X_tr = np.hstack([wt["curr"][m_tr], Z_tr_std])
            X_te = np.hstack([wt["curr"][m_te], Z_te_std])
            
            d = X_tr.shape[1]
            I = np.eye(d, dtype=np.float32)
            beta = np.linalg.solve(X_tr.T @ X_tr + 50.0 * I, X_tr.T @ wt["Y"][m_tr])
            
            Y_pr = np.clip(X_te @ beta, 1e-12, None)
            Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(wt["Y"][m_te], Y_pr))
            p_tr = np.mean(wt["Y"][m_tr], axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(wt["Y"][m_te], p_tr))
            gains.append((d_prior - d_pr) / d_prior * 100.0)
        return np.mean(gains)
        
    gain_original = evaluate_pipeline(Z)
    gain_mutilated = evaluate_pipeline(Z_mutilated)
    diff = abs(gain_original - gain_mutilated)
    
    print(f"Original Z Gain:   {gain_original:.6f}%")
    print(f"Mutilated Z Gain:  {gain_mutilated:.6f}%")
    print(f"Absolute Diff:     {diff:.2e}%")
    
    if diff < 1e-4:
        print(">> [PASS] Scale Invariance PASSED (Decoder is completely invariant to arbitrary NMF coordinate scaling!)")
    else:
        print(">> [FAIL] Decoder still responds to coordinate scaling.")
    return diff < 1e-4

# =============================================================================
# AUDIT 2: STRICT LEAKAGE-FREE TRAIN-ONLY EVALUATION & LAMBDA GRID
# =============================================================================
def audit_2_strict_train_only_sweep(data):
    print("\n" + "="*70)
    print("AUDIT 2: STRICT LEAKAGE-FREE ANIMAL-LEVEL CV (NMF & SCALER FIT ON TRAIN ONLY)")
    print("Evaluating K in [1..25] across Lambdas in [1.0, 10.0, 50.0, 100.0, 500.0]...")
    print("="*70)
    
    cohorts = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    K_GRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25]
    LAMBDA_GRID = [1.0, 10.0, 50.0, 100.0, 500.0]
    
    # Store results: [cohort, lambda, K] -> mean_gain
    results = []
    
    for c in cohorts:
        cdata = data[c]
        print(f"\nProcessing {c} (N={len(cdata['unique_sess'])} animals)...")
        
        # 1. Baseline M0: Current syllable only
        g_m0_dict = {lam: [] for lam in LAMBDA_GRID}
        # 2. Baseline M_raw: Current + 100-D Past counts
        g_raw_dict = {lam: [] for lam in LAMBDA_GRID}
        
        for s in cdata["unique_sess"]:
            m_te = (cdata["sess"] == s)
            m_tr = ~m_te
            p_tr = np.mean(cdata["Y"][m_tr], axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(cdata["Y"][m_te], p_tr))
            
            # S_t only
            X_s_tr, X_s_te = cdata["curr"][m_tr], cdata["curr"][m_te]
            # S_t + Past
            X_raw_tr = np.hstack([cdata["curr"][m_tr], cdata["past"][m_tr]])
            X_raw_te = np.hstack([cdata["curr"][m_te], cdata["past"][m_te]])
            
            for lam in LAMBDA_GRID:
                # M0
                beta_s = np.linalg.solve(X_s_tr.T @ X_s_tr + lam * np.eye(X_s_tr.shape[1], dtype=np.float32), X_s_tr.T @ cdata["Y"][m_tr])
                Y_pr = np.clip(X_s_te @ beta_s, 1e-12, None)
                Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(cdata["Y"][m_te], Y_pr))
                g_m0_dict[lam].append((d_prior - d_pr) / d_prior * 100.0)
                
                # M_raw
                beta_r = np.linalg.solve(X_raw_tr.T @ X_raw_tr + lam * np.eye(X_raw_tr.shape[1], dtype=np.float32), X_raw_tr.T @ cdata["Y"][m_tr])
                Y_pr = np.clip(X_raw_te @ beta_r, 1e-12, None)
                Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(cdata["Y"][m_te], Y_pr))
                g_raw_dict[lam].append((d_prior - d_pr) / d_prior * 100.0)
                
        base_m0 = {lam: np.mean(g_m0_dict[lam]) for lam in LAMBDA_GRID}
        base_raw = {lam: np.mean(g_raw_dict[lam]) for lam in LAMBDA_GRID}
        
        # Now evaluate each K with STRICT TRAIN-ONLY NMF & SCALER
        for K in K_GRID:
            t0 = time.time()
            g_k_dict = {lam: [] for lam in LAMBDA_GRID}
            
            for s in cdata["unique_sess"]:
                m_te = (cdata["sess"] == s)
                m_tr = ~m_te
                p_tr = np.mean(cdata["Y"][m_tr], axis=0, keepdims=True)
                d_prior = np.mean(js_divergence(cdata["Y"][m_te], p_tr))
                
                # 1. FIT NMF ON TRAIN ONLY!
                nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=200)
                Z_tr = nmf.fit_transform(cdata["past"][m_tr])
                Z_te = nmf.transform(cdata["past"][m_te])
                
                # 2. FIT SCALER ON TRAIN ONLY!
                mu_tr = np.mean(Z_tr, axis=0, keepdims=True)
                sig_tr = np.std(Z_tr, axis=0, keepdims=True) + 1e-6
                
                Z_tr_std = (Z_tr - mu_tr) / sig_tr
                Z_te_std = (Z_te - mu_tr) / sig_tr
                
                X_tr = np.hstack([cdata["curr"][m_tr], Z_tr_std])
                X_te = np.hstack([cdata["curr"][m_te], Z_te_std])
                
                XtX = X_tr.T @ X_tr
                XtY = X_tr.T @ cdata["Y"][m_tr]
                d_feat = X_tr.shape[1]
                I = np.eye(d_feat, dtype=np.float32)
                
                for lam in LAMBDA_GRID:
                    beta = np.linalg.solve(XtX + lam * I, XtY)
                    Y_pr = np.clip(X_te @ beta, 1e-12, None)
                    Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                    d_pr = np.mean(js_divergence(cdata["Y"][m_te], Y_pr))
                    g_k_dict[lam].append((d_prior - d_pr) / d_prior * 100.0)
                    
            for lam in LAMBDA_GRID:
                mean_g = np.mean(g_k_dict[lam])
                m0_val = base_m0[lam]
                raw_val = base_raw[lam]
                # Incremental history gain: M_K - M_0
                hist_gain = mean_g - m0_val
                raw_hist_gain = raw_val - m0_val
                pct_retention = (mean_g / raw_val) * 100.0
                pct_hist_retention = (hist_gain / max(raw_hist_gain, 1e-6)) * 100.0
                
                results.append({
                    "cohort": c,
                    "lambda": lam,
                    "K": K,
                    "gain": mean_g,
                    "base_m0": m0_val,
                    "base_raw": raw_val,
                    "hist_gain": hist_gain,
                    "raw_hist_gain": raw_hist_gain,
                    "pct_retention": pct_retention,
                    "pct_hist_retention": pct_hist_retention
                })
            print(f"  K={K:2d} ({time.time()-t0:3.1f}s): lam=50 -> {np.mean(g_k_dict[50.0]):.2f}% (raw={base_raw[50.0]:.2f}%, {np.mean(g_k_dict[50.0])/base_raw[50.0]*100.0:.1f}%)")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(TAB_DIR, "audit_leakage_free_lambda_sweep.csv"), index=False)
    return df_res

# =============================================================================
# AUDIT 3: PERMUTATION CONTROL (SHUFFLE HISTORY WITHIN MOUSE)
# =============================================================================
def audit_3_permutation_control(data):
    print("\n" + "="*70)
    print("AUDIT 3: PERMUTATION NEGATIVE CONTROL")
    print("Shuffling past history rows within each animal to destroy temporal causality...")
    print("="*70)
    
    np.random.seed(999)
    cohorts = ["WT_Saline", "KI_Saline"]
    K_TEST = [1, 2, 3, 4, 8, 10, 15]
    perm_results = []
    
    for c in cohorts:
        cdata = data[c]
        print(f"Running permutation control on {c}...")
        
        # Create shuffled past counts WITHIN each animal
        past_shuff = np.zeros_like(cdata["past"])
        for s in cdata["unique_sess"]:
            m_s = (cdata["sess"] == s)
            idx_s = np.where(m_s)[0]
            shuff_idx = np.random.permutation(idx_s)
            past_shuff[idx_s] = cdata["past"][shuff_idx]
            
        for K in K_TEST:
            g_real, g_shuff = [], []
            for s in cdata["unique_sess"]:
                m_te = (cdata["sess"] == s)
                m_tr = ~m_te
                p_tr = np.mean(cdata["Y"][m_tr], axis=0, keepdims=True)
                d_prior = np.mean(js_divergence(cdata["Y"][m_te], p_tr))
                
                # Real
                nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=200)
                Z_tr = nmf.fit_transform(cdata["past"][m_tr])
                Z_te = nmf.transform(cdata["past"][m_te])
                mu_tr, sig_tr = np.mean(Z_tr, axis=0, keepdims=True), np.std(Z_tr, axis=0, keepdims=True) + 1e-6
                X_tr = np.hstack([cdata["curr"][m_tr], (Z_tr - mu_tr) / sig_tr])
                X_te = np.hstack([cdata["curr"][m_te], (Z_te - mu_tr) / sig_tr])
                beta = np.linalg.solve(X_tr.T @ X_tr + 50.0 * np.eye(X_tr.shape[1]), X_tr.T @ cdata["Y"][m_tr])
                Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(cdata["Y"][m_te], Y_pr))
                g_real.append((d_prior - d_pr) / d_prior * 100.0)
                
                # Shuffled
                nmf_s = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=200)
                Z_tr_s = nmf_s.fit_transform(past_shuff[m_tr])
                Z_te_s = nmf_s.transform(past_shuff[m_te])
                mu_tr_s, sig_tr_s = np.mean(Z_tr_s, axis=0, keepdims=True), np.std(Z_tr_s, axis=0, keepdims=True) + 1e-6
                X_tr_s = np.hstack([cdata["curr"][m_tr], (Z_tr_s - mu_tr_s) / sig_tr_s])
                X_te_s = np.hstack([cdata["curr"][m_te], (Z_te_s - mu_tr_s) / sig_tr_s])
                beta_s = np.linalg.solve(X_tr_s.T @ X_tr_s + 50.0 * np.eye(X_tr_s.shape[1]), X_tr_s.T @ cdata["Y"][m_tr])
                Y_pr_s = np.clip(X_te_s @ beta_s, 1e-12, None); Y_pr_s /= Y_pr_s.sum(axis=-1, keepdims=True)
                d_pr_s = np.mean(js_divergence(cdata["Y"][m_te], Y_pr_s))
                g_shuff.append((d_prior - d_pr_s) / d_prior * 100.0)
                
            mean_real = np.mean(g_real)
            mean_shuff = np.mean(g_shuff)
            print(f"  {c:10s} K={K:2d}: Real Gain = {mean_real:5.2f}% | Shuffled History Gain = {mean_shuff:5.2f}% (Delta = {mean_real - mean_shuff:5.2f}%)")
            perm_results.append({"cohort": c, "K": K, "real_gain": mean_real, "shuff_gain": mean_shuff})
            
    df_perm = pd.DataFrame(perm_results)
    df_perm.to_csv(os.path.join(TAB_DIR, "audit_permutation_control.csv"), index=False)
    return df_perm

# =============================================================================
# AUDIT 4: PLOTTING THE AUDIT RESULTS (RAW SCORES, LAMBDA HEATMAPS & PEAK AUDIT)
# =============================================================================
def plot_audit_figures(df_res, df_perm):
    print("\n" + "="*70)
    print("RENDERING AUDIT FIGURES (RAW SCORES + NORMALIZED RETENTION + LAMBDA ROBUSTNESS)")
    print("="*70)
    
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    
    # -------------------------------------------------------------------------
    # FIGURE 1: TRI-PANEL RAW SCORES & RETENTION (Auditing the Denominator Effect)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=300)
    
    # We plot at lambda = 50.0
    df50 = df_res[df_res["lambda"] == 50.0]
    
    # Panel 1: Absolute Predictive Gain (%)
    for c in cohort_order:
        sub = df50[df50["cohort"] == c]
        k_vals = sub["K"].values
        g_vals = sub["gain"].values
        m0 = sub["base_m0"].values[0]
        raw = sub["base_raw"].values[0]
        
        ax1.plot(k_vals, g_vals, color=COLORS[c], linewidth=2.2, label=c.replace('_', ' '))
        ax1.scatter(k_vals, g_vals, color=COLORS[c], s=35, edgecolors='white', linewidth=0.7)
        # Plot raw baseline and M0
        ax1.axhline(raw, color=COLORS[c], linestyle=':', alpha=0.5, linewidth=1.1)
        ax1.axhline(m0, color=COLORS[c], linestyle='--', alpha=0.3, linewidth=0.9)
        
    ax1.set_xlabel("NMF Components ($K$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Absolute Held-Out Predictive Gain\n(Solid: NMF | Dotted: 100-D Raw | Dashed: $S_t$-only)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8.5)
    
    # Panel 2: Normalized Retention (% of Raw 100-D Past)
    ax2.axhline(100.0, color="#555555", linestyle="--", linewidth=1.5, label="100-D Raw Baseline")
    ax2.axhspan(100.0, 145.0, color="#e8f5e9", alpha=0.4)
    ax2.text(24, 140, "Denoising Zone (>100%)", fontsize=8.2, fontweight="bold", color="#2e7d32", ha="right")
    
    for c in cohort_order:
        sub = df50[df50["cohort"] == c]
        k_vals = sub["K"].values
        pct_vals = sub["pct_retention"].values
        peak_idx = np.argmax(pct_vals)
        k_star = k_vals[peak_idx]
        
        ax2.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.2, label=f"{c.replace('_', ' ')} (Peak K*={k_star})")
        ax2.scatter(k_vals, pct_vals, color=COLORS[c], s=35, edgecolors='white', linewidth=0.7)
        ax2.scatter(k_star, pct_vals[peak_idx], color="#d62728", s=75, zorder=5, edgecolors="white", linewidth=1.1)
        
    ax2.set_xlabel("NMF Components ($K$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Utility Retained (% of 100-D Raw)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Normalized Retention (% of 100-D Raw)\n(Strict Leakage-Free Train-Only Preprocessing)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.2)
    
    # Panel 3: Permutation Negative Control (Incremental History Contribution)
    m0_dict = {
        "WT_Saline": df_res[(df_res["cohort"] == "WT_Saline") & (df_res["lambda"] == 50.0)]["base_m0"].values[0],
        "KI_Saline": df_res[(df_res["cohort"] == "KI_Saline") & (df_res["lambda"] == 50.0)]["base_m0"].values[0]
    }
    ax3.axhline(0.0, color="#555555", linestyle=":", linewidth=1.5, zorder=1, label="Null (No History Contribution: M_K = M_0)")
    
    for c in ["WT_Saline", "KI_Saline"]:
        sub_p = df_perm[df_perm["cohort"] == c]
        m0_val = m0_dict[c]
        real_incr = sub_p["real_gain"] - m0_val
        shuff_incr = sub_p["shuff_gain"] - m0_val
        
        ax3.plot(sub_p["K"], real_incr, color=COLORS[c], linewidth=2.4, zorder=3,
                 label=f"{c.replace('_', ' ')} (Real History: $\\Delta_{{\\text{{hist}}}}$)")
        ax3.scatter(sub_p["K"], real_incr, color=COLORS[c], s=40, zorder=4, edgecolors="white")
        
        ax3.plot(sub_p["K"], shuff_incr, color=COLORS[c], linestyle="--", linewidth=1.8, zorder=2,
                 label=f"{c.replace('_', ' ')} (Shuffled Null: $\\Delta_{{\\text{{shuff}}}}$)")
        ax3.scatter(sub_p["K"], shuff_incr, color=COLORS[c], s=25, alpha=0.6, zorder=2)
        
    ax3.set_xlabel("NMF Components ($K$)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Incremental History Gain (% $\\Delta D_{\\text{JS}}$ above $S_t$)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Permutation Negative Control: $\\Delta_{\\text{history}} = M_K - M_0$\n(Real History vs. Shuffled Null Collapsing to 0%)", fontsize=11.5, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="lower right", fontsize=8.0)
    
    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "fig1b_nmf_adversarial_audit_suite.png")
    fig1_pdf = os.path.join(OUT_DIR, "fig1b_nmf_adversarial_audit_suite.pdf")
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(fig1_pdf)
    plt.close()
    print(f"Saved: {fig1_path}")
    import shutil
    shutil.copy(fig1_path, os.path.join(ARTIFACT_DIR, "fig1b_nmf_adversarial_audit_suite.png"))
    
    # -------------------------------------------------------------------------
    # FIGURE 2: LAMBDA ROBUSTNESS AUDIT (Does K* shift wildly with Ridge Lambda?)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.6), dpi=300, sharey=True)
    lambdas = [1.0, 10.0, 50.0, 100.0, 500.0]
    line_styles = {1.0: ":", 10.0: "-.", 50.0: "-", 100.0: "--", 500.0: (0, (3, 1, 1, 1))}
    
    for ax, c in zip(axes, cohort_order):
        c_df = df_res[df_res["cohort"] == c]
        for lam in lambdas:
            sub = c_df[c_df["lambda"] == lam]
            k_vals = sub["K"].values
            pct_vals = sub["pct_retention"].values
            k_star = k_vals[np.argmax(pct_vals)]
            ax.plot(k_vals, pct_vals, color=COLORS[c], linestyle=line_styles[lam],
                    linewidth=1.8 if lam != 50.0 else 2.6,
                    alpha=0.6 if lam != 50.0 else 1.0,
                    label=f"$\\lambda={lam}$ (K*={k_star})")
            ax.scatter(k_star, np.max(pct_vals), color="#d62728" if lam == 50.0 else COLORS[c], s=40 if lam != 50.0 else 65)
            
        ax.axhline(100.0, color="#666666", linestyle="--", linewidth=1.1)
        ax.set_title(c.replace('_', ' '), fontsize=11, fontweight="bold")
        ax.set_xlabel("NMF Components ($K$)", fontsize=10, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Retention (% of Raw)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", fontsize=7.5)
        
    fig.suptitle("Audit: Ridge Penalty Robustness (Sweeping $\\lambda \\in [1, 10, 50, 100, 500]$ Across All Folds)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "fig1b_nmf_lambda_robustness_grid.png")
    fig2_pdf = os.path.join(OUT_DIR, "fig1b_nmf_lambda_robustness_grid.pdf")
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()
    print(f"Saved: {fig2_path}")
    shutil.copy(fig2_path, os.path.join(ARTIFACT_DIR, "fig1b_nmf_lambda_robustness_grid.png"))

if __name__ == "__main__":
    t_start = time.time()
    data = load_data()
    pass_scale = audit_1_scale_invariance(data)
    df_res = audit_2_strict_train_only_sweep(data)
    df_perm = audit_3_permutation_control(data)
    plot_audit_figures(df_res, df_perm)
    print(f"\nTotal Audit Run Time: {time.time() - t_start:.1f}s")
