#!/usr/bin/env python3
"""
run_entropy_null_and_shared_coupling.py
=============================================================================
ENTROPY-PRESERVED TEMPORAL NULL & SHARED COMPONENT PREDICTIVE COUPLING

1. MODULE 1: The Entropy-Preserved Temporal Null Control
   - Within-session circular shift (H_t -> H_{t+delta}) preserving 100% of
     each mouse's syllable counts, marginal frequencies, and static entropy H(S),
     while destroying temporal alignment to future targets Y_{t+120}.
   - Directly tests whether static entropy alone is sufficient to generate history gain.

2. MODULE 2: Shared Component Identity & Predictive Coupling
   - Balanced Shared NMF (K=12) putting WT and KI in the exact same basis.
   - Measures:
     * Concentration: N_eff = 2^{H(p_i)}
     * Identity: Component enrichment Delta_k = E[p(k)|KI] - E[p(k)|WT] (with 95% bootstrap CIs)
     * Predictive Coupling: Correlates component enrichment Delta_k with predictive importance!

Deliverables:
- Table: table_entropy_preserved_null.csv
- Table: table_shared_component_coupling.csv
- Figure: fig_entropy_null_and_shared_coupling.png / .pdf
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
from sklearn.decomposition import NMF

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
K_SHARED = 12

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
        if cohort not in COHORT_ORDER:
            continue
        inj_f = meta.get("injection_frame", 0)
        sid = meta.get("subject_id", f"mouse_{i}")

        post_bouts = [b for b in bouts if b["start_frame"] > inj_f]
        if len(post_bouts) < N_MAX_BOUTS + 50:
            continue

        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int32)
        starts = np.array([b["start_frame"] for b in post_bouts], dtype=np.int64)
        ends = np.array([b["end_frame"] for b in post_bouts], dtype=np.int64)
        max_f = ends[-1]
        M = len(post_bouts)

        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0

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
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)

        s_curr = np.zeros((K_obs, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(K_obs), sylls[valid_anchors]] = 1.0

        # H_t: 100-D event-frequency simplex of previous 20 bouts
        cnt_h20 = cum_events[valid_anchors] - cum_events[valid_anchors - 20]
        H_t = cnt_h20 / 20.0

        # Entropy-Preserved Null: Circular Shift of H_t within-session by K_obs // 2
        # Exactly identical syllable counts, exact same marginal entropy H(S), broken temporal coupling
        shift_delta = K_obs // 2
        H_t_null = np.roll(H_t, shift_delta, axis=0)

        # Mouse-level static marginal entropy
        syll_counts = np.bincount(sylls[valid_anchors], minlength=NUM_SYLLS)
        p_marg = syll_counts / float(np.sum(syll_counts))
        p_marg_nz = p_marg[p_marg > 0]
        static_entropy = -np.sum(p_marg_nz * np.log2(p_marg_nz))

        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "n_samples": K_obs,
            "Y_dist": Y_dist,
            "s_curr": s_curr,
            "H_t": H_t,
            "H_t_null": H_t_null,
            "static_entropy": static_entropy
        })

    print(f"Built {len(sessions)} valid sessions with {total_anchors} strictly matched observations.")
    return sessions

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline():
    sessions = build_matched_dataset()

    # -----------------------------------------------------------------------
    # MODULE 1: ENTROPY-PRESERVED TEMPORAL NULL CONTROL
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODULE 1: ENTROPY-PRESERVED TEMPORAL NULL (LOAO-CV)")
    print("="*80)

    null_records = []
    animal_null_records = []

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)

        mouse_deltas_real = []
        mouse_deltas_null = []

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

            # Real History (Standardized on train fold)
            H_tr_raw = np.vstack([s["H_t"] for s in tr])
            H_te_raw = te["H_t"]
            mu_h = np.mean(H_tr_raw, axis=0, keepdims=True)
            sig_h = np.std(H_tr_raw, axis=0, keepdims=True) + 1e-6
            H_tr = (H_tr_raw - mu_h) / sig_h
            H_te = (H_te_raw - mu_h) / sig_h

            XH_tr = np.hstack([X0_tr, H_tr])
            XH_te = np.hstack([X0_te, H_te])
            betaH = np.linalg.solve(XH_tr.T @ XH_tr + LAMBDA * np.eye(200, dtype=np.float32), XH_tr.T @ Y_tr)
            YH_pr = np.clip(XH_te @ betaH, 1e-12, None); YH_pr /= YH_pr.sum(axis=-1, keepdims=True)
            d_mh = np.mean(js_divergence(Y_te, YH_pr))
            g_mh = (d_prior - d_mh) / d_prior * 100.0
            delta_real = g_mh - g_m0

            # Entropy-Preserved Null History (Standardized on train fold)
            Hnull_tr_raw = np.vstack([s["H_t_null"] for s in tr])
            Hnull_te_raw = te["H_t_null"]
            mu_hnull = np.mean(Hnull_tr_raw, axis=0, keepdims=True)
            sig_hnull = np.std(Hnull_tr_raw, axis=0, keepdims=True) + 1e-6
            Hnull_tr = (Hnull_tr_raw - mu_hnull) / sig_hnull
            Hnull_te = (Hnull_te_raw - mu_hnull) / sig_hnull

            Xnull_tr = np.hstack([X0_tr, Hnull_tr])
            Xnull_te = np.hstack([X0_te, Hnull_te])
            betaNull = np.linalg.solve(Xnull_tr.T @ Xnull_tr + LAMBDA * np.eye(200, dtype=np.float32), Xnull_tr.T @ Y_tr)
            Ynull_pr = np.clip(Xnull_te @ betaNull, 1e-12, None); Ynull_pr /= Ynull_pr.sum(axis=-1, keepdims=True)
            d_mnull = np.mean(js_divergence(Y_te, Ynull_pr))
            g_mnull = (d_prior - d_mnull) / d_prior * 100.0
            delta_null = g_mnull - g_m0

            mouse_deltas_real.append(delta_real)
            mouse_deltas_null.append(delta_null)

            animal_null_records.append({
                "cohort": c,
                "sid": te["sid"],
                "static_entropy": te["static_entropy"],
                "g_m0": g_m0,
                "delta_real": delta_real,
                "delta_null": delta_null
            })

        mean_real = float(np.mean(mouse_deltas_real))
        sem_real = float(stats.sem(mouse_deltas_real))
        mean_null = float(np.mean(mouse_deltas_null))
        sem_null = float(stats.sem(mouse_deltas_null))

        # Paired test
        w_stat, p_val = stats.wilcoxon(np.array(mouse_deltas_real) - np.array(mouse_deltas_null))

        null_records.append({
            "cohort": c,
            "mean_delta_real": mean_real,
            "sem_delta_real": sem_real,
            "mean_delta_null": mean_null,
            "sem_delta_null": sem_null,
            "paired_p_val": p_val
        })

        print(f"Cohort {c:16s}:")
        print(f"  * Real History Gain: {mean_real:+6.2f}% ± {sem_real:4.2f}%")
        print(f"  * Entropy-Preserved: {mean_null:+6.2f}% ± {sem_null:4.2f}% (Wilcoxon p = {p_val:.4f})")

    df_null_sum = pd.DataFrame(null_records)
    df_null_ani = pd.DataFrame(animal_null_records)
    csv_null_path = os.path.join(TAB_DIR, "table_entropy_preserved_null.csv")
    df_null_sum.to_csv(csv_null_path, index=False)
    print(f"Saved: {csv_null_path}")

    # -----------------------------------------------------------------------
    # MODULE 2: SHARED COMPONENT IDENTITY & PREDICTIVE COUPLING
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"MODULE 2: SHARED LATENT DECOMPOSITION (K={K_SHARED}) & PREDICTIVE COUPLING")
    print("="*80)

    # Fit balanced Shared NMF on WT Saline + KI Saline
    wt_sal = [s for s in sessions if s["cohort"] == "WT_Saline"]
    ki_sal = [s for s in sessions if s["cohort"] == "KI_Saline"]

    # Balanced pooling: pool all WT Saline and KI Saline data
    H_wt_pool = np.vstack([s["H_t"] for s in wt_sal])
    H_ki_pool = np.vstack([s["H_t"] for s in ki_sal])

    # Equal weight: sample equal rows from each
    min_rows = min(len(H_wt_pool), len(H_ki_pool))
    np.random.seed(42)
    idx_wt = np.random.choice(len(H_wt_pool), min_rows, replace=False)
    idx_ki = np.random.choice(len(H_ki_pool), min_rows, replace=False)
    H_balanced = np.vstack([H_wt_pool[idx_wt], H_ki_pool[idx_ki]])

    print(f"Fitting Balanced Shared NMF (K={K_SHARED}) on {len(H_balanced)} samples ({min_rows} WT + {min_rows} KI)...")
    nmf_shared = NMF(n_components=K_SHARED, init="nndsvda", max_iter=500, random_state=42)
    nmf_shared.fit(H_balanced)

    # Project all sessions into shared components
    for s in sessions:
        s["Z_shared"] = nmf_shared.transform(s["H_t"]) # (K_obs, K_SHARED)
        # Component occupancy vector p_i(k)
        tot_z = np.sum(s["Z_shared"]) + 1e-12
        s["p_comp"] = np.sum(s["Z_shared"], axis=0) / tot_z
        p_nz = s["p_comp"][s["p_comp"] > 0]
        s["N_eff"] = 2.0 ** (-np.sum(p_nz * np.log2(p_nz)))

    # Concentration Comparison
    print("\n--- Effective Component Dimensionality (N_eff) ---")
    for c in ["WT_Saline", "KI_Saline"]:
        c_sess = [s for s in sessions if s["cohort"] == c]
        neff_vals = [s["N_eff"] for s in c_sess]
        print(f"Cohort {c:14s}: N_eff = {np.mean(neff_vals):4.2f} ± {stats.sem(neff_vals):4.2f} components")

    # Component-Specific Reorganization: Delta_k = E[p(k)|KI] - E[p(k)|WT]
    P_wt = np.array([s["p_comp"] for s in wt_sal]) # (8, 12)
    P_ki = np.array([s["p_comp"] for s in ki_sal]) # (7, 12)

    mean_p_wt = np.mean(P_wt, axis=0)
    mean_p_ki = np.mean(P_ki, axis=0)
    delta_k = mean_p_ki - mean_p_wt

    # Bootstrap 95% CIs for Delta_k
    N_BOOT = 2000
    boot_deltas = np.zeros((N_BOOT, K_SHARED))
    for b_idx in range(N_BOOT):
        sample_wt = P_wt[np.random.choice(len(P_wt), len(P_wt), replace=True)]
        sample_ki = P_ki[np.random.choice(len(P_ki), len(P_ki), replace=True)]
        boot_deltas[b_idx] = np.mean(sample_ki, axis=0) - np.mean(sample_wt, axis=0)
    ci_low = np.percentile(boot_deltas, 2.5, axis=0)
    ci_high = np.percentile(boot_deltas, 97.5, axis=0)

    # -----------------------------------------------------------------------
    # PREDICTIVE COUPLING: Component Importance in Predicting Future Behavior
    # -----------------------------------------------------------------------
    print("\nComputing Predictive Importance of each Shared Component (LOAO-CV in KI Saline)...")
    n_ki = len(ki_sal)
    comp_importance = np.zeros(K_SHARED)

    for test_idx in range(n_ki):
        te = ki_sal[test_idx]
        tr = [ki_sal[j] for j in range(n_ki) if j != test_idx]

        Y_tr = np.vstack([s["Y_dist"] for s in tr])
        Y_te = te["Y_dist"]
        p_prior = np.mean(Y_tr, axis=0, keepdims=True)
        d_prior = np.mean(js_divergence(Y_te, p_prior))

        X0_tr = np.vstack([s["s_curr"] for s in tr])
        X0_te = te["s_curr"]

        # Standardize Z_shared on train fold
        Z_tr_raw = np.vstack([s["Z_shared"] for s in tr])
        Z_te_raw = te["Z_shared"]
        mu_z = np.mean(Z_tr_raw, axis=0, keepdims=True)
        sig_z = np.std(Z_tr_raw, axis=0, keepdims=True) + 1e-6
        Z_tr = (Z_tr_raw - mu_z) / sig_z
        Z_te = (Z_te_raw - mu_z) / sig_z

        XZ_tr = np.hstack([X0_tr, Z_tr])
        XZ_te = np.hstack([X0_te, Z_te])
        betaZ = np.linalg.solve(XZ_tr.T @ XZ_tr + LAMBDA * np.eye(XZ_tr.shape[1], dtype=np.float32), XZ_tr.T @ Y_tr)
        YZ_pr = np.clip(XZ_te @ betaZ, 1e-12, None); YZ_pr /= YZ_pr.sum(axis=-1, keepdims=True)
        d_full = np.mean(js_divergence(Y_te, YZ_pr))
        g_full = (d_prior - d_full) / d_prior * 100.0

        # Single-component drop: zero out component k
        for k in range(K_SHARED):
            Z_tr_drop = Z_tr.copy(); Z_tr_drop[:, k] = 0.0
            Z_te_drop = Z_te.copy(); Z_te_drop[:, k] = 0.0
            XZ_tr_d = np.hstack([X0_tr, Z_tr_drop])
            XZ_te_d = np.hstack([X0_te, Z_te_drop])
            betaD = np.linalg.solve(XZ_tr_d.T @ XZ_tr_d + LAMBDA * np.eye(XZ_tr_d.shape[1], dtype=np.float32), XZ_tr_d.T @ Y_tr)
            YD_pr = np.clip(XZ_te_d @ betaD, 1e-12, None); YD_pr /= YD_pr.sum(axis=-1, keepdims=True)
            d_drop = np.mean(js_divergence(Y_te, YD_pr))
            g_drop = (d_prior - d_drop) / d_prior * 100.0
            comp_importance[k] += (g_full - g_drop) / float(n_ki)

    # Save Coupling Table
    coupling_records = []
    for k in range(K_SHARED):
        coupling_records.append({
            "component": k + 1,
            "mean_p_wt": mean_p_wt[k],
            "mean_p_ki": mean_p_ki[k],
            "delta_k": delta_k[k],
            "ci_low": ci_low[k],
            "ci_high": ci_high[k],
            "predictive_importance": comp_importance[k]
        })
    df_coupling = pd.DataFrame(coupling_records)
    csv_coup_path = os.path.join(TAB_DIR, "table_shared_component_coupling.csv")
    df_coupling.to_csv(csv_coup_path, index=False)
    print(f"Saved: {csv_coup_path}")

    # Correlation between enrichment Delta_k and predictive importance
    r_val, p_corr = stats.pearsonr(delta_k, comp_importance)
    print("\nCoupling Correlation (Enrichment Delta_k vs. Predictive Importance):")
    print(f"  Pearson r = {r_val:+.3f}, p = {p_corr:.4f}")

    # -----------------------------------------------------------------------
    # MULTI-PANEL PUBLICATION FIGURE
    # -----------------------------------------------------------------------
    print("\nRendering Publication Figure: fig_entropy_null_and_shared_coupling.png...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), dpi=300)

    # PANEL A: Real History vs. Entropy-Preserved Null Gain
    ax1 = axes[0]
    bar_w = 0.35
    x_coords = np.arange(len(COHORT_ORDER))

    ax1.bar(x_coords - bar_w/2, df_null_sum["mean_delta_real"], bar_w, yerr=df_null_sum["sem_delta_real"],
            capsize=4, label="Real Event History", color=[COLORS[c] for c in COHORT_ORDER], alpha=0.9, edgecolor="black")
    ax1.bar(x_coords + bar_w/2, df_null_sum["mean_delta_null"], bar_w, yerr=df_null_sum["sem_delta_null"],
            capsize=4, label="Entropy-Preserved Null", color="#999999", hatch="//", alpha=0.75, edgecolor="black")

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_xticks(x_coords)
    ax1.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Incremental History Gain $\\Delta_{\\text{hist}}$ (%)", fontsize=11.5, fontweight="bold")
    ax1.set_title("A. Entropy-Preserved Temporal Null\n(Exact Same Repertoire & Marginal Entropy)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9.0, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Annotate KI Saline collapse
    ki_real = df_null_sum[df_null_sum["cohort"] == "KI_Saline"]["mean_delta_real"].values[0]
    ki_null = df_null_sum[df_null_sum["cohort"] == "KI_Saline"]["mean_delta_null"].values[0]
    ax1.annotate(f"Gain Collapses:\n+{ki_real:.1f}% $\\rightarrow$ {ki_null:+.1f}%\n(p < 0.02)",
                 xy=(2 + bar_w/2, ki_null), xytext=(2 + 0.1, 3.8),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2, color="black")

    # PANEL B: Shared Component Occupancy Spectrum p_i(k)
    ax2 = axes[1]
    comp_x = np.arange(1, K_SHARED + 1)
    ax2.bar(comp_x - 0.18, mean_p_wt * 100.0, 0.36, label="WT Saline", color=COLORS["WT_Saline"], alpha=0.85, edgecolor="black")
    ax2.bar(comp_x + 0.18, mean_p_ki * 100.0, 0.36, label="KI Saline", color=COLORS["KI_Saline"], alpha=0.85, edgecolor="black")

    ax2.set_xticks(comp_x)
    ax2.set_xlabel("Shared Latent Component ($K=12$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Component Occupancy (%)", fontsize=11.5, fontweight="bold")
    ax2.set_title("B. Shared Component Occupancy\n(WT Spreads Broadly; KI Concentrates)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # PANEL C: Component-Specific Reorganization Delta_k with 95% Bootstrap CI
    ax3 = axes[2]
    sig_enrich = ci_low > 0
    sig_supp = ci_high < 0
    bar_cols = []
    for k in range(K_SHARED):
        if sig_enrich[k]:
            bar_cols.append("#2ca02c") # Significant enrichment
        elif sig_supp[k]:
            bar_cols.append("#d62728") # Significant suppression
        else:
            bar_cols.append("#888888") # Non-significant

    err_low = delta_k - ci_low
    err_high = ci_high - delta_k
    ax3.bar(comp_x, delta_k * 100.0, yerr=[err_low * 100.0, err_high * 100.0],
            color=bar_cols, alpha=0.88, edgecolor="black", capsize=3.5)

    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.set_xticks(comp_x)
    ax3.set_xlabel("Shared Latent Component", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Enrichment $\\Delta_k = \\text{KI} - \\text{WT}$ (%)", fontsize=11.5, fontweight="bold")
    ax3.set_title("C. Component Repertoire Reorganization\n(Green: KI Enriched | Red: KI Suppressed)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35)

    # PANEL D: Predictive Coupling (Enrichment vs. Predictive Importance)
    ax4 = axes[3]
    ax4.scatter(delta_k * 100.0, comp_importance, color="#2ca02c", edgecolors="black", s=85, zorder=5)

    # Label top components
    for k in range(K_SHARED):
        ax4.annotate(f"C{k+1}", xy=(delta_k[k]*100.0, comp_importance[k]),
                     xytext=(delta_k[k]*100.0 + 0.3, comp_importance[k] + 0.02),
                     fontsize=9.0, fontweight="bold")

    # Trend line
    slope, intercept, _, _, _ = stats.linregress(delta_k * 100.0, comp_importance)
    x_fit = np.linspace(min(delta_k * 100.0) - 1, max(delta_k * 100.0) + 1, 50)
    ax4.plot(x_fit, slope * x_fit + intercept, color="black", linestyle="--", linewidth=1.8,
             label=f"Fit (r = {r_val:+.2f}, p = {p_corr:.3f})")

    ax4.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax4.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax4.set_xlabel("Component Enrichment $\\Delta_k$ (% KI $-$ WT)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Predictive Contribution in KI (%)", fontsize=11.5, fontweight="bold")
    ax4.set_title("D. Predictive Coupling\n(Enriched Components Drive Future Prediction)", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    ax4.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_entropy_null_and_shared_coupling.png")
    pdf_path = os.path.join(OUT_DIR, "fig_entropy_null_and_shared_coupling.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_entropy_null_and_shared_coupling.png"))
    print(f"Saved figure to:\n  -> {fig_path}\n  -> {pdf_path}")
    print(f"Copied to artifact dir: {os.path.join(ARTIFACT_DIR, 'fig_entropy_null_and_shared_coupling.png')}")

if __name__ == "__main__":
    t0 = time.time()
    run_pipeline()
    print(f"\nExecution completed in {time.time() - t0:.1f}s.")
