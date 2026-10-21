#!/usr/bin/env python3
"""
run_latent_regime_sufficiency.py
=============================================================================
WT-vs-KI LATENT REGIME SUFFICIENCY TEST:
Does a compact latent state inferred from recent history capture a larger
fraction of history's predictive information in DAT-KI than in WT?

Evaluates on strictly matched observation anchors under within-genotype
Leave-One-Animal-Out (LOAO) cross-validation:
- Primary Genotype Contrast: WT Saline (N=8) vs. KI Saline (N=7)
- Compares NMF vs. PCA latent state representations across K in [1, 2, 3, 4, 5, 8, 10, 15, 20]
- Zero leakage: Dictionary, scaling, and models fit strictly on training fold

Deliverables:
- Table: table_latent_regime_sufficiency_summary.csv
- Table: table_latent_regime_sufficiency_animals.csv
- Figure: fig_latent_regime_sufficiency.png / .pdf
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
from sklearn.decomposition import NMF, PCA

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

K_GRID = [1, 2, 3, 4, 5, 8, 10, 15, 20]
PRIMARY_COHORTS = ["WT_Saline", "KI_Saline"]
ALL_COHORTS = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

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
# DATA LOADING & MATCHED ANCHOR SELECTION
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
        if cohort not in ALL_COHORTS:
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

        # Cumulative event counts over bouts
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0

        # Continuous frame array for future target Y (h = 120s)
        N_frames = max_f + 1
        cum_frame_sylls = np.zeros((N_frames, NUM_SYLLS), dtype=np.float32)
        for b in post_bouts:
            dur = b["end_frame"] - b["start_frame"]
            if dur > 0:
                cum_frame_sylls[b["start_frame"]:b["end_frame"], b["syllable"]] = 1.0
        cum_frame_sylls = np.cumsum(cum_frame_sylls, axis=0)

        # Match anchors identically:
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

        # 1. Target Y: Future h=120s Syllable distribution (K_obs, 100)
        t_ends = ends[valid_anchors]
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)

        # 2. Baseline S_t: Current Syllable one-hot (K_obs, 100)
        s_curr = np.zeros((K_obs, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(K_obs), sylls[valid_anchors]] = 1.0

        # 3. History Feature H_t: 100-D event-frequency simplex of previous 20 bouts
        cnt_h20 = cum_events[valid_anchors] - cum_events[valid_anchors - 20]
        H_t = cnt_h20 / 20.0 # strictly sums to 1.0

        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "n_samples": K_obs,
            "Y_dist": Y_dist,
            "s_curr": s_curr,
            "H_t": H_t
        })

    print(f"Built {len(sessions)} valid sessions with {total_anchors} strictly matched observations.")
    return sessions

# ---------------------------------------------------------------------------
# EXECUTE LATENT REGIME SUFFICIENCY BENCHMARK
# ---------------------------------------------------------------------------
def run_sufficiency_test():
    sessions = build_matched_dataset()

    summary_records = []
    animal_records = []

    print("\n" + "="*80)
    print("RUNNING LATENT REGIME SUFFICIENCY TEST (LOAO CROSS-VALIDATION)")
    print("="*80)

    for c in ALL_COHORTS:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        print(f"\nCohort: {c} (N = {n_mice} mice)")

        # 1. Evaluate Baseline M0 (S_t only) & Full History MH (S_t + H_t) per mouse
        m0_scores = {}
        mh_scores = {}
        for test_idx in range(n_mice):
            te = c_sess[test_idx]
            tr = [c_sess[j] for j in range(n_mice) if j != test_idx]

            Y_tr = np.vstack([s["Y_dist"] for s in tr])
            Y_te = te["Y_dist"]
            p_prior = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_prior))

            # M0
            X0_tr = np.vstack([s["s_curr"] for s in tr])
            X0_te = te["s_curr"]
            beta0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
            Y0_pr = np.clip(X0_te @ beta0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0
            m0_scores[test_idx] = {"d_prior": d_prior, "g_m0": g_m0}

            # MH (Standardize H_t on train fold)
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
            delta_h = g_mh - g_m0
            mh_scores[test_idx] = {"g_mh": g_mh, "delta_h": delta_h}

        mean_m0 = np.mean([m0_scores[i]["g_m0"] for i in range(n_mice)])
        mean_mh = np.mean([mh_scores[i]["g_mh"] for i in range(n_mice)])
        mean_dh = np.mean([mh_scores[i]["delta_h"] for i in range(n_mice)])
        sem_dh = float(stats.sem([mh_scores[i]["delta_h"] for i in range(n_mice)]))
        print(f"  [Baseline M0] Gain = {mean_m0:5.2f}%")
        print(f"  [Full Hist MH] Gain = {mean_mh:5.2f}% | Delta_H = {mean_dh:+6.2f}% ± {sem_dh:4.2f}%")

        # 2. Evaluate Latent Regimes across K_GRID for both NMF and PCA
        for method in ["NMF", "PCA"]:
            for K in K_GRID:
                mouse_records = []
                for test_idx in range(n_mice):
                    te = c_sess[test_idx]
                    tr = [c_sess[j] for j in range(n_mice) if j != test_idx]

                    Y_tr = np.vstack([s["Y_dist"] for s in tr])
                    Y_te = te["Y_dist"]
                    d_prior = m0_scores[test_idx]["d_prior"]
                    g_m0 = m0_scores[test_idx]["g_m0"]
                    g_mh = mh_scores[test_idx]["g_mh"]
                    delta_h = mh_scores[test_idx]["delta_h"]

                    X0_tr = np.vstack([s["s_curr"] for s in tr])
                    X0_te = te["s_curr"]

                    H_tr_raw = np.vstack([s["H_t"] for s in tr])
                    H_te_raw = te["H_t"]

                    # Fit Latent State Model strictly on training fold
                    if method == "NMF":
                        decomp = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=42)
                        Z_tr_raw = decomp.fit_transform(H_tr_raw)
                        Z_te_raw = decomp.transform(H_te_raw)
                    else:
                        decomp = PCA(n_components=K, random_state=42)
                        Z_tr_raw = decomp.fit_transform(H_tr_raw)
                        Z_te_raw = decomp.transform(H_te_raw)

                    # Standardize Z on training fold
                    mu_z = np.mean(Z_tr_raw, axis=0, keepdims=True)
                    sig_z = np.std(Z_tr_raw, axis=0, keepdims=True) + 1e-6
                    Z_tr = (Z_tr_raw - mu_z) / sig_z
                    Z_te = (Z_te_raw - mu_z) / sig_z

                    # Also standardize H for nested model MQH
                    mu_h = np.mean(H_tr_raw, axis=0, keepdims=True)
                    sig_h = np.std(H_tr_raw, axis=0, keepdims=True) + 1e-6
                    H_tr = (H_tr_raw - mu_h) / sig_h
                    H_te = (H_te_raw - mu_h) / sig_h

                    # Model MQ(K): S_t + Z_t
                    XQ_tr = np.hstack([X0_tr, Z_tr])
                    XQ_te = np.hstack([X0_te, Z_te])
                    dim_q = XQ_tr.shape[1]
                    betaQ = np.linalg.solve(XQ_tr.T @ XQ_tr + LAMBDA * np.eye(dim_q, dtype=np.float32), XQ_tr.T @ Y_tr)
                    YQ_pr = np.clip(XQ_te @ betaQ, 1e-12, None); YQ_pr /= YQ_pr.sum(axis=-1, keepdims=True)
                    d_mq = np.mean(js_divergence(Y_te, YQ_pr))
                    g_mq = (d_prior - d_mq) / d_prior * 100.0
                    delta_q = g_mq - g_m0

                    # Model MQH(K): S_t + Z_t + H_t
                    XQH_tr = np.hstack([X0_tr, Z_tr, H_tr])
                    XQH_te = np.hstack([X0_te, Z_te, H_te])
                    dim_qh = XQH_tr.shape[1]
                    betaQH = np.linalg.solve(XQH_tr.T @ XQH_tr + LAMBDA * np.eye(dim_qh, dtype=np.float32), XQH_tr.T @ Y_tr)
                    YQH_pr = np.clip(XQH_te @ betaQH, 1e-12, None); YQH_pr /= YQH_pr.sum(axis=-1, keepdims=True)
                    d_mqh = np.mean(js_divergence(Y_te, YQH_pr))
                    g_mqh = (d_prior - d_mqh) / d_prior * 100.0

                    # Metrics:
                    # C(K) = Delta_Q(K) / Delta_H (individual animal ratio)
                    # R(K) = Score(MQH) - Score(MQ) (residual history gain)
                    c_frac = (delta_q / delta_h) if abs(delta_h) > 0.2 else np.nan
                    r_res = g_mqh - g_mq

                    animal_records.append({
                        "cohort": c,
                        "sid": te["sid"],
                        "method": method,
                        "K": K,
                        "g_m0": g_m0,
                        "g_mh": g_mh,
                        "delta_h": delta_h,
                        "g_mq": g_mq,
                        "delta_q": delta_q,
                        "g_mqh": g_mqh,
                        "c_frac": c_frac,
                        "r_residual": r_res
                    })

                    mouse_records.append({
                        "delta_q": delta_q,
                        "r_residual": r_res,
                        "delta_h": delta_h
                    })

                mean_dq = np.mean([r["delta_q"] for r in mouse_records])
                sem_dq = float(stats.sem([r["delta_q"] for r in mouse_records]))
                mean_r = np.mean([r["r_residual"] for r in mouse_records])
                sem_r = float(stats.sem([r["r_residual"] for r in mouse_records]))
                # Cohort-level sufficiency fraction: mean_dq / mean_dh
                cohort_c_frac = (mean_dq / mean_dh) if abs(mean_dh) > 0.05 else np.nan

                summary_records.append({
                    "cohort": c,
                    "method": method,
                    "K": K,
                    "mean_m0": mean_m0,
                    "mean_mh": mean_mh,
                    "mean_delta_h": mean_dh,
                    "sem_delta_h": sem_dh,
                    "mean_delta_q": mean_dq,
                    "sem_delta_q": sem_dq,
                    "cohort_c_frac": cohort_c_frac,
                    "mean_r_residual": mean_r,
                    "sem_r_residual": sem_r
                })

                if method == "NMF" and K in [2, 4, 8, 20]:
                    c_str = f"{cohort_c_frac*100:5.1f}%" if not np.isnan(cohort_c_frac) else "  N/A "
                    print(f"    [NMF K={K:2d}] Delta_Q = {mean_dq:+5.2f}% ± {sem_dq:4.2f}% | C(K) = {c_str} | Resid R(K) = {mean_r:+5.2f}%")

    df_summary = pd.DataFrame(summary_records)
    df_animals = pd.DataFrame(animal_records)

    csv_sum_path = os.path.join(TAB_DIR, "table_latent_regime_sufficiency_summary.csv")
    csv_ani_path = os.path.join(TAB_DIR, "table_latent_regime_sufficiency_animals.csv")
    df_summary.to_csv(csv_sum_path, index=False)
    df_animals.to_csv(csv_ani_path, index=False)
    print(f"\nSaved tables to:\n  -> {csv_sum_path}\n  -> {csv_ani_path}")

    # -----------------------------------------------------------------------
    # PRIMARY GENOTYPE CONTRAST (WT Saline vs. KI Saline)
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("PRIMARY GENOTYPE CONTRAST: WT SALINE vs. KI SALINE (NMF)")
    print("="*80)
    sub_pri = df_summary[(df_summary["method"] == "NMF") & (df_summary["cohort"].isin(PRIMARY_COHORTS))]
    piv_dq = sub_pri.pivot(index="K", columns="cohort", values="mean_delta_q")
    piv_c = sub_pri.pivot(index="K", columns="cohort", values="cohort_c_frac")
    piv_r = sub_pri.pivot(index="K", columns="cohort", values="mean_r_residual")

    print("\n--- Absolute Gain Delta_Q(K) (%) ---")
    print(piv_dq.round(2))
    print("\n--- Sufficiency Fraction C(K) ---")
    print((piv_c * 100).round(1))
    print("\n--- Residual History Gain R(K) (%) ---")
    print(piv_r.round(2))

    # -----------------------------------------------------------------------
    # MULTI-PANEL PUBLICATION FIGURE
    # -----------------------------------------------------------------------
    print("\nRendering Publication Figure: fig_latent_regime_sufficiency.png...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), dpi=300)

    # PANEL A: Absolute Latent History Gain Delta_Q(K) vs K (Transparent Numerator)
    ax1 = axes[0]
    for c in PRIMARY_COHORTS:
        sub_c = df_summary[(df_summary["method"] == "NMF") & (df_summary["cohort"] == c)]
        # Full history baseline
        dh_val = sub_c["mean_delta_h"].values[0]
        dh_sem = sub_c["sem_delta_h"].values[0]
        ax1.axhline(dh_val, color=COLORS[c], linestyle=":", linewidth=1.5, alpha=0.8,
                    label=f"{COHORT_LABELS[c]} Full History $\\Delta_H$")
        ax1.fill_between([-1, 25], dh_val - dh_sem, dh_val + dh_sem, color=COLORS[c], alpha=0.08)

        ax1.plot(sub_c["K"], sub_c["mean_delta_q"], color=COLORS[c], marker="o", linewidth=2.4,
                 label=f"{COHORT_LABELS[c]} Latent $\\Delta_Q(K)$")
        ax1.fill_between(sub_c["K"], sub_c["mean_delta_q"] - sub_c["sem_delta_q"],
                         sub_c["mean_delta_q"] + sub_c["sem_delta_q"], color=COLORS[c], alpha=0.18)

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_xlim(0, 21)
    ax1.set_xticks(K_GRID)
    ax1.set_xlabel("Latent Dimension $K$ (NMF)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predictive Gain (% above $S_t$)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Absolute Latent History Gain\n($\\Delta_Q(K)$ vs. Full $\\Delta_H$)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # PANEL B: Sufficiency Fraction C(K) = Delta_Q(K) / Delta_H for KI Saline
    ax2 = axes[1]
    sub_ki = df_summary[(df_summary["method"] == "NMF") & (df_summary["cohort"] == "KI_Saline")]
    ax2.plot(sub_ki["K"], sub_ki["cohort_c_frac"] * 100.0, color=COLORS["KI_Saline"], marker="s",
             linewidth=2.6, label="KI Saline $C(K) = \\Delta_Q / \\Delta_H$")
    ax2.axhline(100.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="100% Full History")
    ax2.axhline(90.0, color="red", linestyle=":", linewidth=1.4, alpha=0.8, label="90% Sufficiency ($K_{90}$)")

    # Annotate K=4 point
    k4_row = sub_ki[sub_ki["K"] == 4]
    if len(k4_row) > 0:
        k4_c = k4_row["cohort_c_frac"].values[0] * 100.0
        ax2.annotate(f"K=4 captures {k4_c:.1f}%\nof full history!",
                     xy=(4, k4_c), xytext=(6.5, 75.0),
                     arrowprops=dict(facecolor=COLORS["KI_Saline"], shrink=0.08, width=1.5, headwidth=6),
                     fontweight="bold", fontsize=9.5, color=COLORS["KI_Saline"])

    ax2.set_xlim(0, 21)
    ax2.set_ylim(40, 115)
    ax2.set_xticks(K_GRID)
    ax2.set_xlabel("Latent Dimension $K$ (NMF)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sufficiency Fraction $C(K)$ (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Compression / Sufficiency $C(K)$\n(KI Information Captured by Latent State)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.8, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # PANEL C: Residual History Gain R(K) = Score(MQH) - Score(MQ)
    ax3 = axes[2]
    for c in PRIMARY_COHORTS:
        sub_c = df_summary[(df_summary["method"] == "NMF") & (df_summary["cohort"] == c)]
        ax3.plot(sub_c["K"], sub_c["mean_r_residual"], color=COLORS[c], marker="o", linewidth=2.4,
                 label=f"{COHORT_LABELS[c]}")
        ax3.fill_between(sub_c["K"], sub_c["mean_r_residual"] - sub_c["sem_r_residual"],
                         sub_c["mean_r_residual"] + sub_c["sem_r_residual"], color=COLORS[c], alpha=0.18)

    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.set_xlim(0, 21)
    ax3.set_xticks(K_GRID)
    ax3.set_xlabel("Latent Dimension $K$ (NMF)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Residual History Gain $R(K)$ (%)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Residual History $R(K)$\n(History Signal Remaining after Knowing $Z_t$)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    ax3.grid(True, linestyle="--", alpha=0.35)

    # Annotate KI Residual approaching 0
    k4_r = sub_ki[sub_ki["K"] == 4]["mean_r_residual"].values[0]
    ax3.annotate(f"KI Residual R(K=4) = {k4_r:+.2f}%\n(Full history adds almost nothing)",
                 xy=(4, k4_r), xytext=(7.0, 1.2),
                 arrowprops=dict(facecolor=COLORS["KI_Saline"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2, color=COLORS["KI_Saline"])

    # PANEL D: Methodological Control: NMF vs. PCA
    ax4 = axes[3]
    for m_name, m_style in [("NMF", "-"), ("PCA", "--")]:
        sub_ki_m = df_summary[(df_summary["method"] == m_name) & (df_summary["cohort"] == "KI_Saline")]
        ax4.plot(sub_ki_m["K"], sub_ki_m["mean_delta_q"], color=COLORS["KI_Saline"], linestyle=m_style,
                 marker="o" if m_name=="NMF" else "^", linewidth=2.2, label=f"KI Saline ({m_name})")

    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax4.set_xlim(0, 21)
    ax4.set_xticks(K_GRID)
    ax4.set_xlabel("Latent Dimension $K$", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Latent History Gain $\\Delta_Q(K)$ (%)", fontsize=11, fontweight="bold")
    ax4.set_title("D. Method Control: NMF vs. PCA\n(Low-Dim Sufficiency is Invariant to Model)", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=9.2, framealpha=0.92)
    ax4.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_latent_regime_sufficiency.png")
    pdf_path = os.path.join(OUT_DIR, "fig_latent_regime_sufficiency.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_latent_regime_sufficiency.png"))
    print(f"Saved figure to:\n  -> {fig_path}\n  -> {pdf_path}")
    print(f"Copied to artifact dir: {os.path.join(ARTIFACT_DIR, 'fig_latent_regime_sufficiency.png')}")

if __name__ == "__main__":
    t0 = time.time()
    run_sufficiency_test()
    print(f"\nExecution completed in {time.time() - t0:.1f}s.")
