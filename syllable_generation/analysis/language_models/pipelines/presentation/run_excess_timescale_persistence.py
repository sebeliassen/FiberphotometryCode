#!/usr/bin/env python3
"""
run_excess_timescale_persistence.py
=============================================================================
AUDITED PERSISTENCE TIMESCALES & EXCESS AUTOCORRELATION SUITE
Tests the Macro-State Coherence Timescale Hypothesis in WT vs. DAT-KI Mice.

Audit Corrections Applied:
  1. Fixed Common Anchor Set across all lags:
     Anchors are strictly restricted to the subset valid for the largest lag (10 min):
       t in [0 ... T - max_lag - h]
     Guarantees that every lag delta in [0 ... 10 min] evaluates on the EXACT SAME
     physical observations, eliminating session-time selection bias.
  2. Incremental History Gain beyond Current Posture S_t:
     Evaluates both total predictive gain and incremental gain strictly beyond S_t:
       Delta_hist(delta) = Gain([S_t, X_{t-delta}]) - Gain(S_t).
  3. Mean-Centered & Excess Autocorrelation:
     rho_excess(delta) = rho_real(delta) - rho_shuffled(delta),
     neutralizing static repertoire entropy confounds.
  4. Epoch Separation:
     Post-saline (primary) and pre-injection baseline (independent replication)
     evaluated independently with zero boundary-crossing.

Deliverables:
  - Table: table_excess_timescale_persistence.csv
  - Table: table_fixed_anchor_lag_survival.csv
  - Figure: fig_excess_timescale_persistence.png / .pdf
=============================================================================
"""

import os
import pickle
import shutil
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/5872c3a8-21c5-4385-94b1-416fd12d3fdb"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FPS = 80.025
NUM_SYLLS = 100
BLOCK_SEC = 15.0       # 15-second non-overlapping physical time blocks
BLOCK_FRAMES = int(BLOCK_SEC * FPS)

# Evaluation lags in seconds (0 to 10 minutes on fixed common anchors)
LAGS_SEC = np.array([0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600])

COLORS = {
    "WT_Saline": "#1f77b4",
    "KI_Saline": "#2ca02c"
}
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "KI_Saline": "KI Saline"
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
# DATA EXTRACTION PER EPOCH
# ---------------------------------------------------------------------------
def extract_blocks_per_epoch(epoch_type="post"):
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    with open(RAW_PATH, "rb") as f:
        raw_sessions = pickle.load(f)

    animals = []

    for i, bouts in enumerate(session_bouts):
        meta = raw_sessions[i].metadata
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        if cohort not in ["WT_Saline", "KI_Saline"]:
            continue

        inj_f = meta.get("injection_frame", 0)
        aid = meta.get("animal_id", f"mouse_{i}")

        if epoch_type == "post":
            epoch_bouts = [b for b in bouts if b["start_frame"] > inj_f]
            f_start = inj_f + int(2.0 * FPS)
            f_end = bouts[-1]["end_frame"]
        else: # 'pre'
            epoch_bouts = [b for b in bouts if b["end_frame"] <= inj_f]
            f_start = bouts[0]["start_frame"]
            f_end = inj_f - int(2.0 * FPS)

        if len(epoch_bouts) < 100 or (f_end - f_start) < 600 * FPS:
            continue

        N_f = f_end - f_start
        frame_sylls = np.zeros((N_f, NUM_SYLLS), dtype=np.float32)
        for b in epoch_bouts:
            st = max(0, b["start_frame"] - f_start)
            ed = min(N_f, b["end_frame"] - f_start)
            if ed > st:
                frame_sylls[st:ed, b["syllable"]] = 1.0

        cum_frame_sylls = np.cumsum(frame_sylls, axis=0)

        n_blocks = int(N_f // BLOCK_FRAMES)
        block_t_sec = np.arange(n_blocks) * BLOCK_SEC
        block_reps = np.zeros((n_blocks, NUM_SYLLS), dtype=np.float32)
        block_last_syll = np.zeros((n_blocks, NUM_SYLLS), dtype=np.float32)

        for b_idx in range(n_blocks):
            b_st = b_idx * BLOCK_FRAMES
            b_ed = (b_idx + 1) * BLOCK_FRAMES
            counts = cum_frame_sylls[b_ed - 1] - (cum_frame_sylls[b_st - 1] if b_st > 0 else 0)
            tot = np.sum(counts)
            if tot > 0:
                block_reps[b_idx] = counts / tot
            else:
                block_reps[b_idx] = 1.0 / NUM_SYLLS

            # Last frame syllable in this block (for S_t baseline)
            last_syll_idx = np.argmax(frame_sylls[b_ed - 1])
            block_last_syll[b_idx, last_syll_idx] = 1.0

        animals.append({
            "session_idx": i,
            "animal_id": aid,
            "cohort": cohort,
            "epoch": epoch_type,
            "block_t_sec": block_t_sec,
            "block_reps": block_reps,
            "block_last_syll": block_last_syll,
            "n_blocks": n_blocks
        })

    return animals

# ---------------------------------------------------------------------------
# COMPUTE EXCESS AUTOCORRELATION & LAG DECAY
# ---------------------------------------------------------------------------
def compute_excess_autocorr(animals):
    results = []

    for a in animals:
        aid = a["animal_id"]
        c = a["cohort"]
        X = a["block_reps"]
        T, K = X.shape

        X_mu = np.mean(X, axis=0, keepdims=True)
        X_mc = X - X_mu

        norm_mc = np.linalg.norm(X_mc, axis=1, keepdims=True) + 1e-6
        X_mc_norm = X_mc / norm_mc

        shift = T // 2
        X_shuff = np.roll(X_mc_norm, shift, axis=0)

        real_rho_per_lag = []
        excess_rho_per_lag = []

        for lag_s in LAGS_SEC:
            lag_steps = int(round(lag_s / BLOCK_SEC))
            if lag_steps >= T - 2:
                real_rho_per_lag.append(np.nan)
                excess_rho_per_lag.append(np.nan)
                continue

            if lag_steps == 0:
                real_rho_per_lag.append(1.0)
                excess_rho_per_lag.append(1.0)
            else:
                cos_mc = np.sum(X_mc_norm[:-lag_steps] * X_mc_norm[lag_steps:], axis=1)
                rho_val = float(np.mean(cos_mc))

                cos_shuff = np.sum(X_mc_norm[:-lag_steps] * X_shuff[lag_steps:], axis=1)
                shuff_val = float(np.mean(cos_shuff))

                excess_val = rho_val - shuff_val
                real_rho_per_lag.append(rho_val)
                excess_rho_per_lag.append(excess_val)

        # Fit exponential decay
        fit_mask = (LAGS_SEC > 0) & (LAGS_SEC <= 600) & (~np.isnan(real_rho_per_lag))
        t_fit = LAGS_SEC[fit_mask]
        y_fit = np.array(real_rho_per_lag)[fit_mask]

        tau_est = np.nan
        half_life_est = np.nan

        if len(y_fit) >= 4 and np.max(y_fit) > 0.05:
            try:
                def exp_decay(t, A, tau):
                    return A * np.exp(-t / tau)
                popt, _ = curve_fit(exp_decay, t_fit, y_fit, p0=[0.6, 60.0], bounds=([0.0, 1.0], [1.5, 3600.0]))
                tau_est = float(popt[1])
                half_life_est = float(popt[1] * np.log(2.0))
            except Exception:
                y_pos = np.maximum(y_fit, 1e-4)
                slope, _, _, _, _ = stats.linregress(t_fit, np.log(y_pos))
                if slope < 0:
                    tau_est = float(-1.0 / slope)
                    half_life_est = float(tau_est * np.log(2.0))

        results.append({
            "animal_id": aid,
            "cohort": c,
            "epoch": a["epoch"],
            "tau_sec": tau_est,
            "half_life_sec": half_life_est,
            "real_rho": real_rho_per_lag,
            "excess_rho": excess_rho_per_lag
        })

    return results

# ---------------------------------------------------------------------------
# AUDITED CONTINUOUS PREDICTIVE SURVIVAL ON FIXED COMMON ANCHORS
# ---------------------------------------------------------------------------
def compute_fixed_anchor_predictive_survival(animals):
    """
    Evaluates predictive survival across lags strictly on a FIXED COMMON ANCHOR SET
    valid up to 10 minutes (600s).
    Evaluates both:
      1. Total Predictive Gain (% Delta D_JS vs. cohort prior)
      2. Incremental History Gain beyond S_t (Delta_hist = Gain(S_t + X) - Gain(S_t))
    """
    pred_lags_sec = np.array([0, 30, 60, 120, 180, 240, 300, 420, 600]) # up to 10 min
    MAX_LAG_SEC = 600.0
    MAX_LAG_STEPS = int(round(MAX_LAG_SEC / BLOCK_SEC))  # 40 blocks
    H_STEPS = int(round(120.0 / BLOCK_SEC))              # 8 blocks (120s future)

    LAMBDA = 10.0
    pred_results = []

    for c in ["WT_Saline", "KI_Saline"]:
        c_animals = [a for a in animals if a["cohort"] == c]
        n_mice = len(c_animals)

        for te_idx in range(n_mice):
            te_mouse = c_animals[te_idx]
            tr_mice = [c_animals[j] for j in range(n_mice) if j != te_idx]

            # Helper to extract fixed common anchors for an animal
            def get_fixed_common_anchors(mouse):
                X = mouse["block_reps"]
                S = mouse["block_last_syll"]
                T = len(X)
                # FIXED COMMON ANCHORS: Valid for ALL lags up to 10 min!
                max_valid_t = T - MAX_LAG_STEPS - H_STEPS
                if max_valid_t < 15:
                    return None
                common_anchors = np.arange(max_valid_t)
                return common_anchors

            # Training data across fixed common anchors
            # 1. Baseline S_t model on training mice (predicting future at lag 0)
            S_tr_list, Y0_tr_list = [], []
            for tr_m in tr_mice:
                anc = get_fixed_common_anchors(tr_m)
                if anc is not None:
                    S_tr_list.append(tr_m["block_last_syll"][anc])
                    # Future 120s at lag 0
                    Y_fut0 = np.zeros_like(tr_m["block_reps"][anc])
                    for step in range(H_STEPS):
                        Y_fut0 += tr_m["block_reps"][anc + step]
                    Y_fut0 /= np.sum(Y_fut0, axis=1, keepdims=True) + 1e-12
                    Y0_tr_list.append(Y_fut0)

            S_tr = np.vstack(S_tr_list)
            Y0_tr = np.vstack(Y0_tr_list)

            # Held-out test mouse anchors
            anc_te = get_fixed_common_anchors(te_mouse)
            S_te = te_mouse["block_last_syll"][anc_te]
            Y0_te = np.zeros_like(te_mouse["block_reps"][anc_te])
            for step in range(H_STEPS):
                Y0_te += te_mouse["block_reps"][anc_te + step]
            Y0_te /= np.sum(Y0_te, axis=1, keepdims=True) + 1e-12

            # Prior on training data
            prior_Y = np.mean(Y0_tr, axis=0, keepdims=True)
            d_prior = float(np.mean(js_divergence(Y0_te, prior_Y)))

            # S_t only model
            beta_s = np.linalg.solve(S_tr.T @ S_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), S_tr.T @ Y0_tr)
            pred_s = np.clip(S_te @ beta_s, 1e-12, 1.0)
            pred_s /= np.sum(pred_s, axis=1, keepdims=True)
            d_s = float(np.mean(js_divergence(Y0_te, pred_s)))
            gain_s_pct = (d_prior - d_s) / d_prior * 100.0

            total_gains_per_lag = []
            incremental_gains_per_lag = []

            for lag_s in pred_lags_sec:
                lag_steps = int(round(lag_s / BLOCK_SEC))

                # Build training pairs for this lag on FIXED common anchors
                X_tr_list, Y_tr_list = [], []
                for tr_m in tr_mice:
                    anc = get_fixed_common_anchors(tr_m)
                    if anc is not None:
                        X_tr_list.append(tr_m["block_reps"][anc])
                        Y_fut = np.zeros_like(tr_m["block_reps"][anc])
                        for step in range(H_STEPS):
                            Y_fut += tr_m["block_reps"][anc + lag_steps + step]
                        Y_fut /= np.sum(Y_fut, axis=1, keepdims=True) + 1e-12
                        Y_tr_list.append(Y_fut)

                X_tr = np.vstack(X_tr_list)
                Y_tr = np.vstack(Y_tr_list)

                # Test pairs on EXACT SAME common anchors
                X_te = te_mouse["block_reps"][anc_te]
                Y_te = np.zeros_like(X_te)
                for step in range(H_STEPS):
                    Y_te += te_mouse["block_reps"][anc_te + lag_steps + step]
                Y_te /= np.sum(Y_te, axis=1, keepdims=True) + 1e-12

                # 1. History-only model
                beta_x = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X_tr.T @ Y_tr)
                pred_x = np.clip(X_te @ beta_x, 1e-12, 1.0)
                pred_x /= np.sum(pred_x, axis=1, keepdims=True)
                d_x = float(np.mean(js_divergence(Y_te, pred_x)))
                total_gain = (d_prior - d_x) / d_prior * 100.0
                total_gains_per_lag.append(total_gain)

                # 2. Joint [S_t, X_hist] model
                XS_tr = np.hstack([S_tr, X_tr])
                XS_te = np.hstack([S_te, X_te])
                beta_xs = np.linalg.solve(XS_tr.T @ XS_tr + LAMBDA * np.eye(XS_tr.shape[1], dtype=np.float32), XS_tr.T @ Y_tr)
                pred_xs = np.clip(XS_te @ beta_xs, 1e-12, 1.0)
                pred_xs /= np.sum(pred_xs, axis=1, keepdims=True)
                d_xs = float(np.mean(js_divergence(Y_te, pred_xs)))
                joint_gain = (d_prior - d_xs) / d_prior * 100.0

                # Incremental gain beyond S_t
                inc_gain = joint_gain - gain_s_pct
                incremental_gains_per_lag.append(inc_gain)

            pred_results.append({
                "animal_id": te_mouse["animal_id"],
                "cohort": c,
                "n_common_anchors": len(anc_te),
                "gain_s_curr": gain_s_pct,
                "total_gains": total_gains_per_lag,
                "incremental_gains": incremental_gains_per_lag
            })

            print(f"  [{c}] Mouse {te_mouse['animal_id']:6s} (N={len(anc_te)} fixed anchors) | "
                  f"Total Gain: lag=0: {total_gains_per_lag[0]:+5.2f}%, 3m: {total_gains_per_lag[4]:+5.2f}%, 10m: {total_gains_per_lag[-1]:+5.2f}% | "
                  f"Inc Gain beyond St: lag=0: {incremental_gains_per_lag[0]:+5.2f}%, 3m: {incremental_gains_per_lag[4]:+5.2f}%, 10m: {incremental_gains_per_lag[-1]:+5.2f}%")

    return pred_results, pred_lags_sec

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_audited_timescale_suite():
    print("=" * 75)
    print("RUNNING AUDITED PERSISTENCE TIMESCALES (FIXED COMMON ANCHORS)")
    print("=" * 75)

    post_animals = extract_blocks_per_epoch("post")
    post_autocorr = compute_excess_autocorr(post_animals)
    post_pred_survival, pred_lags = compute_fixed_anchor_predictive_survival(post_animals)

    pre_animals = extract_blocks_per_epoch("pre")
    pre_autocorr = compute_excess_autocorr(pre_animals)

    # -----------------------------------------------------------------------
    # STATISTICAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STATISTICAL RESULTS: FIXED ANCHOR SURVIVAL & TIMESCALES")
    print("=" * 75)

    df_post_auto = pd.DataFrame(post_autocorr)
    df_pre_auto = pd.DataFrame(pre_autocorr)

    wt_post_tau = df_post_auto[df_post_auto["cohort"] == "WT_Saline"]["tau_sec"].dropna().values
    ki_post_tau = df_post_auto[df_post_auto["cohort"] == "KI_Saline"]["tau_sec"].dropna().values
    u_tau_post, p_tau_post = stats.mannwhitneyu(ki_post_tau, wt_post_tau, alternative="two-sided")
    t_tau_post, p_t_post = stats.ttest_ind(ki_post_tau, wt_post_tau, equal_var=False)

    print("\n[Post-Saline Epoch] Coherence Timescale tau:")
    print(f"  * WT Saline: {np.mean(wt_post_tau):.1f} ± {stats.sem(wt_post_tau):.1f} s")
    print(f"  * KI Saline: {np.mean(ki_post_tau):.1f} ± {stats.sem(ki_post_tau):.1f} s")
    print(f"  * Difference (KI - WT): {np.mean(ki_post_tau) - np.mean(wt_post_tau):+.1f} s | MWU p = {p_tau_post:.4f}")

    wt_pre_tau = df_pre_auto[df_pre_auto["cohort"] == "WT_Saline"]["tau_sec"].dropna().values
    ki_pre_tau = df_pre_auto[df_pre_auto["cohort"] == "KI_Saline"]["tau_sec"].dropna().values
    u_tau_pre, p_tau_pre = stats.mannwhitneyu(ki_pre_tau, wt_pre_tau, alternative="two-sided")

    print("\n[Pre-Injection Baseline Epoch] Coherence Timescale tau (Replication):")
    print(f"  * WT Saline: {np.mean(wt_pre_tau):.1f} ± {stats.sem(wt_pre_tau):.1f} s")
    print(f"  * KI Saline: {np.mean(ki_pre_tau):.1f} ± {stats.sem(ki_pre_tau):.1f} s")
    print(f"  * Difference (KI - WT): {np.mean(ki_pre_tau) - np.mean(wt_pre_tau):+.1f} s | MWU p = {p_tau_pre:.4f}")

    # Save tables
    tab_path = os.path.join(TAB_DIR, "table_excess_timescale_persistence.csv")
    df_post_auto.to_csv(tab_path, index=False)

    # Save fixed anchor survival table
    surv_rows = []
    for r in post_pred_survival:
        for idx_l, l_s in enumerate(pred_lags):
            surv_rows.append({
                "animal_id": r["animal_id"],
                "cohort": r["cohort"],
                "lag_sec": l_s,
                "lag_min": l_s / 60.0,
                "total_gain_pct": r["total_gains"][idx_l],
                "incremental_gain_pct": r["incremental_gains"][idx_l]
            })
    df_surv = pd.DataFrame(surv_rows)
    df_surv.to_csv(os.path.join(TAB_DIR, "table_fixed_anchor_lag_survival.csv"), index=False)
    print("Saved fixed anchor survival tables.")

    # -----------------------------------------------------------------------
    # PLOTTING PUBLICATION FIGURE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("RENDERING PUBLICATION FIGURE: fig_excess_timescale_persistence.png")
    print("=" * 75)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), dpi=300)

    # Panel A: Mean-Centered Autocorrelation Decay
    ax = axes[0]
    for c in ["WT_Saline", "KI_Saline"]:
        c_rows = [r for r in post_autocorr if r["cohort"] == c]
        rhos = np.array([r["real_rho"] for r in c_rows])
        mean_rho = np.nanmean(rhos, axis=0)
        sem_rho = np.nanstd(rhos, axis=0) / np.sqrt(len(c_rows))
        lags_min = LAGS_SEC / 60.0
        ax.plot(lags_min, mean_rho, "o-", color=COLORS[c], linewidth=2.5, label=f"{COHORT_LABELS[c]}")
        ax.fill_between(lags_min, mean_rho - sem_rho, mean_rho + sem_rho, color=COLORS[c], alpha=0.2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Temporal Lag $\delta$ (minutes)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean-Centered Autocorrelation $\\rho(\\delta)$", fontsize=10, fontweight="bold")
    ax.set_title("A. Behavioral Coherence Decay\n(Mean-Centered Repertoire)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    # Panel B: Baseline-Corrected Excess Autocorrelation
    ax = axes[1]
    for c in ["WT_Saline", "KI_Saline"]:
        c_rows = [r for r in post_autocorr if r["cohort"] == c]
        rhos_ex = np.array([r["excess_rho"] for r in c_rows])
        mean_ex = np.nanmean(rhos_ex, axis=0)
        sem_ex = np.nanstd(rhos_ex, axis=0) / np.sqrt(len(c_rows))
        ax.plot(lags_min, mean_ex, "s-", color=COLORS[c], linewidth=2.5, label=COHORT_LABELS[c])
        ax.fill_between(lags_min, mean_ex - sem_ex, mean_ex + sem_ex, color=COLORS[c], alpha=0.2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Temporal Lag $\delta$ (minutes)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Excess Autocorrelation $\\rho_{\\rm excess}(\\delta)$", fontsize=10, fontweight="bold")
    ax.set_title("B. Entropy-Neutralized Persistence\n[Real - Permuted Baseline]", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    # Panel C: Fixed Common Anchor Predictive Survival Curve (Total Gain)
    ax = axes[2]
    p_lags_min = pred_lags / 60.0
    for c in ["WT_Saline", "KI_Saline"]:
        c_preds = [r for r in post_pred_survival if r["cohort"] == c]
        tot_gains = np.array([r["total_gains"] for r in c_preds])
        mean_g = np.nanmean(tot_gains, axis=0)
        sem_g = np.nanstd(tot_gains, axis=0) / np.sqrt(len(c_preds))
        ax.plot(p_lags_min, mean_g, "D-", color=COLORS[c], linewidth=2.5, label=COHORT_LABELS[c])
        ax.fill_between(p_lags_min, mean_g - sem_g, mean_g + sem_g, color=COLORS[c], alpha=0.2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Displacement $\delta$ (minutes)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total History Gain (%)", fontsize=10, fontweight="bold")
    ax.set_title("C. Fixed-Anchor Predictive Survival\n(Identical Observations Across Lags)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    # Panel D: Incremental History Gain beyond S_t (Fixed Common Anchors)
    ax = axes[3]
    for c in ["WT_Saline", "KI_Saline"]:
        c_preds = [r for r in post_pred_survival if r["cohort"] == c]
        inc_gains = np.array([r["incremental_gains"] for r in c_preds])
        mean_inc = np.nanmean(inc_gains, axis=0)
        sem_inc = np.nanstd(inc_gains, axis=0) / np.sqrt(len(c_preds))
        ax.plot(p_lags_min, mean_inc, "^-", color=COLORS[c], linewidth=2.5, label=COHORT_LABELS[c])
        ax.fill_between(p_lags_min, mean_inc - sem_inc, mean_inc + sem_inc, color=COLORS[c], alpha=0.2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Displacement $\delta$ (minutes)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Incremental Gain beyond $S_t$ (%)", fontsize=10, fontweight="bold")
    ax.set_title("D. Incremental History Survival\n[$\\Delta_{\\rm hist}(\\delta) = {\\rm Gain}(S_t + X_\\delta) - {\\rm Gain}(S_t)$]", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_excess_timescale_persistence.png")
    fig_pdf = os.path.join(OUT_DIR, "fig_excess_timescale_persistence.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")

    try:
        shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_excess_timescale_persistence.png"))
    except Exception as e:
        print("Note on copy:", e)

# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    run_audited_timescale_suite()
    print(f"\nAudited Phase 2 Pipeline completed successfully in {time.time() - t0:.2f}s.")
