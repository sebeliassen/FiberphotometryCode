#!/usr/bin/env python3
"""
run_session_timer_control.py
=============================================================================
SESSION TIMER VS. MACRO-STATE REGIMES: IS CONTEXT JUST HABITUATION/EXPLORATION?
Tests whether the predictive power of the HMM macro-state Q_t is merely a proxy
for elapsed session time (exploration decay, fatigue, or habituation) or whether
Q_t reflects autonomous, recurrent behavioral state dynamics.

MODELS EVALUATED UNDER LEAVE-ONE-ANIMAL-OUT (LOAO):
  M_1:           P(S_{t+1} | S_t)                                      [1st-order baseline]
  M_{time,add}:  P(S_{t+1} | S_t, T_t)                                 [Additive timer]
  M_{time,ker}:  P(S_{t+1} | S_t, T_t, S_t x T_t)                      [Time-varying habituation kernel]
  M_{Q,add}:     P(S_{t+1} | S_t, Q_t)                                 [Additive HMM state]
  M_{QxS}:       P(S_{t+1} | S_t, Q_t, S_t x Q_t)                      [Macro-state transition kernel]
  M_{joint}:     P(S_{t+1} | S_t, Q_t, S_t x Q_t, T_t, S_t x T_t)      [Joint State + Time Kernel]

DECISIVE SCIENTIFIC TESTS:
  1. Gain of Time alone:       NLL(M_1) - NLL(M_{time,ker})
  2. Gain of Macro-State:      NLL(M_1) - NLL(M_{QxS})
  3. Residual Q after Time:    NLL(M_{time,ker}) - NLL(M_{joint})  [Does Q survive after time?]
  4. Residual Time after Q:    NLL(M_{QxS}) - NLL(M_{joint})       [Does time add anything after Q?]
  5. State Recurrence Analysis: Evolution of the 4 macro-states across the 60-min session.

DELIVERABLES:
  - Table: table_session_timer_control.csv
  - Table: table_timer_cohort_summary.csv
  - Table: table_state_recurrence_by_decile.csv
  - Figure: fig_session_timer_vs_macrostate.png / .pdf
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
import torch
import torch.nn as nn
import torch.optim as optim
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from sklearn.decomposition import PCA

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
BLOCK_BOUTS = 20
K_STATES = 4          # 4 discrete macro-states
K_TIME_BASIS = 4      # 4-dimensional temporal RBF basis matching K_STATES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {
    "WT_Saline": "#1f77b4",
    "KI_Saline": "#2ca02c"
}
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "KI_Saline": "KI Saline"
}
REGIME_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

# ---------------------------------------------------------------------------
# TEMPORAL BASIS BUILDER (RBF BASIS OVER SESSION TIME)
# ---------------------------------------------------------------------------
def compute_temporal_basis(normalized_time_arr, n_basis=4, sigma=0.15):
    """
    Constructs an expressive, smooth RBF temporal basis across normalized session time in [0, 1].
    Centers are placed at the quartiles of the session: 0.125, 0.375, 0.625, 0.875.
    Normalized so sum of basis functions equals 1 (partition of unity).
    """
    centers = np.linspace(0.5 / n_basis, 1.0 - 0.5 / n_basis, n_basis)
    diff = normalized_time_arr[:, None] - centers[None, :]  # (N, n_basis)
    rbf = np.exp(- (diff ** 2) / (2.0 * (sigma ** 2)))
    rbf_norm = rbf / np.sum(rbf, axis=1, keepdims=True)
    return rbf_norm.astype(np.float32)

# ---------------------------------------------------------------------------
# DATA PREPARATION (POST-SALINE EPOCH WITH ACCURATE TIMESTAMPS)
# ---------------------------------------------------------------------------
def load_data_with_timestamps():
    print("=" * 80)
    print("LOADING DATA & EXTRACTING ELAPSED SESSION TIMESTAMPS")
    print(f"PyTorch Device: {DEVICE}")
    print("=" * 80)

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

        post_bouts = [b for b in bouts if b["start_frame"] > inj_f]
        M = len(post_bouts)
        n_blocks = M // BLOCK_BOUTS
        if n_blocks < 25:
            continue

        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int64)
        start_frames = np.array([b["start_frame"] for b in post_bouts], dtype=np.float64)

        # Absolute session time elapsed in seconds
        t_start = (start_frames[0] - inj_f) / FPS
        t_end = (post_bouts[-1]["end_frame"] - inj_f) / FPS
        duration_sec = max(t_end - t_start, 1.0)

        # Non-overlapping 20-bout block repertoires for HMM
        block_reps = np.zeros((n_blocks, NUM_SYLLS), dtype=np.float32)
        for b in range(n_blocks):
            b_sylls = sylls[b * BLOCK_BOUTS : (b + 1) * BLOCK_BOUTS]
            counts = np.bincount(b_sylls, minlength=NUM_SYLLS).astype(np.float32)
            block_reps[b] = counts / float(BLOCK_BOUTS)

        # Bout anchors for next-bout ladder (t in [20, M-1))
        anchors = np.arange(BLOCK_BOUTS, M - 1)
        N_anc = len(anchors)

        target_s = sylls[anchors + 1]

        # S_t (one-hot)
        s_curr = np.zeros((N_anc, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(N_anc), sylls[anchors]] = 1.0

        # Normalized session time tau in [0, 1] for each anchor
        anchor_times_sec = (start_frames[anchors] - inj_f) / FPS - t_start
        tau_anchors = np.clip(anchor_times_sec / duration_sec, 0.0, 1.0)

        # 4-dimensional temporal RBF basis matching K=4
        T_basis = compute_temporal_basis(tau_anchors, n_basis=K_TIME_BASIS, sigma=0.15)

        # Block assignment
        block_idx_per_anchor = np.clip(anchors // BLOCK_BOUTS, 0, n_blocks - 1)

        animals.append({
            "session_idx": i,
            "animal_id": aid,
            "cohort": cohort,
            "sylls": sylls,
            "n_blocks": n_blocks,
            "block_reps": block_reps,
            "anchors": anchors,
            "curr_sylls": sylls[anchors],
            "targets": target_s,
            "s_curr": s_curr,
            "tau": tau_anchors,
            "t_basis": T_basis,
            "duration_min": duration_sec / 60.0,
            "block_idx": block_idx_per_anchor,
            "n_samples": N_anc
        })
        print(f"  [{cohort}] Mouse {aid:6s} | {N_anc:,} anchors, duration: {duration_sec/60.0:.1f} min ({duration_sec:.0f}s)")

    return animals

# ---------------------------------------------------------------------------
# FORWARD FILTER IMPLEMENTATION
# ---------------------------------------------------------------------------
def compute_forward_filtering_posteriors(Z_seq, hmm_model):
    B = len(Z_seq)
    K = hmm_model.n_components
    pi = np.maximum(hmm_model.startprob_, 1e-12)
    A = np.maximum(hmm_model.transmat_, 1e-12)
    log_emiss = hmm_model._compute_log_likelihood(Z_seq)

    log_alpha = np.zeros((B, K), dtype=np.float64)
    gamma_filt = np.zeros((B, K), dtype=np.float32)
    gamma_pred = np.zeros((B, K), dtype=np.float32)

    log_alpha[0] = np.log(pi) + log_emiss[0]
    log_alpha[0] -= np.max(log_alpha[0])
    alpha_0 = np.exp(log_alpha[0])
    gamma_filt[0] = alpha_0 / np.sum(alpha_0)
    gamma_pred[0] = pi.astype(np.float32)

    log_A = np.log(A)
    for b in range(1, B):
        gamma_pred[b] = gamma_filt[b - 1] @ A
        prev_log = log_alpha[b - 1][:, None] + log_A
        max_prev = np.max(prev_log, axis=0)
        logsumexp_prev = max_prev + np.log(np.sum(np.exp(prev_log - max_prev), axis=0))
        log_alpha[b] = logsumexp_prev + log_emiss[b]
        m_a = np.max(log_alpha[b])
        norm_a = np.exp(log_alpha[b] - m_a)
        gamma_filt[b] = (norm_a / np.sum(norm_a)).astype(np.float32)

    return gamma_filt, gamma_pred

# ---------------------------------------------------------------------------
# DECODER
# ---------------------------------------------------------------------------
def train_eval_cross_entropy(X_tr, y_tr, X_te, y_te, epochs=35, lr=0.03, wd=1e-4):
    D = X_tr.shape[1]
    C = NUM_SYLLS

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    t_X_te = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    t_y_te = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    model = nn.Linear(D, C).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(t_X_tr)
        loss = crit(logits, t_y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out_te = model(t_X_te)
        log2_p = nn.functional.log_softmax(out_te, dim=-1) / np.log(2.0)
        nll_per_sample = -log2_p[torch.arange(len(t_y_te)), t_y_te]
        mean_nll = float(nll_per_sample.mean().item())

    return mean_nll

# ---------------------------------------------------------------------------
# INTERACTION BUILDER
# ---------------------------------------------------------------------------
def build_interaction_features(s_curr_arr, V_arr):
    N, K = V_arr.shape
    inter = np.zeros((N, NUM_SYLLS * K), dtype=np.float32)
    for idx in range(N):
        syll = s_curr_arr[idx]
        inter[idx, syll * K : (syll + 1) * K] = V_arr[idx]
    return inter

# ---------------------------------------------------------------------------
# MAIN EVALUATION PIPELINE
# ---------------------------------------------------------------------------
def run_session_timer_experiment(animals):
    print("\n" + "=" * 80)
    print("RUNNING SESSION TIMER VS. MACRO-STATE DISCRIMINATION (LOAO)")
    print("=" * 80)

    results = []
    state_occupancies_by_decile = []

    for c in ["WT_Saline", "KI_Saline"]:
        c_animals = [a for a in animals if a["cohort"] == c]
        n_mice = len(c_animals)

        print("\n===========================================================================")
        print(f"Cohort: {COHORT_LABELS[c]} (N={n_mice} mice)")
        print("===========================================================================")

        for te_idx in range(n_mice):
            te_mouse = c_animals[te_idx]
            aid = te_mouse["animal_id"]
            tr_mice = [c_animals[j] for j in range(n_mice) if j != te_idx]

            # 1. Fit HMM on training animals
            X_block_tr = np.vstack([m["block_reps"] for m in tr_mice])
            lens_tr = [m["n_blocks"] for m in tr_mice]

            pca_fold = PCA(n_components=K_STATES, random_state=42)
            Z_block_tr = pca_fold.fit_transform(X_block_tr)
            z_mu = np.mean(Z_block_tr, axis=0, keepdims=True)
            z_sig = np.std(Z_block_tr, axis=0, keepdims=True) + 1e-6
            Z_block_tr = (Z_block_tr - z_mu) / z_sig

            hmm_fold = GaussianHMM(n_components=K_STATES, covariance_type="diag", n_iter=60, random_state=42)
            hmm_fold.fit(Z_block_tr, lens_tr)

            # 2. Forward filter on held-out animal
            X_block_te = te_mouse["block_reps"]
            Z_block_te = (pca_fold.transform(X_block_te) - z_mu) / z_sig
            gamma_filt_te, gamma_pred_te = compute_forward_filtering_posteriors(Z_block_te, hmm_fold)

            b_indices_te = te_mouse["block_idx"]
            Q_soft_te = gamma_pred_te[b_indices_te]  # (N_te, 4)
            T_basis_te = te_mouse["t_basis"]        # (N_te, 4)
            tau_te = te_mouse["tau"]

            # Record state occupancy across 10 deciles of session time
            for dec in range(10):
                mask = (tau_te >= dec * 0.1) & (tau_te < (dec + 1) * 0.1)
                if np.sum(mask) > 0:
                    mean_q_in_dec = np.mean(Q_soft_te[mask], axis=0)
                    for k in range(K_STATES):
                        state_occupancies_by_decile.append({
                            "cohort": c, "animal_id": aid, "decile": dec, "state": f"Q{k+1}",
                            "occupancy": mean_q_in_dec[k]
                        })

            # 3. Construct features for training mice
            tr_Q_soft_list = []
            for m in tr_mice:
                Z_m = (pca_fold.transform(m["block_reps"]) - z_mu) / z_sig
                _, g_pred = compute_forward_filtering_posteriors(Z_m, hmm_fold)
                tr_Q_soft_list.append(g_pred[m["block_idx"]])

            Q_soft_tr = np.vstack(tr_Q_soft_list)
            T_basis_tr = np.vstack([m["t_basis"] for m in tr_mice])

            s_curr_tr = np.vstack([m["s_curr"] for m in tr_mice])
            curr_sylls_tr = np.concatenate([m["curr_sylls"] for m in tr_mice])
            y_tr = np.concatenate([m["targets"] for m in tr_mice])

            s_curr_te = te_mouse["s_curr"]
            curr_sylls_te = te_mouse["curr_sylls"]
            y_te = te_mouse["targets"]

            # Interactions:
            # S_t x Q_t (Macro-state transition kernel)
            inter_q_tr = build_interaction_features(curr_sylls_tr, Q_soft_tr)
            inter_q_te = build_interaction_features(curr_sylls_te, Q_soft_te)

            # S_t x T_t (Time-dependent habituation transition kernel)
            inter_t_tr = build_interaction_features(curr_sylls_tr, T_basis_tr)
            inter_t_te = build_interaction_features(curr_sylls_te, T_basis_te)

            # ---------------------------------------------------------------
            # 4. EVALUATE MODELS
            # ---------------------------------------------------------------
            # M_1: Standard Markov (S_t)
            nll_m1 = train_eval_cross_entropy(s_curr_tr, y_tr, s_curr_te, y_te)

            # M_{time,add}: Additive Session Time (S_t + T_t)
            feat_t_add_tr = np.hstack([s_curr_tr, T_basis_tr])
            feat_t_add_te = np.hstack([s_curr_te, T_basis_te])
            nll_m_time_add = train_eval_cross_entropy(feat_t_add_tr, y_tr, feat_t_add_te, y_te)

            # M_{time,ker}: Time-Conditioned Habituation Kernel (S_t + T_t + S_t x T_t)
            feat_t_ker_tr = np.hstack([s_curr_tr, T_basis_tr, inter_t_tr])
            feat_t_ker_te = np.hstack([s_curr_te, T_basis_te, inter_t_te])
            nll_m_time_ker = train_eval_cross_entropy(feat_t_ker_tr, y_tr, feat_t_ker_te, y_te)

            # M_{Q,add}: Additive HMM Macro-State (S_t + Q_t)
            feat_q_add_tr = np.hstack([s_curr_tr, Q_soft_tr])
            feat_q_add_te = np.hstack([s_curr_te, Q_soft_te])
            nll_m_q_add = train_eval_cross_entropy(feat_q_add_tr, y_tr, feat_q_add_te, y_te)

            # M_{QxS}: State-Conditioned Transition Kernel (S_t + Q_t + S_t x Q_t)
            feat_q_ker_tr = np.hstack([s_curr_tr, Q_soft_tr, inter_q_tr])
            feat_q_ker_te = np.hstack([s_curr_te, Q_soft_te, inter_q_te])
            nll_m_q_ker = train_eval_cross_entropy(feat_q_ker_tr, y_tr, feat_q_ker_te, y_te)

            # M_{joint}: State Kernel + Time Kernel (S_t + Q_t + S_t x Q_t + T_t + S_t x T_t)
            feat_joint_tr = np.hstack([feat_q_ker_tr, T_basis_tr, inter_t_tr])
            feat_joint_te = np.hstack([feat_q_ker_te, T_basis_te, inter_t_te])
            nll_m_joint = train_eval_cross_entropy(feat_joint_tr, y_tr, feat_joint_te, y_te)

            # Metrics
            gain_time_add = nll_m1 - nll_m_time_add
            gain_time_ker = nll_m1 - nll_m_time_ker
            gain_state_add = nll_m1 - nll_m_q_add
            gain_state_ker = nll_m1 - nll_m_q_ker

            # Residual tests:
            residual_q_after_time = nll_m_time_ker - nll_m_joint   # Does Q_t still help after Time is in the kernel?
            residual_time_after_q = nll_m_q_ker - nll_m_joint      # Does Time still help after Q_t is in the kernel?

            results.append({
                "cohort": c,
                "animal_id": aid,
                "n_samples": te_mouse["n_samples"],
                "duration_min": te_mouse["duration_min"],
                "nll_m1": nll_m1,
                "nll_time_add": nll_m_time_add,
                "nll_time_ker": nll_m_time_ker,
                "nll_q_add": nll_m_q_add,
                "nll_q_ker": nll_m_q_ker,
                "nll_joint": nll_m_joint,
                "gain_time_add": gain_time_add,
                "gain_time_ker": gain_time_ker,
                "gain_state_add": gain_state_add,
                "gain_state_ker": gain_state_ker,
                "residual_q_after_time": residual_q_after_time,
                "residual_time_after_q": residual_time_after_q
            })

            print(f"  Mouse {aid:6s} | M1={nll_m1:.3f} | Time_ker={nll_m_time_ker:.3f} (Δ={gain_time_ker:+.3f}b) | "
                  f"State_ker={nll_m_q_ker:.3f} (Δ={gain_state_ker:+.3f}b) | Joint={nll_m_joint:.3f} | "
                  f"Q after Time: {residual_q_after_time:+.3f}b | Time after Q: {residual_time_after_q:+.3f}b")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(TAB_DIR, "table_session_timer_control.csv"), index=False)

    df_occ = pd.DataFrame(state_occupancies_by_decile)
    df_occ.to_csv(os.path.join(TAB_DIR, "table_state_recurrence_by_decile.csv"), index=False)

    # -----------------------------------------------------------------------
    # STATISTICAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY: TIMER VS. STATE KERNEL DISCRIMINATION")
    print("=" * 80)

    summary_rows = []
    for c in ["WT_Saline", "KI_Saline"]:
        sub = df_res[df_res["cohort"] == c]
        n = len(sub)

        m_gain_t, sem_gain_t = np.mean(sub["gain_time_ker"]), stats.sem(sub["gain_time_ker"])
        m_gain_q, sem_gain_q = np.mean(sub["gain_state_ker"]), stats.sem(sub["gain_state_ker"])
        m_res_q, sem_res_q = np.mean(sub["residual_q_after_time"]), stats.sem(sub["residual_q_after_time"])
        m_res_t, sem_res_t = np.mean(sub["residual_time_after_q"]), stats.sem(sub["residual_time_after_q"])

        p_gain_t = stats.ttest_1samp(sub["gain_time_ker"], 0.0).pvalue
        p_gain_q = stats.ttest_1samp(sub["gain_state_ker"], 0.0).pvalue
        p_res_q = stats.ttest_1samp(sub["residual_q_after_time"], 0.0).pvalue
        p_res_t = stats.ttest_1samp(sub["residual_time_after_q"], 0.0).pvalue

        bs_rq = [np.mean(np.random.choice(sub["residual_q_after_time"].values, n, replace=True)) for _ in range(2000)]
        ci_rq = np.percentile(bs_rq, [2.5, 97.5])

        bs_rt = [np.mean(np.random.choice(sub["residual_time_after_q"].values, n, replace=True)) for _ in range(2000)]
        ci_rt = np.percentile(bs_rt, [2.5, 97.5])

        print(f"\n[{COHORT_LABELS[c]}] (N={n} mice):")
        print(f"  * Gain of Session Time Kernel (M_{{time,ker}}):  {m_gain_t:+.4f} ± {sem_gain_t:.4f} b | p = {p_gain_t:.4e}")
        print(f"  * Gain of Macro-State Kernel (M_{{QxS}}):        {m_gain_q:+.4f} ± {sem_gain_q:.4f} b | p = {p_gain_q:.4e}")
        print(f"  * Residual Q_t AFTER Time is in Kernel:        {m_res_q:+.4f} ± {sem_res_q:.4f} b | 95% CI [{ci_rq[0]:.4f}, {ci_rq[1]:.4f}] | p = {p_res_q:.4e}")
        print(f"  * Residual Time AFTER Q_t is in Kernel:        {m_res_t:+.4f} ± {sem_res_t:.4f} b | 95% CI [{ci_rt[0]:.4f}, {ci_rt[1]:.4f}] | p = {p_res_t:.4e}")

        summary_rows.append({
            "cohort": c, "n_mice": n,
            "mean_gain_time_ker": m_gain_t, "sem_gain_time_ker": sem_gain_t, "p_gain_time_ker": p_gain_t,
            "mean_gain_state_ker": m_gain_q, "sem_gain_state_ker": sem_gain_q, "p_gain_state_ker": p_gain_q,
            "mean_res_q_after_time": m_res_q, "sem_res_q_after_time": sem_res_q, "p_res_q_after_time": p_res_q,
            "ci_res_q_low": ci_rq[0], "ci_res_q_high": ci_rq[1],
            "mean_res_time_after_q": m_res_t, "sem_res_time_after_q": sem_res_t, "p_res_time_after_q": p_res_t,
            "ci_res_time_low": ci_rt[0], "ci_res_time_high": ci_rt[1]
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(TAB_DIR, "table_timer_cohort_summary.csv"), index=False)

    render_timer_control_figure(df_res, df_sum, df_occ)

# ---------------------------------------------------------------------------
# FIGURE RENDERING
# ---------------------------------------------------------------------------
def render_timer_control_figure(df_res, df_sum, df_occ):
    print("\nRendering figure: fig_session_timer_vs_macrostate.png...")

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), dpi=300)

    cohorts = ["WT_Saline", "KI_Saline"]
    width = 0.35

    # Panel A: Model Comparison (NLL across model ladder)
    ax0 = axes[0]
    models = ["M1\n(S_t)", "M_{time}\n(Timer Ker)", "M_{QxS}\n(State Ker)", "M_{joint}\n(Both)"]
    x_m = np.arange(len(models))

    for c in cohorts:
        sub = df_res[df_res["cohort"] == c]
        vals = [sub["nll_m1"].values, sub["nll_time_ker"].values, sub["nll_q_ker"].values, sub["nll_joint"].values]
        means = [np.mean(v) for v in vals]
        sems = [stats.sem(v) for v in vals]

        for idx in range(len(sub)):
            ax0.plot(x_m, [vals[s][idx] for s in range(4)], color=COLORS[c], alpha=0.25, lw=1.0)
        ax0.errorbar(x_m, means, yerr=sems, fmt="-o", color=COLORS[c], lw=2.5,
                     markersize=7, capsize=4, label=COHORT_LABELS[c])

    ax0.set_xticks(x_m)
    ax0.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax0.set_ylabel("Held-Out Cross-Entropy NLL (bits/bout)", fontsize=11, fontweight="bold")
    ax0.set_title("A. Timer vs. Macro-State Performance\n[Does Time Kernel Match State Kernel?]", fontsize=11, fontweight="bold")
    ax0.legend(loc="upper right", framealpha=0.9)
    ax0.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Direct Comparison of Predictive Gains
    ax1 = axes[1]
    g_labels = ["Gain of\nTime Kernel", "Gain of\nState Kernel", "State Gain\nAFTER Time", "Time Gain\nAFTER State"]
    x_g = np.arange(len(g_labels))

    for i, c in enumerate(cohorts):
        sub = df_res[df_res["cohort"] == c]
        m_vals = [np.mean(sub["gain_time_ker"]), np.mean(sub["gain_state_ker"]),
                  np.mean(sub["residual_q_after_time"]), np.mean(sub["residual_time_after_q"])]
        s_vals = [stats.sem(sub["gain_time_ker"]), stats.sem(sub["gain_state_ker"]),
                  stats.sem(sub["residual_q_after_time"]), stats.sem(sub["residual_time_after_q"])]

        offset = (i - 0.5) * width
        ax1.bar(x_g + offset, m_vals, yerr=s_vals, width=width, color=COLORS[c],
                alpha=0.85, capsize=4, label=COHORT_LABELS[c])

        for j, col in enumerate(["gain_time_ker", "gain_state_ker", "residual_q_after_time", "residual_time_after_q"]):
            jitter = (np.random.rand(len(sub)) - 0.5) * 0.1
            ax1.scatter(np.full(len(sub), x_g[j] + offset) + jitter, sub[col],
                        color="black", s=18, alpha=0.6, zorder=3)

    ax1.axhline(0, color="gray", linestyle="--", lw=1.2)
    ax1.set_xticks(x_g)
    ax1.set_xticklabels(g_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("NLL Improvement (bits/bout)", fontsize=11, fontweight="bold")
    ax1.set_title("B. Independent Predictive Residuals\n[Q_t vs. Session Habituation Time]", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel C: Macro-State Recurrence across Session Time (WT)
    ax2 = axes[2]
    deciles = np.arange(10) * 10 + 5  # 5%, 15%, ..., 95% of session
    sub_occ_wt = df_occ[df_occ["cohort"] == "WT_Saline"]

    for k in range(K_STATES):
        st_name = f"Q{k+1}"
        st_sub = sub_occ_wt[sub_occ_wt["state"] == st_name]
        mean_by_dec = [np.mean(st_sub[st_sub["decile"] == d]["occupancy"]) for d in range(10)]
        sem_by_dec = [stats.sem(st_sub[st_sub["decile"] == d]["occupancy"]) for d in range(10)]
        ax2.plot(deciles, mean_by_dec, "-o", color=REGIME_COLORS[k], lw=2.0, label=st_name)
        ax2.fill_between(deciles, np.array(mean_by_dec) - np.array(sem_by_dec),
                         np.array(mean_by_dec) + np.array(sem_by_dec), color=REGIME_COLORS[k], alpha=0.15)

    ax2.set_xlabel("Elapsed Session Time (% of 60-min session)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("State Probability / Occupancy", fontsize=11, fontweight="bold")
    ax2.set_title("C. Macro-State Recurrence (WT Saline)\n[States Recur Repeatedly Across Session]", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.6)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel D: Macro-State Recurrence across Session Time (KI)
    ax3 = axes[3]
    sub_occ_ki = df_occ[df_occ["cohort"] == "KI_Saline"]

    for k in range(K_STATES):
        st_name = f"Q{k+1}"
        st_sub = sub_occ_ki[sub_occ_ki["state"] == st_name]
        mean_by_dec = [np.mean(st_sub[st_sub["decile"] == d]["occupancy"]) for d in range(10)]
        sem_by_dec = [stats.sem(st_sub[st_sub["decile"] == d]["occupancy"]) for d in range(10)]
        ax3.plot(deciles, mean_by_dec, "-s", color=REGIME_COLORS[k], lw=2.0, label=st_name)
        ax3.fill_between(deciles, np.array(mean_by_dec) - np.array(sem_by_dec),
                         np.array(mean_by_dec) + np.array(sem_by_dec), color=REGIME_COLORS[k], alpha=0.15)

    ax3.set_xlabel("Elapsed Session Time (% of 60-min session)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("State Probability / Occupancy", fontsize=11, fontweight="bold")
    ax3.set_title("D. Macro-State Recurrence (KI Saline)\n[States Recur Repeatedly Across Session]", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 0.6)
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path_png = os.path.join(OUT_DIR, "fig_session_timer_vs_macrostate.png")
    fig_path_pdf = os.path.join(OUT_DIR, "fig_session_timer_vs_macrostate.pdf")
    plt.savefig(fig_path_png, bbox_inches="tight")
    plt.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path_png, os.path.join(ARTIFACT_DIR, "fig_session_timer_vs_macrostate.png"))
    print(f"Saved figure to: {fig_path_png}")

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    animals = load_data_with_timestamps()
    run_session_timer_experiment(animals)
    print(f"\nExecution finished in {time.time() - t0:.2f}s.")
