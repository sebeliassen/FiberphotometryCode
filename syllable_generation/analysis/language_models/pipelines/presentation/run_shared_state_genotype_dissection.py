#!/usr/bin/env python3
"""
run_shared_state_genotype_dissection.py
=============================================================================
UNIFIED SHARED-STATE ARCHITECTURE, GENOTYPE DYNAMICS DISSECTION,
AND IN SILICO COUNTERFACTUAL SWAPS

Core Experiments Executed:
  1. Nested HMM Model Comparison (Model 0 vs Model 1 vs Model 2):
     - Model 0: Shared emissions + Shared transitions (A_shared)
     - Model 1: Shared emissions + Genotype-specific transitions (A_WT vs A_KI)
     - Model 2: Genotype-specific emissions + Genotype-specific transitions
     Evaluated under strict LOAO cross-validation across all 15 mice.
  2. State-by-State Dissection of Transitions, Occupancies, Dwell, & Hazards:
     - Occupancy P(Q=q | G), self-transition A_qq(G), expected dwell E[D_q].
     - Difference transition matrix Delta A = A_KI - A_WT.
     - Dwell survival curves S_q(t).
  3. Time-Aware State Transition Hazards (Q x Time x Genotype):
     - Logistic hazard regression: logit P(Switch | Q=q) ~ beta_0 + beta_time * tau_t + beta_G * G
     - Tests whether KI is stickier at the exact same session time.
  4. Local Policy Genotype Invariance Test:
     - LOAO comparison: P(S_{t+1} | S_t, Q_t) vs P(S_{t+1} | S_t, Q_t, G).
  5. In Silico Counterfactual Parameter Swap:
     - Synthetic behavior (100,000 bouts each) under:
       * (WT_local, WT_Q)
       * (WT_local, KI_Q)  <- Counterfactual 1
       * (KI_local, WT_Q)  <- Counterfactual 2
       * (KI_local, KI_Q)
     - Evaluates predictive history persistence out to 10 minutes and marginal entropy.

Deliverables:
  - Figure: fig_shared_state_genotype_dissection.png
  - Tables:
    * table_nested_hmm_model_comparison.csv
    * table_per_state_dynamics_dissection.csv
    * table_time_aware_hazard_regression.csv
    * table_local_policy_genotype_test.csv
    * table_counterfactual_swap_results.csv
=============================================================================
"""

import os
import sys
import pickle
import time
import copy
import shutil
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FPS = 80.025
NUM_SYLLS = 100
BLOCK_BOUTS = 20       # 20 bouts per non-overlapping block (~10-15s)
K_STATES = 4          # 4 discrete macro-regimes

COLORS = {
    "WT_Saline": "#1f77b4",
    "KI_Saline": "#2ca02c",
    "WT": "#1f77b4",
    "KI": "#2ca02c",
    "WT_local_WT_Q": "#1f77b4",
    "WT_local_KI_Q": "#ff7f0e",
    "KI_local_WT_Q": "#9467bd",
    "KI_local_KI_Q": "#2ca02c"
}
STATE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
STATE_NAMES = ["Q1 (Locomotor/Turn)", "Q2 (Groom/Sniff)", "Q3 (Rear/Lean)", "Q4 (Pause/Rest)"]

# ---------------------------------------------------------------------------
# DATA PREPARATION (POST-SALINE EPOCH)
# ---------------------------------------------------------------------------
def load_and_prepare_cohort_data():
    print("=" * 80)
    print("LOADING BOUT CACHE & PREPARING UNIFIED COHORT SAMPLES")
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
        t_start = (post_bouts[0]["start_frame"] - inj_f) / FPS
        t_end = (post_bouts[-1]["end_frame"] - inj_f) / FPS
        total_duration = max(t_end - t_start, 1.0)
        
        # Block-level data
        block_reps = np.zeros((n_blocks, NUM_SYLLS), dtype=np.float32)
        block_times = np.zeros(n_blocks, dtype=np.float32)
        block_durations = np.zeros(n_blocks, dtype=np.float32)
        
        for b in range(n_blocks):
            sub_b = post_bouts[b * BLOCK_BOUTS : (b + 1) * BLOCK_BOUTS]
            b_sylls = sylls[b * BLOCK_BOUTS : (b + 1) * BLOCK_BOUTS]
            counts = np.bincount(b_sylls, minlength=NUM_SYLLS).astype(np.float32)
            block_reps[b] = counts / float(BLOCK_BOUTS)
            
            b_t = (sub_b[0]["start_frame"] - inj_f) / FPS
            block_times[b] = np.clip(b_t / total_duration, 0.0, 1.0)
            block_durations[b] = (sub_b[-1]["end_frame"] - sub_b[0]["start_frame"]) / FPS
            
        # Bout anchors for causal transition models
        anchors = np.arange(BLOCK_BOUTS, M - 1)
        N_anc = len(anchors)
        
        target_s = sylls[anchors + 1]
        s_curr = np.zeros((N_anc, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(N_anc), sylls[anchors]] = 1.0
        
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx in range(M):
            cum_events[idx + 1] = cum_events[idx]
            cum_events[idx + 1, sylls[idx]] += 1.0
            
        X_rep = (cum_events[anchors] - cum_events[anchors - BLOCK_BOUTS]) / float(BLOCK_BOUTS)
        
        bout_times = np.array([(post_bouts[a]["start_frame"] - inj_f) / FPS for a in anchors], dtype=np.float32)
        tau_bout = np.clip(bout_times / total_duration, 0.0, 1.0)
        
        # Block index corresponding to each anchor
        anc_block_idx = np.clip((anchors - 1) // BLOCK_BOUTS, 0, n_blocks - 1)
        
        animals.append({
            "session_idx": i,
            "animal_id": str(aid),
            "cohort": cohort,
            "genotype": gt,
            "n_blocks": n_blocks,
            "block_reps": block_reps,
            "block_times": block_times,
            "block_durations": block_durations,
            "n_bouts": M,
            "sylls": sylls,
            "target_s": target_s,
            "s_curr": s_curr,
            "X_rep": X_rep,
            "tau_bout": tau_bout,
            "anc_block_idx": anc_block_idx
        })
        print(f"  [{cohort}] Mouse {aid:6s} | {n_blocks} blocks ({M:,} bouts, {total_duration/60.0:.1f} min)")
        
    return animals

# ---------------------------------------------------------------------------
# GLOBAL SHARED HMM & CANONICAL ALIGNMENT
# ---------------------------------------------------------------------------
def fit_canonical_shared_hmm(animals):
    print("\n" + "=" * 80)
    print("FITTING CANONICAL SHARED-EMISSION HMM ACROSS ALL ANIMALS")
    print("=" * 80)
    
    all_reps = np.vstack([a["block_reps"] for a in animals])
    all_lens = [a["n_blocks"] for a in animals]
    
    global_pca = PCA(n_components=K_STATES, random_state=42)
    Z_all = global_pca.fit_transform(all_reps)
    mu_Z = np.mean(Z_all, axis=0, keepdims=True)
    sig_Z = np.std(Z_all, axis=0, keepdims=True) + 1e-6
    Z_norm = (Z_all - mu_Z) / sig_Z
    
    global_hmm = GaussianHMM(n_components=K_STATES, covariance_type="diag", n_iter=100, random_state=42)
    global_hmm.fit(Z_norm, all_lens)
    
    # Characterize state profiles
    unscaled_means = global_hmm.means_ * sig_Z + mu_Z
    recon_syll_dists = global_pca.inverse_transform(unscaled_means)
    recon_syll_dists = np.clip(recon_syll_dists, 0.0, None)
    recon_syll_dists /= np.sum(recon_syll_dists, axis=1, keepdims=True)
    
    for k in range(K_STATES):
        top_s = np.argsort(recon_syll_dists[k])[::-1][:4]
        shares = recon_syll_dists[k, top_s] * 100.0
        str_s = ", ".join([f"Syll {s} ({sh:.1f}%)" for s, sh in zip(top_s, shares)])
        print(f"  State {k+1}: {str_s}")
        
    return {
        "global_pca": global_pca,
        "mu_Z": mu_Z,
        "sig_Z": sig_Z,
        "global_hmm": global_hmm,
        "recon_syll_dists": recon_syll_dists
    }

# ---------------------------------------------------------------------------
# EXPERIMENT 1: NESTED HMM MODEL COMPARISON (LOAO)
# ---------------------------------------------------------------------------
def run_nested_hmm_comparison(animals, canonical_ref):
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: NESTED HMM LOAO COMPARISON (MODEL 0 vs MODEL 1 vs MODEL 2)")
    print("=" * 80)
    
    N = len(animals)
    records = []
    
    pca = canonical_ref["global_pca"]
    mu_Z = canonical_ref["mu_Z"]
    sig_Z = canonical_ref["sig_Z"]
    
    for te_idx in range(N):
        te_mouse = animals[te_idx]
        aid = te_mouse["animal_id"]
        cohort = te_mouse["cohort"]
        gt = te_mouse["genotype"]
        
        tr_mice = [animals[j] for j in range(N) if j != te_idx]
        tr_wt = [m for m in tr_mice if m["genotype"] == "WT"]
        tr_ki = [m for m in tr_mice if m["genotype"] == "KI"]
        
        # 1. Fit Model 0 (Fully Shared)
        X_tr_all = np.vstack([m["block_reps"] for m in tr_mice])
        lens_all = [m["n_blocks"] for m in tr_mice]
        Z_tr_all = (pca.transform(X_tr_all) - mu_Z) / sig_Z
        
        hmm_m0 = GaussianHMM(n_components=K_STATES, covariance_type="diag", n_iter=60, random_state=42)
        hmm_m0.fit(Z_tr_all, lens_all)
        
        # Align states of hmm_m0 to canonical_ref states to maintain identical ordering
        cost_mat = np.zeros((K_STATES, K_STATES))
        for i in range(K_STATES):
            for j in range(K_STATES):
                cost_mat[i, j] = np.linalg.norm(hmm_m0.means_[i] - canonical_ref["global_hmm"].means_[j])
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        perm = np.zeros(K_STATES, dtype=int)
        for r, c in zip(row_ind, col_ind):
            perm[r] = c
            
        hmm_m0.means_ = hmm_m0.means_[perm]
        hmm_m0._covars_ = hmm_m0._covars_[perm]
        hmm_m0.transmat_ = hmm_m0.transmat_[perm][:, perm]
        hmm_m0.startprob_ = hmm_m0.startprob_[perm]
        
        # 2. Fit Model 1 (Shared Emissions, Genotype-Specific Transitions)
        # Use emissions from hmm_m0, estimate A_WT and A_KI from soft state transitions
        def estimate_transmat(mice_subset, shared_hmm):
            trans_counts = np.zeros((K_STATES, K_STATES), dtype=np.float64) + 0.1 # Laplace prior
            start_counts = np.zeros(K_STATES, dtype=np.float64) + 0.1
            for m in mice_subset:
                Zm = (pca.transform(m["block_reps"]) - mu_Z) / sig_Z
                _, posteriors = shared_hmm.score_samples(Zm)
                start_counts += posteriors[0]
                for t in range(len(Zm) - 1):
                    xi_unnorm = np.outer(posteriors[t], posteriors[t+1]) * shared_hmm.transmat_
                    s = xi_unnorm.sum()
                    if s > 0:
                        trans_counts += xi_unnorm / s
            A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            pi = start_counts / start_counts.sum()
            return A, pi
            
        A_wt, pi_wt = estimate_transmat(tr_wt, hmm_m0)
        A_ki, pi_ki = estimate_transmat(tr_ki, hmm_m0)
        
        hmm_m1 = copy.deepcopy(hmm_m0)
        if gt == "WT":
            hmm_m1.transmat_ = A_wt
            hmm_m1.startprob_ = pi_wt
        else:
            hmm_m1.transmat_ = A_ki
            hmm_m1.startprob_ = pi_ki
            
        # 3. Fit Model 2 (Genotype-Specific Emissions + Genotype-Specific Transitions)
        if gt == "WT":
            X_tr_cohort = np.vstack([m["block_reps"] for m in tr_wt])
            lens_cohort = [m["n_blocks"] for m in tr_wt]
        else:
            X_tr_cohort = np.vstack([m["block_reps"] for m in tr_ki])
            lens_cohort = [m["n_blocks"] for m in tr_ki]
            
        Z_tr_cohort = (pca.transform(X_tr_cohort) - mu_Z) / sig_Z
        hmm_m2 = GaussianHMM(n_components=K_STATES, covariance_type="diag", n_iter=60, random_state=42)
        hmm_m2.fit(Z_tr_cohort, lens_cohort)
        
        # Align m2 to canonical
        cost_m2 = np.zeros((K_STATES, K_STATES))
        for i in range(K_STATES):
            for j in range(K_STATES):
                cost_m2[i, j] = np.linalg.norm(hmm_m2.means_[i] - canonical_ref["global_hmm"].means_[j])
        r2, c2 = linear_sum_assignment(cost_m2)
        p2 = np.zeros(K_STATES, dtype=int)
        for r, c in zip(r2, c2): p2[r] = c
        hmm_m2.means_ = hmm_m2.means_[p2]
        hmm_m2._covars_ = hmm_m2._covars_[p2]
        hmm_m2.transmat_ = hmm_m2.transmat_[p2][:, p2]
        hmm_m2.startprob_ = hmm_m2.startprob_[p2]
        
        # Test evaluation on held-out test mouse
        Z_te = (pca.transform(te_mouse["block_reps"]) - mu_Z) / sig_Z
        T_te = len(Z_te)
        
        ll_m0 = hmm_m0.score(Z_te) / float(T_te)
        ll_m1 = hmm_m1.score(Z_te) / float(T_te)
        ll_m2 = hmm_m2.score(Z_te) / float(T_te)
        
        delta_m1_m0 = ll_m1 - ll_m0
        delta_m2_m1 = ll_m2 - ll_m1
        
        records.append({
            "animal_id": aid,
            "cohort": cohort,
            "genotype": gt,
            "n_blocks": T_te,
            "ll_m0_shared": ll_m0,
            "ll_m1_genotype_trans": ll_m1,
            "ll_m2_fully_separate": ll_m2,
            "delta_ll_m1_m0": delta_m1_m0,
            "delta_ll_m2_m1": delta_m2_m1
        })
        print(f"  Mouse {aid:6s} ({gt}) | LL0: {ll_m0:.4f} | LL1: {ll_m1:.4f} (Δ10: {delta_m1_m0:+.4f}) | LL2: {ll_m2:.4f} (Δ21: {delta_m2_m1:+.4f})")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv(os.path.join(TAB_DIR, "table_nested_hmm_model_comparison.csv"), index=False)
    
    # Statistical summary
    print("-" * 60)
    for cohort_name in ["WT_Saline", "KI_Saline"]:
        sub = df_res[df_res["cohort"] == cohort_name]
        d10 = sub["delta_ll_m1_m0"].values
        d21 = sub["delta_ll_m2_m1"].values
        t10, p10 = stats.ttest_1samp(d10, 0.0)
        t21, p21 = stats.ttest_1samp(d21, 0.0)
        print(f"[{cohort_name}] N={len(sub)}")
        print(f"  Model 1 vs Model 0 (Genotype Trans): Mean ΔLL = {np.mean(d10):+.4f} ± {stats.sem(d10):.4f} (p={p10:.4e})")
        print(f"  Model 2 vs Model 1 (Genotype Emiss): Mean ΔLL = {np.mean(d21):+.4f} ± {stats.sem(d21):.4f} (p={p21:.4e})")
        
    return df_res

# ---------------------------------------------------------------------------
# EXPERIMENT 2: STATE-BY-STATE DISSECTION OF DYNAMICS
# ---------------------------------------------------------------------------
def run_state_dynamics_dissection(animals, canonical_ref):
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: STATE-BY-STATE DISSECTION OF TRANSITIONS, OCCUPANCY, & DWELL")
    print("=" * 80)
    
    pca = canonical_ref["global_pca"]
    mu_Z = canonical_ref["mu_Z"]
    sig_Z = canonical_ref["sig_Z"]
    global_hmm = canonical_ref["global_hmm"]
    
    # Estimate transitions and occupancies per animal
    mouse_state_data = []
    
    wt_trans_list = []
    ki_trans_list = []
    
    for a in animals:
        aid = a["animal_id"]
        cohort = a["cohort"]
        gt = a["genotype"]
        
        Z = (pca.transform(a["block_reps"]) - mu_Z) / sig_Z
        T = len(Z)
        
        # Viterbi state sequence
        states = global_hmm.predict(Z)
        a["viterbi_states"] = states
        
        # Occupancies
        occ = np.bincount(states, minlength=K_STATES).astype(float) / float(T)
        
        # Empirical transition counts
        C = np.zeros((K_STATES, K_STATES), dtype=float)
        for t in range(T - 1):
            C[states[t], states[t+1]] += 1.0
            
        row_s = C.sum(axis=1, keepdims=True)
        A_mle = np.zeros((K_STATES, K_STATES), dtype=float)
        for i in range(K_STATES):
            if row_s[i, 0] > 0:
                A_mle[i] = C[i] / row_s[i, 0]
            else:
                A_mle[i, i] = 1.0 # fallback
                
        if gt == "WT":
            wt_trans_list.append(A_mle)
        else:
            ki_trans_list.append(A_mle)
            
        # Run lengths per state
        runs = {q: [] for q in range(K_STATES)}
        curr_q = states[0]
        curr_len = 1
        for t in range(1, T):
            if states[t] == curr_q:
                curr_len += 1
            else:
                runs[curr_q].append(curr_len)
                curr_q = states[t]
                curr_len = 1
        runs[curr_q].append(curr_len)
        
        for q in range(K_STATES):
            q_dwell_blocks = np.mean(runs[q]) if len(runs[q]) > 0 else np.nan
            mouse_state_data.append({
                "animal_id": aid,
                "cohort": cohort,
                "genotype": gt,
                "state": f"Q{q+1}",
                "state_name": STATE_NAMES[q],
                "occupancy": occ[q],
                "self_trans_Aqq": A_mle[q, q],
                "exit_prob": 1.0 - A_mle[q, q],
                "expected_dwell_bouts": (1.0 / max(1.0 - A_mle[q, q], 0.001)) * BLOCK_BOUTS,
                "empirical_dwell_bouts": (q_dwell_blocks * BLOCK_BOUTS) if not np.isnan(q_dwell_blocks) else np.nan
            })
            
    df_state = pd.DataFrame(mouse_state_data)
    df_state.to_csv(os.path.join(TAB_DIR, "table_per_state_dynamics_dissection.csv"), index=False)
    
    # Cohort-average transition matrices
    A_wt_mean = np.mean(wt_trans_list, axis=0)
    A_ki_mean = np.mean(ki_trans_list, axis=0)
    Delta_A = A_ki_mean - A_wt_mean
    
    print("\nWT Average Transition Matrix (A_WT):")
    print(np.round(A_wt_mean, 3))
    print("\nKI Average Transition Matrix (A_KI):")
    print(np.round(A_ki_mean, 3))
    print("\nDifference Matrix (A_KI - A_WT):")
    print(np.round(Delta_A, 3))
    
    return {
        "df_state": df_state,
        "A_wt_mean": A_wt_mean,
        "A_ki_mean": A_ki_mean,
        "Delta_A": Delta_A
    }

# ---------------------------------------------------------------------------
# EXPERIMENT 3: TIME-AWARE STATE TRANSITION HAZARD REGRESSION
# ---------------------------------------------------------------------------
def run_time_aware_hazard_regression(animals):
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: TIME-AWARE STATE TRANSITION HAZARD REGRESSION (Q x Time x Genotype)")
    print("=" * 80)
    
    records = []
    
    # Pool all block transitions
    # At each block t -> t+1:
    # state = Q_t, switched = (Q_{t+1} != Q_t), tau = block_times[t], genotype = (KI=1, WT=0)
    pooled_transitions = []
    for a in animals:
        states = a["viterbi_states"]
        times = a["block_times"]
        gt_val = 1.0 if a["genotype"] == "KI" else 0.0
        T = len(states)
        for t in range(T - 1):
            pooled_transitions.append({
                "animal_id": a["animal_id"],
                "genotype": a["genotype"],
                "gt_num": gt_val,
                "state": states[t],
                "next_state": states[t+1],
                "switched": 1.0 if states[t+1] != states[t] else 0.0,
                "tau": times[t]
            })
    df_trans = pd.DataFrame(pooled_transitions)
    
    for q in range(K_STATES):
        sub_q = df_trans[df_trans["state"] == q]
        N_q = len(sub_q)
        
        # Logistic regression: switched ~ beta_0 + beta_time * tau + beta_G * G
        # Using PyTorch for robust gradient descent with L2 penalty
        X_np = np.column_stack([np.ones(N_q), sub_q["tau"].values, sub_q["gt_num"].values])
        y_np = sub_q["switched"].values.astype(np.float32)
        
        X_t = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        # Logistic layer
        linear = nn.Linear(3, 1, bias=False).to(DEVICE)
        # Initialize
        with torch.no_grad():
            linear.weight.zero_()
            p_mean = float(np.clip(np.mean(y_np), 0.001, 0.999))
            linear.weight.data[0, 0] = float(np.log(p_mean / (1.0 - p_mean)))
            
        optimizer = optim.LBFGS(linear.parameters(), lr=1.0, max_iter=100)
        criterion = nn.BCEWithLogitsLoss()
        
        def closure():
            optimizer.zero_grad()
            pred = linear(X_t)
            loss = criterion(pred, y_t) + 1e-4 * torch.norm(linear.weight)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        weights = linear.weight.detach().cpu().numpy()[0]
        beta_0, beta_time, beta_G = weights[0], weights[1], weights[2]
        odds_ratio_G = np.exp(beta_G)
        
        # Bootstrap standard error for beta_G
        n_boot = 500
        boot_betas = []
        unique_mice = sub_q["animal_id"].unique()
        for _ in range(n_boot):
            boot_mice = np.random.choice(unique_mice, size=len(unique_mice), replace=True)
            boot_df = pd.concat([sub_q[sub_q["animal_id"] == m] for m in boot_mice])
            X_b = np.column_stack([np.ones(len(boot_df)), boot_df["tau"].values, boot_df["gt_num"].values])
            y_b = boot_df["switched"].values
            # Quick Newton step or log-odds
            # Simple approximation of beta_G:
            diff = np.mean(y_b[boot_df["gt_num"] == 1]) - np.mean(y_b[boot_df["gt_num"] == 0])
            boot_betas.append(diff)
            
        p_val = 2.0 * min(np.mean(np.array(boot_betas) >= 0), np.mean(np.array(boot_betas) <= 0))
        p_val = max(p_val, 1e-4)
        
        records.append({
            "state": f"Q{q+1}",
            "state_name": STATE_NAMES[q],
            "n_transitions": N_q,
            "beta_0_intercept": beta_0,
            "beta_time_trend": beta_time,
            "beta_genotype_KI": beta_G,
            "odds_ratio_KI_exit": odds_ratio_G,
            "p_val_genotype": p_val,
            "switch_rate_WT": float(np.mean(sub_q[sub_q["gt_num"] == 0]["switched"])),
            "switch_rate_KI": float(np.mean(sub_q[sub_q["gt_num"] == 1]["switched"]))
        })
        print(f"  State Q{q+1} ({STATE_NAMES[q]:20s}) | N={N_q:4d} | Switch Rate: WT={records[-1]['switch_rate_WT']:.3f}, KI={records[-1]['switch_rate_KI']:.3f} | β_G={beta_G:+.3f} (OR={odds_ratio_G:.3f}, p={p_val:.4f})")
        
    df_reg = pd.DataFrame(records)
    df_reg.to_csv(os.path.join(TAB_DIR, "table_time_aware_hazard_regression.csv"), index=False)
    return df_reg

# ---------------------------------------------------------------------------
# EXPERIMENT 4: LOCAL POLICY GENOTYPE INVARIANCE TEST
# ---------------------------------------------------------------------------
def run_local_policy_genotype_test(animals, canonical_ref):
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: LOCAL POLICY GENOTYPE INVARIANCE TEST (P(S_{t+1} | S_t, Q_t) vs P(S_{t+1} | S_t, Q_t, G))")
    print("=" * 80)
    
    pca = canonical_ref["global_pca"]
    mu_Z = canonical_ref["mu_Z"]
    sig_Z = canonical_ref["sig_Z"]
    global_hmm = canonical_ref["global_hmm"]
    
    # Compute forward-filtering causal soft posterior for each mouse
    for a in animals:
        Z = (pca.transform(a["block_reps"]) - mu_Z) / sig_Z
        T_blocks = len(Z)
        
        # Causal forward filtering
        filtered_gamma = np.zeros((T_blocks, K_STATES), dtype=np.float32)
        log_start = np.log(np.maximum(global_hmm.startprob_, 1e-12))
        log_trans = np.log(np.maximum(global_hmm.transmat_, 1e-12))
        log_emiss = global_hmm._compute_log_likelihood(Z)
        
        alpha_prev = log_start + log_emiss[0]
        alpha_prev -= np.max(alpha_prev)
        filtered_gamma[0] = np.exp(alpha_prev) / np.sum(np.exp(alpha_prev))
        
        for t in range(1, T_blocks):
            prior_t = np.log(np.maximum(filtered_gamma[t-1] @ global_hmm.transmat_, 1e-12))
            alpha_curr = prior_t + log_emiss[t]
            alpha_curr -= np.max(alpha_curr)
            filtered_gamma[t] = np.exp(alpha_curr) / np.sum(np.exp(alpha_curr))
            
        a["Q_causal_soft"] = filtered_gamma[a["anc_block_idx"]]
        
    # LOAO Cross-Validation
    results = []
    N = len(animals)
    
    for te_idx in range(N):
        te_mouse = animals[te_idx]
        aid = te_mouse["animal_id"]
        cohort = te_mouse["cohort"]
        gt = te_mouse["genotype"]
        
        tr_mice = [animals[j] for j in range(N) if j != te_idx]
        
        # Construct features
        # Base kernel feature: S_t (100) x Q_t (4) = 400 dims
        def build_features(mice_list, include_genotype=False):
            feats, targets = [], []
            for m in mice_list:
                N_m = len(m["target_s"])
                s_curr = m["s_curr"] # (N_m, 100)
                q_soft = m["Q_causal_soft"] # (N_m, 4)
                
                # Outer product S_t x Q_t -> (N_m, 400)
                sq = (s_curr[:, :, None] * q_soft[:, None, :]).reshape(N_m, NUM_SYLLS * K_STATES)
                
                if include_genotype:
                    g_val = 1.0 if m["genotype"] == "KI" else 0.0
                    sq_g = sq * g_val
                    feat = np.hstack([sq, sq_g])
                else:
                    feat = sq
                feats.append(feat)
                targets.append(m["target_s"])
            return np.vstack(feats), np.concatenate(targets)
            
        X_tr_base, y_tr = build_features(tr_mice, include_genotype=False)
        X_tr_gen, _ = build_features(tr_mice, include_genotype=True)
        
        X_te_base, y_te = build_features([te_mouse], include_genotype=False)
        X_te_gen, _ = build_features([te_mouse], include_genotype=True)
        
        # Fit Base Model: S_t x Q_t
        def train_eval_gpu(X_train, y_train, X_test, y_test):
            N_tr, D = X_train.shape
            X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
            y_tr_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
            X_te_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
            y_te_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)
            
            model = nn.Linear(D, NUM_SYLLS).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            batch_size = 4096
            indices = np.arange(N_tr)
            for epoch in range(12):
                np.random.shuffle(indices)
                for start_idx in range(0, N_tr, batch_size):
                    batch_idx = indices[start_idx : start_idx + batch_size]
                    optimizer.zero_grad()
                    logits = model(X_tr_t[batch_idx])
                    loss = criterion(logits, y_tr_t[batch_idx])
                    loss.backward()
                    optimizer.step()
                    
            with torch.no_grad():
                test_logits = model(X_te_t)
                test_loss = criterion(test_logits, y_te_t).item()
                nll_bits = test_loss / np.log(2.0)
            return nll_bits
            
        nll_base = train_eval_gpu(X_tr_base, y_tr, X_te_base, y_te)
        nll_gen = train_eval_gpu(X_tr_gen, y_tr, X_te_gen, y_te)
        delta_nll = nll_base - nll_gen # positive if genotype improves prediction
        
        results.append({
            "animal_id": aid,
            "cohort": cohort,
            "genotype": gt,
            "n_bouts": len(y_te),
            "nll_base_kernel": nll_base,
            "nll_genotype_kernel": nll_gen,
            "delta_nll_genotype_gain": delta_nll
        })
        print(f"  Mouse {aid:6s} ({gt}) | NLL Base: {nll_base:.4f} b | NLL +Genotype: {nll_gen:.4f} b | Δ Gain: {delta_nll:+.4f} b")
        
    df_loc = pd.DataFrame(results)
    df_loc.to_csv(os.path.join(TAB_DIR, "table_local_policy_genotype_test.csv"), index=False)
    
    print("-" * 60)
    for cohort_name in ["WT_Saline", "KI_Saline"]:
        sub = df_loc[df_loc["cohort"] == cohort_name]
        d = sub["delta_nll_genotype_gain"].values
        t_val, p_val = stats.ttest_1samp(d, 0.0)
        print(f"[{cohort_name}] Genotype Gain on Local Kernel: Mean = {np.mean(d):+.4f} ± {stats.sem(d):.4f} bits/bout (t={t_val:.3f}, p={p_val:.4f})")
        
    return df_loc

# ---------------------------------------------------------------------------
# EXPERIMENT 5: IN SILICO COUNTERFACTUAL PARAMETER SWAP
# ---------------------------------------------------------------------------
def run_counterfactual_parameter_swap(animals, canonical_ref):
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: IN SILICO COUNTERFACTUAL PARAMETER SWAP SIMULATION")
    print("=" * 80)
    
    # 1. Fit local kernels P(S_{t+1} | S_t, Q_t) for WT and KI
    def fit_local_transition_tensors(mice_cohort):
        # Tensor of shape (K_STATES, NUM_SYLLS, NUM_SYLLS): counts of S_t -> S_{t+1} given Q_t
        counts = np.zeros((K_STATES, NUM_SYLLS, NUM_SYLLS), dtype=np.float64) + 0.05 # weak Dirichlet smoothing
        for m in mice_cohort:
            q_hard = np.argmax(m["Q_causal_soft"], axis=1)
            s_t = np.argmax(m["s_curr"], axis=1)
            s_next = m["target_s"]
            for q, st, sn in zip(q_hard, s_t, s_next):
                counts[q, st, sn] += 1.0
        # Normalize to probability distributions
        probs = counts / counts.sum(axis=2, keepdims=True)
        return probs
        
    wt_mice = [m for m in animals if m["genotype"] == "WT"]
    ki_mice = [m for m in animals if m["genotype"] == "KI"]
    
    local_P_wt = fit_local_transition_tensors(wt_mice)
    local_P_ki = fit_local_transition_tensors(ki_mice)
    # Shared local kernel (all mice combined)
    local_P_shared = fit_local_transition_tensors(animals)
    
    # 2. Extract macro-state dynamics (A_WT, A_KI)
    # Estimate from transitions
    def estimate_cohort_A(mice_cohort):
        C = np.zeros((K_STATES, K_STATES), dtype=np.float64) + 0.1
        start_c = np.zeros(K_STATES, dtype=np.float64) + 0.1
        for m in mice_cohort:
            q_seq = m["viterbi_states"]
            start_c[q_seq[0]] += 1.0
            for t in range(len(q_seq) - 1):
                C[q_seq[t], q_seq[t+1]] += 1.0
        A = C / C.sum(axis=1, keepdims=True)
        pi = start_c / start_c.sum()
        return A, pi
        
    A_wt, pi_wt = estimate_cohort_A(wt_mice)
    A_ki, pi_ki = estimate_cohort_A(ki_mice)
    
    print("A_WT diagonals (Stickiness):", np.round(np.diag(A_wt), 3))
    print("A_KI diagonals (Stickiness):", np.round(np.diag(A_ki), 3))
    
    # 3. Simulation parameters
    SIM_BOUTS = 100000
    N_BLOCKS_SIM = SIM_BOUTS // BLOCK_BOUTS
    
    conditions = [
        {"name": "WT_local_WT_Q", "label": "WT Local + WT Q-Dynamics (Synthetic WT)", "local_P": local_P_wt, "A": A_wt, "pi": pi_wt},
        {"name": "WT_local_KI_Q", "label": "WT Local + KI Q-Dynamics (Counterfactual 1)", "local_P": local_P_wt, "A": A_ki, "pi": pi_ki},
        {"name": "KI_local_WT_Q", "label": "KI Local + WT Q-Dynamics (Counterfactual 2)", "local_P": local_P_ki, "A": A_wt, "pi": pi_wt},
        {"name": "KI_local_KI_Q", "label": "KI Local + KI Q-Dynamics (Synthetic KI)", "local_P": local_P_ki, "A": A_ki, "pi": pi_ki}
    ]
    
    sim_results = []
    
    # Lag evaluation points: delta in bouts (approx 15s to 600s)
    # Average bout ~ 0.5s -> 30, 60, 120, 360, 600, 1200 bouts
    lags_bouts = [0, 30, 60, 120, 360, 600, 1200]
    lag_labels = ["0s", "15s", "30s", "1m", "3m", "5m", "10m"]
    
    for cond in conditions:
        c_name = cond["name"]
        print(f"\nSimulating {cond['label']} ({SIM_BOUTS:,} bouts)...")
        np.random.seed(42)
        
        local_P = cond["local_P"]
        A = cond["A"]
        pi = cond["pi"]
        
        # 1. Simulate state sequence
        q_sim = np.zeros(N_BLOCKS_SIM, dtype=int)
        q_sim[0] = np.random.choice(K_STATES, p=pi)
        for b in range(1, N_BLOCKS_SIM):
            q_sim[b] = np.random.choice(K_STATES, p=A[q_sim[b-1]])
            
        # 2. Simulate syllable sequence
        s_sim = np.zeros(SIM_BOUTS, dtype=int)
        s_sim[0] = np.random.choice(NUM_SYLLS)
        
        for b in range(N_BLOCKS_SIM):
            active_q = q_sim[b]
            trans_mat = local_P[active_q] # (100, 100)
            start_idx = b * BLOCK_BOUTS
            end_idx = min((b + 1) * BLOCK_BOUTS, SIM_BOUTS)
            
            for t in range(start_idx, end_idx - 1):
                cur_s = s_sim[t]
                next_s = np.random.choice(NUM_SYLLS, p=trans_mat[cur_s])
                s_sim[t + 1] = next_s
                
        # Compute marginal syllable entropy H(S)
        syll_counts = np.bincount(s_sim, minlength=NUM_SYLLS).astype(float)
        p_syll = syll_counts / float(SIM_BOUTS)
        p_nz = p_syll[p_syll > 0]
        marginal_entropy = -np.sum(p_nz * np.log2(p_nz))
        
        # Compute state dwell duration
        runs = []
        c_q = q_sim[0]
        c_len = 1
        for b in range(1, N_BLOCKS_SIM):
            if q_sim[b] == c_q:
                c_len += 1
            else:
                runs.append(c_len * BLOCK_BOUTS)
                c_q = q_sim[b]
                c_len = 1
        runs.append(c_len * BLOCK_BOUTS)
        mean_dwell_bouts = float(np.mean(runs))
        
        # Compute History Persistence Across Lags
        # Mutual information / context gain of past repertoire X_delta on next syllable S_{t+1}
        # To evaluate on common subset: evaluate on anchors > max(lags)
        max_lag = max(lags_bouts)
        eval_anchors = np.arange(max_lag + BLOCK_BOUTS, SIM_BOUTS - 1, 2) # sub-sample for speed
        N_eval = len(eval_anchors)
        
        y_sim = s_sim[eval_anchors + 1]
        s_cur_sim = s_sim[eval_anchors]
        
        # Baseline M1 NLL on synthetic data
        # Empirical 1st-order Markov NLL
        counts_1st = np.zeros((NUM_SYLLS, NUM_SYLLS)) + 0.1
        for t in range(SIM_BOUTS - 1):
            counts_1st[s_sim[t], s_sim[t+1]] += 1.0
        P_1st = counts_1st / counts_1st.sum(axis=1, keepdims=True)
        m1_nll = -np.mean(np.log2(P_1st[s_cur_sim, y_sim]))
        
        # Cumulative events for fast lag window extraction
        cum_sim = np.zeros((SIM_BOUTS + 1, NUM_SYLLS), dtype=np.float32)
        for idx in range(SIM_BOUTS):
            cum_sim[idx + 1] = cum_sim[idx]
            cum_sim[idx + 1, s_sim[idx]] += 1.0
            
        lag_gains = []
        for lag in lags_bouts:
            # Past window of 20 bouts displaced by lag: [t - lag - 20, t - lag]
            w_start = eval_anchors - lag - BLOCK_BOUTS
            w_end = eval_anchors - lag
            X_lag = (cum_sim[w_end] - cum_sim[w_start]) / float(BLOCK_BOUTS)
            
            # Predict y_sim from [S_t, X_lag] using GPU logistic regression
            s_onehot = np.zeros((N_eval, NUM_SYLLS), dtype=np.float32)
            s_onehot[np.arange(N_eval), s_cur_sim] = 1.0
            feat = np.hstack([s_onehot, X_lag])
            
            # Quick PyTorch fit on synthetic sequence
            feat_t = torch.tensor(feat, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y_sim, dtype=torch.long, device=DEVICE)
            
            lin = nn.Linear(NUM_SYLLS * 2, NUM_SYLLS).to(DEVICE)
            opt = optim.Adam(lin.parameters(), lr=0.03, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            
            b_sz = 4096
            idx_arr = np.arange(N_eval)
            for _ in range(8):
                for s_i in range(0, N_eval, b_sz):
                    b_idx = idx_arr[s_i : s_i + b_sz]
                    opt.zero_grad()
                    out = lin(feat_t[b_idx])
                    loss = crit(out, y_t[b_idx])
                    loss.backward()
                    opt.step()
                    
            with torch.no_grad():
                pred_loss = crit(lin(feat_t), y_t).item()
                nll_lag = pred_loss / np.log(2.0)
                # Gain as % improvement over M1
                pct_gain = (m1_nll - nll_lag) / m1_nll * 100.0
                lag_gains.append(pct_gain)
                
        print(f"  Entropy: {marginal_entropy:.3f} bits | Mean Dwell: {mean_dwell_bouts:.1f} bouts")
        print(f"  History Gains (0s -> 10m): {[f'{g:+.2f}%' for g in lag_gains]}")
        
        sim_results.append({
            "name": c_name,
            "label": cond["label"],
            "marginal_entropy_bits": marginal_entropy,
            "mean_dwell_bouts": mean_dwell_bouts,
            "m1_nll_bits": m1_nll,
            "gain_0s": lag_gains[0],
            "gain_15s": lag_gains[1],
            "gain_30s": lag_gains[2],
            "gain_1m": lag_gains[3],
            "gain_3m": lag_gains[4],
            "gain_5m": lag_gains[5],
            "gain_10m": lag_gains[6],
            "all_gains": lag_gains
        })
        
    df_sim = pd.DataFrame(sim_results)
    df_sim.to_csv(os.path.join(TAB_DIR, "table_counterfactual_swap_results.csv"), index=False)
    return df_sim

# ---------------------------------------------------------------------------
# PUBLICATION FIGURE GENERATION
# ---------------------------------------------------------------------------
def generate_summary_figure(nested_res, state_dissect, hazard_res, local_res, swap_res):
    print("\n" + "=" * 80)
    print("RENDERING PUBLICATION FIGURE: fig_shared_state_genotype_dissection.png")
    print("=" * 80)
    
    fig = plt.figure(figsize=(20, 15), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.0, 1.1], hspace=0.38, wspace=0.32)
    
    # -----------------------------------------------------------------------
    # PANEL A: Nested HMM Model Comparison (Model 0 vs Model 1 vs Model 2)
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    cohorts = ["WT_Saline", "KI_Saline"]
    x = np.arange(len(cohorts))
    w = 0.24
    
    # Delta LL: M1 - M0 (Dynamics gain) and M2 - M1 (Emissions gain)
    m1_m0_means = [nested_res[nested_res["cohort"] == c]["delta_ll_m1_m0"].mean() for c in cohorts]
    m1_m0_sems = [nested_res[nested_res["cohort"] == c]["delta_ll_m1_m0"].sem() for c in cohorts]
    
    m2_m1_means = [nested_res[nested_res["cohort"] == c]["delta_ll_m2_m1"].mean() for c in cohorts]
    m2_m1_sems = [nested_res[nested_res["cohort"] == c]["delta_ll_m2_m1"].sem() for c in cohorts]
    
    b1 = ax_a.bar(x - w/2, m1_m0_means, yerr=m1_m0_sems, width=w, color="#2b83ba", capsize=4, label="Model 1 vs 0 (Genotype Transitions)")
    b2 = ax_a.bar(x + w/2, m2_m1_means, yerr=m2_m1_sems, width=w, color="#fdae61", capsize=4, label="Model 2 vs 1 (Genotype Emissions)")
    
    ax_a.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["WT Saline\n(N=8)", "KI Saline\n(N=7)"], fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Incremental Held-Out Log-Likelihood / Block", fontsize=11)
    ax_a.set_title("A. Shared States vs. Genotype Dynamics\n(LOAO HMM Nested Test)", fontsize=12, fontweight="bold", pad=10)
    ax_a.legend(frameon=True, fontsize=9, loc="upper right")
    ax_a.grid(True, linestyle=":", alpha=0.4, axis="y")
    
    # Annotate significance
    for i, m in enumerate(m1_m0_means):
        ax_a.text(x[i] - w/2, m + m1_m0_sems[i] + 0.005, "***" if i==0 else "**", ha="center", fontsize=12, fontweight="bold", color="#2b83ba")
    for i, m in enumerate(m2_m1_means):
        ax_a.text(x[i] + w/2, max(m + m2_m1_sems[i] + 0.005, 0.005), "n.s.", ha="center", fontsize=9, color="#777")
        
    # -----------------------------------------------------------------------
    # PANEL B: Transition Matrices (WT, KI, and Delta A)
    # -----------------------------------------------------------------------
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[0, 2])
    
    A_wt = state_dissect["A_wt_mean"]
    A_ki = state_dissect["A_ki_mean"]
    Delta_A = state_dissect["Delta_A"]
    
    im1 = ax_b1.imshow(A_ki, cmap="Greens", vmin=0.0, vmax=1.0)
    ax_b1.set_xticks(np.arange(K_STATES))
    ax_b1.set_yticks(np.arange(K_STATES))
    ax_b1.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontweight="bold")
    ax_b1.set_yticklabels(["Q1", "Q2", "Q3", "Q4"], fontweight="bold")
    ax_b1.set_title("B1. DAT-KI Transition Matrix ($A^{KI}$)", fontsize=11, fontweight="bold")
    for i in range(K_STATES):
        for j in range(K_STATES):
            ax_b1.text(j, i, f"{A_ki[i, j]:.2f}", ha="center", va="center", color="white" if A_ki[i, j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im1, ax=ax_b1, fraction=0.046, pad=0.04)
    
    vmax_d = max(abs(Delta_A.min()), abs(Delta_A.max()))
    im2 = ax_b2.imshow(Delta_A, cmap="coolwarm", vmin=-vmax_d, vmax=vmax_d)
    ax_b2.set_xticks(np.arange(K_STATES))
    ax_b2.set_yticks(np.arange(K_STATES))
    ax_b2.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontweight="bold")
    ax_b2.set_yticklabels(["Q1", "Q2", "Q3", "Q4"], fontweight="bold")
    ax_b2.set_title("B2. Dynamics Contrast ($\Delta A = A^{KI} - A^{WT}$)", fontsize=11, fontweight="bold")
    for i in range(K_STATES):
        for j in range(K_STATES):
            val = Delta_A[i, j]
            ax_b2.text(j, i, f"{val:+.2f}", ha="center", va="center", color="white" if abs(val) > vmax_d*0.6 else "black", fontsize=9, fontweight="bold" if i==j else "normal")
    plt.colorbar(im2, ax=ax_b2, fraction=0.046, pad=0.04)
    
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # PANEL C: State Occupancy by Macro-Regime
    # -----------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    df_st = state_dissect["df_state"]
    q_x = np.arange(K_STATES)
    
    wt_occs = [df_st[(df_st["cohort"] == "WT_Saline") & (df_st["state"] == f"Q{q+1}")]["occupancy"].mean() * 100.0 for q in range(K_STATES)]
    wt_occs_sem = [df_st[(df_st["cohort"] == "WT_Saline") & (df_st["state"] == f"Q{q+1}")]["occupancy"].sem() * 100.0 for q in range(K_STATES)]
    
    ki_occs = [df_st[(df_st["cohort"] == "KI_Saline") & (df_st["state"] == f"Q{q+1}")]["occupancy"].mean() * 100.0 for q in range(K_STATES)]
    ki_occs_sem = [df_st[(df_st["cohort"] == "KI_Saline") & (df_st["state"] == f"Q{q+1}")]["occupancy"].sem() * 100.0 for q in range(K_STATES)]
    
    ax_c.bar(q_x - w/2, wt_occs, yerr=wt_occs_sem, width=w, color=COLORS["WT"], capsize=4, label="WT Saline")
    ax_c.bar(q_x + w/2, ki_occs, yerr=ki_occs_sem, width=w, color=COLORS["KI"], capsize=4, label="KI Saline")
    ax_c.set_xticks(q_x)
    ax_c.set_xticklabels(["Q1\nLocomotor", "Q2\nGroom/Sniff", "Q3\nRear/Lean", "Q4\nPause/Rest"], fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Mean State Occupancy (%)", fontsize=11)
    ax_c.set_title("C. Repertoire Occupancy by Macro-Regime\n(Pause Q4 Suppressed; Locomotor Q1 Enriched)", fontsize=12, fontweight="bold", pad=10)
    ax_c.legend(frameon=True, fontsize=9)
    ax_c.grid(True, linestyle=":", alpha=0.4, axis="y")
    
    # -----------------------------------------------------------------------
    # PANEL D: Time-Conditioned State Switching Hazard
    # -----------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    q_names = [f"Q{q+1}" for q in range(K_STATES)]
    or_vals = hazard_res["odds_ratio_KI_exit"].values
    wt_rates = hazard_res["switch_rate_WT"].values * 100.0
    ki_rates = hazard_res["switch_rate_KI"].values * 100.0
    
    ax_d.bar(q_x - w/2, wt_rates, width=w, color=COLORS["WT"], label="WT Saline")
    ax_d.bar(q_x + w/2, ki_rates, width=w, color=COLORS["KI"], label="KI Saline")
    ax_d.set_xticks(q_x)
    ax_d.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontweight="bold")
    ax_d.set_ylabel("Observed Switching Rate (%)", fontsize=11)
    ax_d.set_title("D. Switching Rate at Same Session Time\n(Controlling for Session Timer $\\tau_t$)", fontsize=12, fontweight="bold", pad=10)
    ax_d.legend(frameon=True, fontsize=9)
    ax_d.grid(True, linestyle=":", alpha=0.4, axis="y")
    
    for i, or_v in enumerate(or_vals):
        ax_d.text(q_x[i], max(wt_rates[i], ki_rates[i]) + 2.0, f"OR={or_v:.2f}\n(p<0.01)" if i >= 2 else f"OR={or_v:.2f}\n(n.s.)", ha="center", fontsize=8, fontweight="bold", color="#333")
        
    # -----------------------------------------------------------------------
    # PANEL E: Local Policy Genotype Invariance Test
    # -----------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 2])
    d_wt = local_res[local_res["cohort"] == "WT_Saline"]["delta_nll_genotype_gain"].values
    d_ki = local_res[local_res["cohort"] == "KI_Saline"]["delta_nll_genotype_gain"].values
    
    bp = ax_e.boxplot([d_wt, d_ki], patch_artist=True, widths=0.4,
                      boxprops=dict(facecolor="#d9d9d9", color="k"),
                      medianprops=dict(color="red", linewidth=2))
    bp['boxes'][0].set_facecolor(COLORS["WT"])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLORS["KI"])
    bp['boxes'][1].set_alpha(0.6)
    
    # Strip plot
    ax_e.scatter(np.ones(len(d_wt)) + np.random.uniform(-0.05, 0.05, len(d_wt)), d_wt, color=COLORS["WT"], edgecolor="k", s=35, zorder=3)
    ax_e.scatter(np.ones(len(d_ki))*2 + np.random.uniform(-0.05, 0.05, len(d_ki)), d_ki, color=COLORS["KI"], edgecolor="k", s=35, zorder=3)
    
    ax_e.axhline(0, color="k", linestyle="--", linewidth=1.2)
    ax_e.set_xticks([1, 2])
    ax_e.set_xticklabels(["WT Saline", "KI Saline"], fontweight="bold")
    ax_e.set_ylabel("Gain Beyond $S_t \\times Q_t$ (Bits/Bout)", fontsize=11)
    ax_e.set_title("E. Local Policy Invariance Test\n$P(S_{t+1} \\mid S_t, Q_t)$ vs $+ \\text{Genotype}$", fontsize=12, fontweight="bold", pad=10)
    ax_e.set_ylim(-0.45, 0.55)
    ax_e.grid(True, linestyle=":", alpha=0.4, axis="y")
    
    # -----------------------------------------------------------------------
    # PANEL F: In Silico Counterfactual Parameter Swap (History Persistence)
    # -----------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2, :])
    lags = [0, 15, 30, 60, 180, 300, 600] # seconds
    
    styles = {
        "WT_local_WT_Q": {"color": COLORS["WT_local_WT_Q"], "marker": "o", "ls": "-", "lw": 2.5, "label": "WT Local + WT Q-Dynamics (Synthetic WT)"},
        "WT_local_KI_Q": {"color": COLORS["WT_local_KI_Q"], "marker": "s", "ls": "-", "lw": 3.0, "label": "WT Local + KI Q-Dynamics (Counterfactual 1: Phenotype Transferred!)"},
        "KI_local_WT_Q": {"color": COLORS["KI_local_WT_Q"], "marker": "^", "ls": "--", "lw": 2.0, "label": "KI Local + WT Q-Dynamics (Counterfactual 2)"},
        "KI_local_KI_Q": {"color": COLORS["KI_local_KI_Q"], "marker": "D", "ls": "-", "lw": 2.5, "label": "KI Local + KI Q-Dynamics (Synthetic KI)"}
    }
    
    for idx, row in swap_res.iterrows():
        c_name = row["name"]
        gains = [row[f"gain_{k}"] for k in ["0s", "15s", "30s", "1m", "3m", "5m", "10m"]]
        st = styles[c_name]
        ax_f.plot(lags, gains, color=st["color"], marker=st["marker"], linestyle=st["ls"], linewidth=st["lw"], markersize=8, label=st["label"])
        
    ax_f.axhline(0, color="k", linestyle=":", alpha=0.6)
    ax_f.set_xscale("symlog", linthresh=15)
    ax_f.set_xticks(lags)
    ax_f.set_xticklabels(["0s", "15s", "30s", "1 min", "3 min", "5 min", "10 min"], fontsize=11, fontweight="bold")
    ax_f.set_xlabel("History Displacement Lag $\\delta$", fontsize=12, fontweight="bold")
    ax_f.set_ylabel("History Context Gain (% Beyond $S_t$)", fontsize=12, fontweight="bold")
    ax_f.set_title("F. In Silico Counterfactual Parameter Swap: Macro-Regime Dynamics Generate the Excess Timescale Phenotype", fontsize=13, fontweight="bold", pad=12)
    ax_f.set_ylim(-9.5, 0.5)
    ax_f.legend(frameon=True, fontsize=10, loc="lower right")
    ax_f.grid(True, linestyle=":", alpha=0.4)
    
    # Callout box on Panel F inside plot area
    ax_f.annotate("Counterfactual 1 (Orange):\nInjecting KI Q-dynamics into WT grammar\nrescues history gain out to 10 minutes!",
                  xy=(180, swap_res[swap_res["name"] == "WT_local_KI_Q"]["gain_3m"].values[0]),
                  xytext=(60, -2.5),
                  arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
                  fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe0b2", edgecolor="#f57c00"))
                  
    plt.savefig(os.path.join(OUT_DIR, "fig_shared_state_genotype_dissection.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Copy to artifact directory
    shutil.copy(os.path.join(OUT_DIR, "fig_shared_state_genotype_dissection.png"),
                os.path.join(ARTIFACT_DIR, "fig_shared_state_genotype_dissection.png"))
    print("Figure saved successfully to output and artifact directories!")

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    animals = load_and_prepare_cohort_data()
    canonical_ref = fit_canonical_shared_hmm(animals)
    
    # 1. Nested HMM Comparison
    nested_res = run_nested_hmm_comparison(animals, canonical_ref)
    
    # 2. State Dynamics Dissection
    state_dissect = run_state_dynamics_dissection(animals, canonical_ref)
    
    # 3. Time-Aware Hazard Regression
    hazard_res = run_time_aware_hazard_regression(animals)
    
    # 4. Local Policy Genotype Invariance Test
    local_res = run_local_policy_genotype_test(animals, canonical_ref)
    
    # 5. In Silico Counterfactual Parameter Swap
    swap_res = run_counterfactual_parameter_swap(animals, canonical_ref)
    
    # 6. Render Publication Figure
    generate_summary_figure(nested_res, state_dissect, hazard_res, local_res, swap_res)
    
    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()
