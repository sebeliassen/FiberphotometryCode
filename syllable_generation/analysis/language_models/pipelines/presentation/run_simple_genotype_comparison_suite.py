#!/usr/bin/env python3
"""
run_simple_genotype_comparison_suite.py
=============================================================================
SIMPLE GENOTYPE COMPARISONS SUITE USING THE SHARED Q1..Q4 COORDINATE SYSTEM

Methodological Principles:
  - Preserve strict Leave-One-Animal-Out (LOAO) / Leave-One-Mouse-Out (LOMO).
  - Mouse is the fundamental, independent unit of inference (N=15 mice).
  - Keep states Q1..Q4 strictly unlabeled throughout.
  - Do not treat the coordinate system itself as a biological finding.
  - Report exact effect sizes, mouse-level SEM, 95% CIs, and permutation p-values.

Comparison Suite (8 Analyses):
  1. Plain State Occupancy by Genotype (Mouse-level P(Q), omnibus + state-specific)
  2. Time-Resolved Occupancy (P(Q_t=q | G, session time), decile curves, G x Time test)
  3. Entry Rate vs Exit Rate Decomposition (Entry deficit vs exit excess)
  4. Routing After Exit (P(Q_{t+1}=j | Q_t=i, exit, G) with mouse-level permutation)
  5. Within-State Syllable Composition (P(S | Q=q, G), Jensen-Shannon Divergence)
  6. Within-State Local Transitions (P(S_{t+1} | S_t, Q=q, G), targeted cross-transfer)
  7. State-Age Trajectories by Genotype (P(S_{t+1} | S_t, Q, A, G), G x A tests)
  8. Genotype Decoder (Can state usage identify genotype? LOMO feature ablation)
=============================================================================
"""

import os
import sys
import pickle
import time
import copy
import shutil
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
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

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "run12_continuous_bout_cache.pkl")
RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation", "data", "jonathan_amph_sessions_run12_w_sylls.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/e3532183-ee20-47bf-86a2-d66c7d4e288d"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

from math import comb as n_choose_k

def exact_or_single_perm(g1, g2, n_mc=10000):
    """
    Computes exact combinatorial permutation test when total combinations <= 10,000,
    or a single-shuffle complementary Monte Carlo permutation test otherwise.
    Never calls np.random.permutation twice on the same null draw.
    """
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    n1 = len(g1)
    n2 = len(g2)
    diff_obs = float(np.mean(g2) - np.mean(g1))
    comb = np.concatenate([g1, g2])
    n_total = n1 + n2
    
    total_combos = n_choose_k(n_total, n1)
    if total_combos <= 10000:
        combos = np.array(list(itertools.combinations(range(n_total), n1)))
        all_idx = set(range(n_total))
        comp_combos = np.array([list(all_idx - set(c)) for c in combos])
        perm_diffs = comb[comp_combos].mean(axis=1) - comb[combos].mean(axis=1)
        p_val = float((1.0 + np.sum(np.abs(perm_diffs) >= np.abs(diff_obs) - 1e-9)) / (total_combos + 1.0))
        return p_val, diff_obs
    else:
        count_ge = 0
        for _ in range(n_mc):
            shuff = np.random.permutation(comb)
            perm_diff = np.mean(shuff[n1:]) - np.mean(shuff[:n1])
            if abs(perm_diff) >= abs(diff_obs) - 1e-9:
                count_ge += 1
        p_val = float((1.0 + count_ge) / (n_mc + 1.0))
        return p_val, diff_obs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FPS = 80.025
NUM_SYLLS = 100
BLOCK_BOUTS = 20       # 20 bouts per block
K_STATES = 4          # Q1..Q4
Q_NAMES = ["Q1", "Q2", "Q3", "Q4"]

COLORS = {
    "WT_Saline": "#1f77b4",
    "KI_Saline": "#2ca02c",
    "WT": "#1f77b4",
    "KI": "#2ca02c"
}
Q_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

# ---------------------------------------------------------------------------
# 0. LOAD & PREPARE COHORT DATA (IDENTICAL TO CANONICAL REFERENCE)
# ---------------------------------------------------------------------------
def load_and_prepare_cohort_data():
    print("=" * 80)
    print("LOADING BOUT CACHE & PREPARING UNIFIED COHORT DATA")
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
        aid = str(meta.get("animal_id", f"mouse_{i}"))
        
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
            
        # Bout anchors for next-bout prediction
        anchors = np.arange(BLOCK_BOUTS, M - 1)
        N_anc = len(anchors)
        
        target_s = sylls[anchors + 1]
        
        # Current syllable S_t (one-hot)
        s_curr = np.zeros((N_anc, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(N_anc), sylls[anchors]] = 1.0
        
        # Prior posture S_{t-1} (one-hot)
        s_prev = np.zeros((N_anc, NUM_SYLLS), dtype=np.float32)
        s_prev[np.arange(N_anc), sylls[anchors - 1]] = 1.0
        
        # Elapsed session time
        bout_times = np.array([(post_bouts[a]["start_frame"] - inj_f) / FPS for a in anchors], dtype=np.float32)
        tau_bout = np.clip(bout_times / total_duration, 0.0, 1.0)
        anc_block_idx = np.clip((anchors - 1) // BLOCK_BOUTS, 0, n_blocks - 1)
        bout_starts_sec = np.array([(post_bouts[a]["start_frame"] - inj_f) / FPS for a in anchors], dtype=np.float32)
        
        animals.append({
            "session_idx": i,
            "animal_id": aid,
            "cohort": cohort,
            "genotype": gt,
            "n_blocks": n_blocks,
            "block_reps": block_reps,
            "block_times": block_times,
            "block_durations": block_durations,
            "n_bouts": M,
            "sylls": sylls,
            "anchors": anchors,
            "target_s": target_s,
            "s_curr": s_curr,
            "s_prev": s_prev,
            "tau_bout": tau_bout,
            "bout_starts_sec": bout_starts_sec,
            "anc_block_idx": anc_block_idx,
            "total_duration": total_duration,
            "post_bouts": post_bouts,
            "inj_frame": inj_f
        })
        print(f"  [{cohort}] Mouse {aid:6s} | {n_blocks} blocks ({M:,} bouts, {total_duration/60.0:.1f} min)")
        
    print(f"Loaded {len(animals)} mice: "
          f"{sum(1 for a in animals if a['cohort']=='WT_Saline')} WT, "
          f"{sum(1 for a in animals if a['cohort']=='KI_Saline')} KI.")
    return animals


# ---------------------------------------------------------------------------
# CANONICAL SHARED HMM FIT & FORWARD FILTERING (EXACT SAME COORDINATE SYSTEM)
# ---------------------------------------------------------------------------
def fit_canonical_shared_hmm(animals):
    print("\n" + "=" * 80)
    print("FITTING SHARED-EMISSION HMM (CANONICAL Q1..Q4 COORDINATE SYSTEM)")
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
    
    for a in animals:
        Z = (global_pca.transform(a["block_reps"]) - mu_Z) / sig_Z
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
            
        a["Q_filtered"] = filtered_gamma
        causal_map_blocks = np.argmax(filtered_gamma, axis=1)
        a["causal_q_blocks"] = causal_map_blocks
        
        # Bout anchors mapping
        anc_b = a["anc_block_idx"]
        a["q_curr"] = causal_map_blocks[anc_b]
        
        # Compute block entry times for state age
        block_entry_indices = np.zeros(T_blocks, dtype=int)
        cur_s = causal_map_blocks[0]
        entry_idx = 0
        for b in range(T_blocks):
            if causal_map_blocks[b] != cur_s:
                cur_s = causal_map_blocks[b]
                entry_idx = b
            block_entry_indices[b] = entry_idx
            
        N_anc = len(a["anchors"])
        age_bouts = np.zeros(N_anc, dtype=np.float32)
        age_sec = np.zeros(N_anc, dtype=np.float32)
        block_starts_sec = np.array([a["block_times"][b] * a["total_duration"] for b in range(T_blocks)], dtype=np.float32)
        
        for i in range(N_anc):
            b_curr = anc_b[i]
            b_entry = block_entry_indices[b_curr]
            entry_bout_idx = b_entry * BLOCK_BOUTS
            age_bouts[i] = max(a["anchors"][i] - entry_bout_idx, 0.0)
            age_sec[i] = max(a["bout_starts_sec"][i] - block_starts_sec[b_entry], 0.0)
            
        a["age_bouts"] = age_bouts
        a["age_sec"] = age_sec
        
    print("Shared HMM canonical coordinate system mapped successfully.")
    return global_hmm


# ---------------------------------------------------------------------------
# 1. PLAIN STATE OCCUPANCY BY GENOTYPE
# ---------------------------------------------------------------------------
def run_comparison_1_plain_occupancy(animals):
    print("\n" + "=" * 80)
    print("1. PLAIN STATE OCCUPANCY BY GENOTYPE (MOUSE-LEVEL)")
    print("=" * 80)
    
    records = []
    for a in animals:
        q_blocks = a["causal_q_blocks"]
        durations = a["block_durations"]
        counts = np.bincount(q_blocks, minlength=K_STATES).astype(float)
        occ = counts / np.sum(counts)
        
        # Physical time duration in seconds
        time_per_state = np.zeros(K_STATES, dtype=float)
        for b_idx in range(len(q_blocks)):
            st = q_blocks[b_idx]
            time_per_state[st] += durations[b_idx]
        total_time_sec = np.sum(time_per_state) + 1e-9
        time_occ = time_per_state / total_time_sec
        
        rec = {
            "animal_id": a["animal_id"],
            "cohort": a["cohort"],
            "genotype": a["genotype"],
            "total_time_min": total_time_sec / 60.0
        }
        for q in range(K_STATES):
            rec[f"P_Q{q+1}"] = occ[q]                     # Block fraction
            rec[f"time_pct_Q{q+1}"] = time_occ[q]        # Physical time percentage
            rec[f"time_sec_Q{q+1}"] = time_per_state[q]  # Physical time in seconds
            rec[f"time_min_Q{q+1}"] = time_per_state[q] / 60.0
        records.append(rec)
        
    df_mice = pd.DataFrame(records)
    
    wt_rows = df_mice[df_mice["genotype"] == "WT"]
    ki_rows = df_mice[df_mice["genotype"] == "KI"]
    
    # Omnibus PERMANOVA permutation test on Aitchison distance (on physical time occupancy)
    def comp_aitchison_dist(X):
        eps = 1e-4
        X_adj = X + eps
        X_adj = X_adj / X_adj.sum(axis=1, keepdims=True)
        clr = np.log(X_adj) - np.mean(np.log(X_adj), axis=1, keepdims=True)
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(clr, metric="euclidean"))
        
    all_X = df_mice[[f"time_pct_Q{q+1}" for q in range(K_STATES)]].values
    dist_mat = comp_aitchison_dist(all_X)
    N_total = len(df_mice)
    labels = (df_mice["genotype"] == "KI").values.astype(int)
    
    def calc_pseudo_f(D, g):
        N = len(g)
        n1 = np.sum(g == 0)
        n2 = np.sum(g == 1)
        if n1 == 0 or n2 == 0:
            return 0.0
        ss_tot = np.sum(D**2) / (2.0 * N)
        D0 = D[np.ix_(g == 0, g == 0)]
        D1 = D[np.ix_(g == 1, g == 1)]
        ss_w = (np.sum(D0**2) / (2.0 * n1)) + (np.sum(D1**2) / (2.0 * n2))
        ss_a = ss_tot - ss_w
        pseudo_f = (ss_a / 1.0) / (ss_w / (N - 2.0))
        return pseudo_f
        
    f_obs = calc_pseudo_f(dist_mat, labels)
    
    # Exact combinatorial mouse-level permutation for Omnibus PERMANOVA
    combos_all = np.array(list(itertools.combinations(range(N_total), np.sum(labels == 0))))
    perm_f = []
    for c in combos_all:
        shuff_g = np.ones(N_total, dtype=int)
        shuff_g[c] = 0
        perm_f.append(calc_pseudo_f(dist_mat, shuff_g))
    perm_f = np.array(perm_f)
    p_omnibus = (1.0 + np.sum(perm_f >= f_obs - 1e-9)) / (len(combos_all) + 1.0)
    
    print(f"Omnibus PERMANOVA Pseudo-F = {f_obs:.4f}, exact permutation p = {p_omnibus:.4f}")
    
    # State-specific comparisons (Both Block Fraction and Physical Time Duration)
    summary_rows = []
    for q in range(K_STATES):
        col_block = f"P_Q{q+1}"
        col_time = f"time_pct_Q{q+1}"
        col_sec = f"time_sec_Q{q+1}"
        
        wt_block = wt_rows[col_block].values
        ki_block = ki_rows[col_block].values
        wt_time = wt_rows[col_time].values
        ki_time = ki_rows[col_time].values
        wt_sec = wt_rows[col_sec].values
        ki_sec = ki_rows[col_sec].values
        
        m_wt_b, s_wt_b = np.mean(wt_block), stats.sem(wt_block)
        m_ki_b, s_ki_b = np.mean(ki_block), stats.sem(ki_block)
        diff_b = m_ki_b - m_wt_b
        p_val_block, _ = exact_or_single_perm(wt_block, ki_block)
        
        m_wt_t, s_wt_t = np.mean(wt_time), stats.sem(wt_time)
        m_ki_t, s_ki_t = np.mean(ki_time), stats.sem(ki_time)
        diff_t = m_ki_t - m_wt_t
        p_val_time, _ = exact_or_single_perm(wt_time, ki_time)
        
        m_wt_sec = np.mean(wt_sec)
        m_ki_sec = np.mean(ki_sec)
        
        # Bootstrap 95% CI on physical time difference
        boot_diffs = []
        for _ in range(10000):
            b_wt = np.random.choice(wt_time, size=len(wt_time), replace=True)
            b_ki = np.random.choice(ki_time, size=len(ki_time), replace=True)
            boot_diffs.append(np.mean(b_ki) - np.mean(b_wt))
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        
        # Cohen's d on physical time
        s_pooled = np.sqrt(((len(wt_time)-1)*np.var(wt_time, ddof=1) + (len(ki_time)-1)*np.var(ki_time, ddof=1)) / (len(wt_time)+len(ki_time)-2))
        d_time = diff_t / (s_pooled + 1e-12)
        
        summary_rows.append({
            "state": f"Q{q+1}",
            "wt_block_fraction": m_wt_b,
            "ki_block_fraction": m_ki_b,
            "diff_block_fraction": diff_b,
            "p_exact_block": p_val_block,
            "wt_time_fraction": m_wt_t,
            "ki_time_fraction": m_ki_t,
            "diff_time_fraction": diff_t,
            "ci_95_low_time": ci_low,
            "ci_95_high_time": ci_high,
            "cohens_d_time": d_time,
            "p_exact_time": p_val_time,
            "wt_time_sec_mean": m_wt_sec,
            "ki_time_sec_mean": m_ki_sec,
            "wt_time_min_mean": m_wt_sec / 60.0,
            "ki_time_min_mean": m_ki_sec / 60.0,
            "omnibus_permanova_p": p_omnibus
        })
        print(f"State Q{q+1}: Time Occupancy WT={m_wt_t*100:.1f}% ({m_wt_sec/60:.1f}m) vs KI={m_ki_t*100:.1f}% ({m_ki_sec/60:.1f}m) | "
              f"Δ={diff_t*100:+.1f}% (p_exact={p_val_time:.4f}, d={d_time:+.2f}) | Block Δ={diff_b*100:+.1f}% (p_exact={p_val_block:.4f})")
              
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp1_plain_occupancy.csv"), index=False)
    return df_mice, df_summary


# ---------------------------------------------------------------------------
# 2. TIME-RESOLVED OCCUPANCY
# ---------------------------------------------------------------------------
def run_comparison_2_time_resolved_occupancy(animals):
    print("\n" + "=" * 80)
    print("2. TIME-RESOLVED OCCUPANCY (SESSION DECILES & G x TIME INTERACTION)")
    print("=" * 80)
    
    n_deciles = 10
    records = []
    
    for a in animals:
        q_blocks = a["causal_q_blocks"]
        t_blocks = a["block_times"]
        aid = a["animal_id"]
        gt = a["genotype"]
        
        deciles = np.clip(np.floor(t_blocks * n_deciles).astype(int), 0, n_deciles - 1)
        
        for d in range(n_deciles):
            mask = (deciles == d)
            if np.sum(mask) > 0:
                counts = np.bincount(q_blocks[mask], minlength=K_STATES).astype(float)
                occ = counts / np.sum(counts)
            else:
                occ = np.zeros(K_STATES)
            for q in range(K_STATES):
                records.append({
                    "animal_id": aid,
                    "genotype": gt,
                    "decile": d,
                    "state": f"Q{q+1}",
                    "occupancy": occ[q]
                })
                
    df_time = pd.DataFrame(records)
    summary_rows = []
    
    for q in range(K_STATES):
        sub = df_time[df_time["state"] == f"Q{q+1}"]
        
        mouse_stats = []
        for aid, g in sub.groupby("animal_id"):
            gt_val = 1 if g["genotype"].iloc[0] == "KI" else 0
            dec_vals = g["decile"].values
            occ_vals = g["occupancy"].values
            slope, intercept, _, _, _ = stats.linregress(dec_vals, occ_vals)
            mean_occ = np.mean(occ_vals)
            mouse_stats.append({
                "animal_id": aid,
                "genotype": gt_val,
                "mean_occ": mean_occ,
                "slope": slope
            })
        df_ms = pd.DataFrame(mouse_stats)
        
        wt_m = df_ms[df_ms["genotype"] == 0]["mean_occ"].values
        ki_m = df_ms[df_ms["genotype"] == 1]["mean_occ"].values
        diff_mean = np.mean(ki_m) - np.mean(wt_m)
        
        wt_s = df_ms[df_ms["genotype"] == 0]["slope"].values
        ki_s = df_ms[df_ms["genotype"] == 1]["slope"].values
        diff_slope = np.mean(ki_s) - np.mean(wt_s)
        
        # Exact permutation tests
        perm_p_g, _ = exact_or_single_perm(wt_m, ki_m)
        perm_p_gt, _ = exact_or_single_perm(wt_s, ki_s)
        
        summary_rows.append({
            "state": f"Q{q+1}",
            "diff_mean_occ": diff_mean,
            "p_genotype_main": perm_p_g,
            "mean_slope_wt": np.mean(wt_s),
            "mean_slope_ki": np.mean(ki_s),
            "diff_slope_GT": diff_slope,
            "p_genotype_x_time": perm_p_gt
        })
        print(f"State Q{q+1}: G main p={perm_p_g:.4f}, Slope WT={np.mean(wt_s)*10:+.3f}/sess, "
              f"KI={np.mean(ki_s)*10:+.3f}/sess, GxTime p={perm_p_gt:.4f}")
              
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp2_time_resolved_occupancy.csv"), index=False)
    return df_time, df_summary


# ---------------------------------------------------------------------------
# 3. ENTRY RATE VERSUS EXIT RATE DECOMPOSITION
# ---------------------------------------------------------------------------
def run_comparison_3_entry_exit_decomposition(animals):
    print("\n" + "=" * 80)
    print("3. ENTRY RATE VERSUS EXIT RATE DECOMPOSITION")
    print("=" * 80)
    
    records = []
    for a in animals:
        q_blocks = a["causal_q_blocks"]
        n_b = len(q_blocks)
        aid = a["animal_id"]
        gt = a["genotype"]
        
        for q in range(K_STATES):
            is_in = (q_blocks == q)
            blocks_in = np.sum(is_in)
            blocks_out = n_b - blocks_in
            
            entries = 0
            if is_in[0]:
                entries += 1
            entries += np.sum((~is_in[:-1]) & is_in[1:])
            
            exits = np.sum(is_in[:-1] & (~is_in[1:]))
            
            lambda_enter = entries / float(max(blocks_out, 1))
            lambda_exit = exits / float(max(blocks_in, 1)) if blocks_in > 0 else np.nan
            dwell = float(blocks_in) / float(max(exits, 1)) if blocks_in > 0 else np.nan
            
            records.append({
                "animal_id": aid,
                "genotype": gt,
                "state": f"Q{q+1}",
                "blocks_in": blocks_in,
                "blocks_out": blocks_out,
                "entries": entries,
                "exits": exits,
                "entry_rate": lambda_enter,
                "exit_rate": lambda_exit,
                "mean_dwell_blocks": dwell
            })
            
    df_ee = pd.DataFrame(records)
    
    summary_rows = []
    for q in range(K_STATES):
        sub = df_ee[df_ee["state"] == f"Q{q+1}"]
        wt_sub = sub[sub["genotype"] == "WT"]
        ki_sub = sub[sub["genotype"] == "KI"]
        
        wt_valid_enter = wt_sub["entry_rate"].dropna().values
        ki_valid_enter = ki_sub["entry_rate"].dropna().values
        wt_valid_exit = wt_sub["exit_rate"].dropna().values
        ki_valid_exit = ki_sub["exit_rate"].dropna().values
        
        m_ent_wt, s_ent_wt = np.mean(wt_valid_enter), stats.sem(wt_valid_enter) if len(wt_valid_enter)>1 else 0.0
        m_ent_ki, s_ent_ki = np.mean(ki_valid_enter), stats.sem(ki_valid_enter) if len(ki_valid_enter)>1 else 0.0
        diff_ent = m_ent_ki - m_ent_wt
        
        p_ent, _ = exact_or_single_perm(wt_valid_enter, ki_valid_enter)
        
        support_exit = (len(wt_valid_exit) >= 4 and len(ki_valid_exit) >= 4)
        if support_exit:
            m_exit_wt, s_exit_wt = np.mean(wt_valid_exit), stats.sem(wt_valid_exit)
            m_exit_ki, s_exit_ki = np.mean(ki_valid_exit), stats.sem(ki_valid_exit)
            diff_exit = m_exit_ki - m_exit_wt
            p_exit, _ = exact_or_single_perm(wt_valid_exit, ki_valid_exit)
            
            delta_log_enter = np.log(max(m_ent_ki, 1e-6)) - np.log(max(m_ent_wt, 1e-6))
            delta_log_exit = -(np.log(max(m_exit_ki, 1e-6)) - np.log(max(m_exit_wt, 1e-6)))
        else:
            m_exit_wt = np.mean(wt_valid_exit) if len(wt_valid_exit)>0 else np.nan
            s_exit_wt = stats.sem(wt_valid_exit) if len(wt_valid_exit)>1 else 0.0
            m_exit_ki = np.mean(ki_valid_exit) if len(ki_valid_exit)>0 else np.nan
            s_exit_ki = stats.sem(ki_valid_exit) if len(ki_valid_exit)>1 else 0.0
            diff_exit = m_exit_ki - m_exit_wt if not np.isnan(m_exit_wt) and not np.isnan(m_exit_ki) else np.nan
            p_exit = np.nan
            delta_log_enter = np.nan
            delta_log_exit = np.nan
            
        summary_rows.append({
            "state": f"Q{q+1}",
            "support_adequate": support_exit,
            "n_mice_wt_exit": len(wt_valid_exit),
            "n_mice_ki_exit": len(ki_valid_exit),
            "entry_rate_wt": m_ent_wt,
            "entry_rate_ki": m_ent_ki,
            "diff_entry_rate": diff_ent,
            "p_entry_rate": p_ent,
            "exit_rate_wt": m_exit_wt,
            "exit_rate_ki": m_exit_ki,
            "diff_exit_rate": diff_exit,
            "p_exit_rate": p_exit,
            "delta_log_enter_contrib": delta_log_enter,
            "delta_log_exit_contrib": delta_log_exit
        })
        p_exit_str = f"{p_exit:.4f}" if not np.isnan(p_exit) else "N/A"
        print(f"State Q{q+1} (Support={support_exit}): Entry WT={m_ent_wt:.4f}, KI={m_ent_ki:.4f} (p={p_ent:.4f}); "
              f"Exit WT={m_exit_wt:.4f}, KI={m_exit_ki:.4f} (p={p_exit_str})")
              
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp3_entry_exit_decomposition.csv"), index=False)
    return df_ee, df_summary


# ---------------------------------------------------------------------------
# 4. ROUTING AFTER EXIT
# ---------------------------------------------------------------------------
def run_comparison_4_routing_after_exit(animals):
    print("\n" + "=" * 80)
    print("4. ROUTING AFTER EXIT (CONDITIONAL TRANSITION PROBABILITIES)")
    print("=" * 80)
    
    routing_records = []
    for a in animals:
        q_blocks = a["causal_q_blocks"]
        aid = a["animal_id"]
        gt = a["genotype"]
        
        exits_from = {q: [] for q in range(K_STATES)}
        for t in range(len(q_blocks) - 1):
            curr_q = q_blocks[t]
            next_q = q_blocks[t+1]
            if curr_q != next_q:
                exits_from[curr_q].append(next_q)
                
        for i in range(K_STATES):
            destinations = exits_from[i]
            n_ex = len(destinations)
            counts = np.bincount(destinations, minlength=K_STATES)
            for j in range(K_STATES):
                if i == j:
                    continue
                prob = counts[j] / float(n_ex) if n_ex > 0 else np.nan
                routing_records.append({
                    "animal_id": aid,
                    "genotype": gt,
                    "from_state": f"Q{i+1}",
                    "to_state": f"Q{j+1}",
                    "n_exits_mouse": n_ex,
                    "dest_count": counts[j],
                    "routing_prob": prob
                })
                
    df_route = pd.DataFrame(routing_records)
    summary_rows = []
    supported_pairs = [("Q2", "Q1"), ("Q2", "Q4"), ("Q4", "Q1"), ("Q4", "Q2")]
    
    for from_s, to_s in supported_pairs:
        sub = df_route[(df_route["from_state"] == from_s) & (df_route["to_state"] == to_s)]
        wt_vals = sub[sub["genotype"] == "WT"]["routing_prob"].dropna().values
        ki_vals = sub[sub["genotype"] == "KI"]["routing_prob"].dropna().values
        
        m_wt, s_wt = np.mean(wt_vals), stats.sem(wt_vals) if len(wt_vals)>1 else 0.0
        m_ki, s_ki = np.mean(ki_vals), stats.sem(ki_vals) if len(ki_vals)>1 else 0.0
        diff = m_ki - m_wt
        
        p_val, _ = exact_or_single_perm(wt_vals, ki_vals)
                                 
        boot_diffs = [np.mean(np.random.choice(ki_vals, size=len(ki_vals), replace=True)) - 
                      np.mean(np.random.choice(wt_vals, size=len(wt_vals), replace=True))
                      for _ in range(5000)]
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        
        summary_rows.append({
            "transition": f"{from_s}->{to_s}",
            "from_state": from_s,
            "to_state": to_s,
            "n_wt_mice": len(wt_vals),
            "n_ki_mice": len(ki_vals),
            "wt_mean": m_wt,
            "wt_sem": s_wt,
            "ki_mean": m_ki,
            "ki_sem": s_ki,
            "diff_ki_wt": diff,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "perm_p_val": p_val
        })
        print(f"Routing {from_s}->{to_s}: WT={m_wt*100:.1f}±{s_wt*100:.1f}%, KI={m_ki*100:.1f}±{s_ki*100:.1f}%, "
              f"Δ={diff*100:+.1f}% (95% CI [{ci_low*100:+.1f}%, {ci_high*100:+.1f}%]), p={p_val:.4f}")
              
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp4_routing_after_exit.csv"), index=False)
    return df_route, df_summary


# ---------------------------------------------------------------------------
# 5. WITHIN-STATE SYLLABLE COMPOSITION
# ---------------------------------------------------------------------------
def run_comparison_5_within_state_syllable_composition(animals):
    print("\n" + "=" * 80)
    print("5. WITHIN-STATE SYLLABLE COMPOSITION (P(S | Q=q, G) & JSD)")
    print("=" * 80)
    
    mouse_syll_dist = {q: {} for q in range(K_STATES)}
    
    for a in animals:
        aid = a["animal_id"]
        gt = a["genotype"]
        q_anc = a["q_curr"]
        sylls_anc = a["sylls"][a["anchors"]]
        
        for q in range(K_STATES):
            mask = (q_anc == q)
            if np.sum(mask) >= 50:
                counts = np.bincount(sylls_anc[mask], minlength=NUM_SYLLS).astype(float)
                dist = counts / np.sum(counts)
                mouse_syll_dist[q][aid] = {
                    "genotype": gt,
                    "dist": dist,
                    "n_bouts": np.sum(mask)
                }
                
    summary_rows = []
    
    for q in range(K_STATES):
        mice_q = mouse_syll_dist[q]
        wt_aids = [aid for aid, d in mice_q.items() if d["genotype"] == "WT"]
        ki_aids = [aid for aid, d in mice_q.items() if d["genotype"] == "KI"]
        
        if len(wt_aids) < 2 or len(ki_aids) < 2:
            print(f"State Q{q+1}: Inadequate dual-cohort support (WT={len(wt_aids)}, KI={len(ki_aids)})")
            summary_rows.append({
                "state": f"Q{q+1}",
                "support_adequate": False,
                "n_wt_mice": len(wt_aids),
                "n_ki_mice": len(ki_aids),
                "cohort_jsd": np.nan,
                "within_wt_jsd": np.nan,
                "within_ki_jsd": np.nan,
                "between_cohort_jsd": np.nan,
                "perm_p_val": np.nan
            })
            continue
            
        wt_pooled = np.mean([mice_q[aid]["dist"] for aid in wt_aids], axis=0)
        ki_pooled = np.mean([mice_q[aid]["dist"] for aid in ki_aids], axis=0)
        cohort_jsd = jensenshannon(wt_pooled, ki_pooled)
        
        within_wt = []
        for i in range(len(wt_aids)):
            for j in range(i+1, len(wt_aids)):
                within_wt.append(jensenshannon(mice_q[wt_aids[i]]["dist"], mice_q[wt_aids[j]]["dist"]))
                
        within_ki = []
        for i in range(len(ki_aids)):
            for j in range(i+1, len(ki_aids)):
                within_ki.append(jensenshannon(mice_q[ki_aids[i]]["dist"], mice_q[ki_aids[j]]["dist"]))
                
        between = []
        for w in wt_aids:
            for k in ki_aids:
                between.append(jensenshannon(mice_q[w]["dist"], mice_q[k]["dist"]))
                
        m_within_wt = np.mean(within_wt) if len(within_wt)>0 else np.nan
        m_within_ki = np.mean(within_ki) if len(within_ki)>0 else np.nan
        m_between = np.mean(between)
        
        all_aids = wt_aids + ki_aids
        n_wt = len(wt_aids)
        perm_jsds = []
        for _ in range(5000):
            shuff = np.random.permutation(all_aids)
            p_wt = np.mean([mice_q[aid]["dist"] for aid in shuff[:n_wt]], axis=0)
            p_ki = np.mean([mice_q[aid]["dist"] for aid in shuff[n_wt:]], axis=0)
            perm_jsds.append(jensenshannon(p_wt, p_ki))
        p_val = np.mean(np.array(perm_jsds) >= cohort_jsd)
        
        summary_rows.append({
            "state": f"Q{q+1}",
            "support_adequate": True,
            "n_wt_mice": len(wt_aids),
            "n_ki_mice": len(ki_aids),
            "cohort_jsd": cohort_jsd,
            "within_wt_jsd": m_within_wt,
            "within_ki_jsd": m_within_ki,
            "between_cohort_jsd": m_between,
            "perm_p_val": p_val
        })
        print(f"State Q{q+1}: Cohort JSD={cohort_jsd:.4f} (Within WT={m_within_wt:.4f}, KI={m_within_ki:.4f}, Between={m_between:.4f}), Perm p={p_val:.4f}")
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp5_within_state_syllables.csv"), index=False)
    return df_summary


# ---------------------------------------------------------------------------
# 6. WITHIN-STATE LOCAL TRANSITIONS
# ---------------------------------------------------------------------------
def run_comparison_6_within_state_local_transitions(animals):
    print("\n" + "=" * 80)
    print("6. WITHIN-STATE LOCAL TRANSITION CROSS-TRANSFER (STATE-BY-STATE)")
    print("=" * 80)
    
    records = []
    for test_idx, test_a in enumerate(animals):
        test_aid = test_a["animal_id"]
        test_gt = test_a["genotype"]
        train_animals = [a for i, a in enumerate(animals) if i != test_idx]
        
        test_q = test_a["q_curr"]
        test_s = test_a["sylls"][test_a["anchors"]]
        test_target = test_a["target_s"]
        
        for q in range(K_STATES):
            test_mask = (test_q == q)
            if np.sum(test_mask) < 20:
                continue
                
            test_curr = test_s[test_mask]
            test_targ = test_target[test_mask]
            
            train_wt_curr, train_wt_targ = [], []
            train_ki_curr, train_ki_targ = [], []
            
            for tr_a in train_animals:
                tr_mask = (tr_a["q_curr"] == q)
                if np.sum(tr_mask) == 0:
                    continue
                tr_curr = tr_a["sylls"][tr_a["anchors"]][tr_mask]
                tr_targ = tr_a["target_s"][tr_mask]
                if tr_a["genotype"] == "WT":
                    train_wt_curr.extend(tr_curr)
                    train_wt_targ.extend(tr_targ)
                else:
                    train_ki_curr.extend(tr_curr)
                    train_ki_targ.extend(tr_targ)
                    
            if len(train_wt_curr) < 50 or len(train_ki_curr) < 50:
                continue
                
            n_match = min(len(train_wt_curr), len(train_ki_curr))
            idx_wt = np.random.choice(len(train_wt_curr), size=n_match, replace=False)
            idx_ki = np.random.choice(len(train_ki_curr), size=n_match, replace=False)
            
            alpha = 0.1
            def fit_transmat(c_list, t_list):
                mat = np.zeros((NUM_SYLLS, NUM_SYLLS), dtype=np.float64) + alpha
                np.add.at(mat, (c_list, t_list), 1.0)
                mat /= mat.sum(axis=1, keepdims=True)
                return mat
                
            mat_wt = fit_transmat(np.array(train_wt_curr)[idx_wt], np.array(train_wt_targ)[idx_wt])
            mat_ki = fit_transmat(np.array(train_ki_curr)[idx_ki], np.array(train_ki_targ)[idx_ki])
            mat_pooled = 0.5 * (mat_wt + mat_ki)
            
            probs_pool = mat_pooled[test_curr, test_targ]
            nll_pool = -np.mean(np.log2(np.maximum(probs_pool, 1e-12)))
            
            mat_matched = mat_ki if test_gt == "KI" else mat_wt
            probs_matched = mat_matched[test_curr, test_targ]
            nll_matched = -np.mean(np.log2(np.maximum(probs_matched, 1e-12)))
            
            mat_cross = mat_wt if test_gt == "KI" else mat_ki
            probs_cross = mat_cross[test_curr, test_targ]
            nll_cross = -np.mean(np.log2(np.maximum(probs_cross, 1e-12)))
            
            records.append({
                "animal_id": test_aid,
                "genotype": test_gt,
                "state": f"Q{q+1}",
                "n_test_bouts": len(test_curr),
                "nll_pooled": nll_pool,
                "nll_matched": nll_matched,
                "nll_cross": nll_cross,
                "gain_matched": nll_pool - nll_matched,
                "cross_penalty": nll_cross - nll_matched
            })
            
    df_trans = pd.DataFrame(records)
    summary_rows = []
    
    for q in range(K_STATES):
        sub = df_trans[df_trans["state"] == f"Q{q+1}"]
        if len(sub) < 4:
            continue
        gains = sub["gain_matched"].values
        penalties = sub["cross_penalty"].values
        
        m_gain, s_gain = np.mean(gains), stats.sem(gains)
        t_stat, p_gain = stats.ttest_1samp(gains, 0.0)
        
        m_pen, s_pen = np.mean(penalties), stats.sem(penalties)
        _, p_pen = stats.ttest_1samp(penalties, 0.0)
        
        ci_gain = [np.percentile([np.mean(np.random.choice(gains, size=len(gains), replace=True)) for _ in range(5000)], [2.5, 97.5])]
        
        summary_rows.append({
            "state": f"Q{q+1}",
            "n_mice": len(sub),
            "mean_gain_matched_bits": m_gain,
            "sem_gain_matched": s_gain,
            "ci_95_low": ci_gain[0][0],
            "ci_95_high": ci_gain[0][1],
            "p_gain": p_gain,
            "mean_cross_penalty_bits": m_pen,
            "sem_cross_penalty": s_pen,
            "p_penalty": p_pen
        })
        print(f"State Q{q+1}: Matched Gain = {m_gain:+.4f}±{s_gain:.4f} b (p={p_gain:.4f}), "
              f"Cross Penalty = {m_pen:+.4f}±{s_pen:.4f} b (p={p_pen:.4f})")
              
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp6_within_state_local_transitions.csv"), index=False)
    return df_trans, df_summary


# ---------------------------------------------------------------------------
# 7. STATE-AGE TRAJECTORIES BY GENOTYPE
# ---------------------------------------------------------------------------
def run_comparison_7_state_age_trajectories(animals):
    print("\n" + "=" * 80)
    print("7. STATE-AGE TRAJECTORIES BY GENOTYPE (G x A INTERACTION)")
    print("=" * 80)
    
    records = []
    for test_idx, test_a in enumerate(animals):
        test_aid = test_a["animal_id"]
        test_gt = test_a["genotype"]
        train_animals = [a for i, a in enumerate(animals) if i != test_idx]
        
        test_s = test_a["sylls"][test_a["anchors"]]
        test_targ = test_a["target_s"]
        test_q = test_a["q_curr"]
        test_age = test_a["age_bouts"]
        
        for q in range(K_STATES):
            test_mask = (test_q == q)
            if np.sum(test_mask) < 20:
                continue
                
            q_test_s = test_s[test_mask]
            q_test_targ = test_targ[test_mask]
            q_test_age = np.clip(test_age[test_mask] / 50.0, 0.0, 5.0)
            
            train_s_list = []
            train_targ_list = []
            train_age_list = []
            train_gt_list = []
            
            for tr_a in train_animals:
                tr_mask = (tr_a["q_curr"] == q)
                if np.sum(tr_mask) == 0:
                    continue
                tr_s = tr_a["sylls"][tr_a["anchors"]][tr_mask]
                tr_targ = tr_a["target_s"][tr_mask]
                tr_age = np.clip(tr_a["age_bouts"][tr_mask] / 50.0, 0.0, 5.0)
                tr_gt = 1.0 if tr_a["genotype"] == "KI" else 0.0
                
                train_s_list.extend(tr_s)
                train_targ_list.extend(tr_targ)
                train_age_list.extend(tr_age)
                train_gt_list.extend([tr_gt] * len(tr_s))
                
            if len(train_s_list) < 100:
                continue
                
            N_tr = len(train_s_list)
            tr_s_arr = np.array(train_s_list, dtype=np.int64)
            tr_targ_arr = np.array(train_targ_list, dtype=np.int64)
            tr_age_arr = np.array(train_age_list, dtype=np.float32)[:, None]
            tr_gt_arr = np.array(train_gt_list, dtype=np.float32)[:, None]
            
            te_s_arr = np.array(q_test_s, dtype=np.int64)
            te_targ_arr = np.array(q_test_targ, dtype=np.int64)
            te_age_arr = np.array(q_test_age, dtype=np.float32)[:, None]
            te_gt_val = 1.0 if test_gt == "KI" else 0.0
            te_gt_arr = np.full((len(te_s_arr), 1), te_gt_val, dtype=np.float32)
            
            class AgeModel(nn.Module):
                def __init__(self, use_genotype=False):
                    super().__init__()
                    self.use_genotype = use_genotype
                    self.emb = nn.Embedding(NUM_SYLLS, 32)
                    in_dim = 32 + 1
                    if use_genotype:
                        in_dim += 2
                    self.fc = nn.Linear(in_dim, NUM_SYLLS)
                    
                def forward(self, s, age, gt):
                    feat = [self.emb(s), age]
                    if self.use_genotype:
                        feat.extend([gt, gt * age])
                    x = torch.cat(feat, dim=1)
                    return self.fc(x)
                    
            m_base = AgeModel(use_genotype=False).to(DEVICE)
            m_gt = AgeModel(use_genotype=True).to(DEVICE)
            
            crit = nn.CrossEntropyLoss()
            opt_base = optim.AdamW(m_base.parameters(), lr=0.01, weight_decay=1e-3)
            opt_gt = optim.AdamW(m_gt.parameters(), lr=0.01, weight_decay=1e-3)
            
            idx_all = np.arange(N_tr)
            np.random.shuffle(idx_all)
            bs = 512
            
            for _ in range(5):
                for start in range(0, min(N_tr, 20000), bs):
                    b_idx = idx_all[start:start+bs]
                    s_b = torch.tensor(tr_s_arr[b_idx], device=DEVICE)
                    t_b = torch.tensor(tr_targ_arr[b_idx], device=DEVICE)
                    a_b = torch.tensor(tr_age_arr[b_idx], device=DEVICE)
                    g_b = torch.tensor(tr_gt_arr[b_idx], device=DEVICE)
                    
                    opt_base.zero_grad()
                    loss_b = crit(m_base(s_b, a_b, g_b), t_b)
                    loss_b.backward()
                    opt_base.step()
                    
                    opt_gt.zero_grad()
                    loss_g = crit(m_gt(s_b, a_b, g_b), t_b)
                    loss_g.backward()
                    opt_gt.step()
                    
            m_base.eval()
            m_gt.eval()
            with torch.no_grad():
                te_s_t = torch.tensor(te_s_arr, device=DEVICE)
                te_t_t = torch.tensor(te_targ_arr, device=DEVICE)
                te_a_t = torch.tensor(te_age_arr, device=DEVICE)
                te_g_t = torch.tensor(te_gt_arr, device=DEVICE)
                
                nll_base = crit(m_base(te_s_t, te_a_t, te_g_t), te_t_t).item() / np.log(2.0)
                nll_gt = crit(m_gt(te_s_t, te_a_t, te_g_t), te_t_t).item() / np.log(2.0)
                
            records.append({
                "animal_id": test_aid,
                "genotype": test_gt,
                "state": f"Q{q+1}",
                "nll_base": nll_base,
                "nll_with_gt_age": nll_gt,
                "gain_gt_age": nll_base - nll_gt
            })
            
    df_age = pd.DataFrame(records)
    summary_rows = []
    
    for q in range(K_STATES):
        sub = df_age[df_age["state"] == f"Q{q+1}"]
        if len(sub) < 4:
            continue
        gains = sub["gain_gt_age"].values
        m_gain, s_gain = np.mean(gains), stats.sem(gains)
        t_stat, p_val = stats.ttest_1samp(gains, 0.0)
        
        summary_rows.append({
            "state": f"Q{q+1}",
            "n_mice": len(sub),
            "mean_gain_gt_age_bits": m_gain,
            "sem_gain": s_gain,
            "p_val": p_val
        })
        print(f"State Q{q+1}: G x Age interaction gain = {m_gain:+.4f}±{s_gain:.4f} b (p={p_val:.4f})")
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(TAB_DIR, "table_comp7_state_age_trajectories.csv"), index=False)
    return df_age, df_summary


# ---------------------------------------------------------------------------
# 8. CAN STATE USAGE IDENTIFY GENOTYPE? (MOUSE-LEVEL DECODER)
# ---------------------------------------------------------------------------
def run_comparison_8_genotype_decoder(animals, df_mice, df_ee, df_route):
    print("\n" + "=" * 80)
    print("8. CAN STATE USAGE IDENTIFY GENOTYPE? (LOMO FEATURE ABLATION DECODER)")
    print("=" * 80)
    
    mouse_ids = [a["animal_id"] for a in animals]
    labels = np.array([1 if a["genotype"] == "KI" else 0 for a in animals])
    N_mice = len(animals)
    
    # 1. Feature Set: Occupancy Only (4 features)
    feat_occ = []
    for aid in mouse_ids:
        row = df_mice[df_mice["animal_id"] == aid].iloc[0]
        feat_occ.append([row[f"P_Q{q+1}"] for q in range(K_STATES)])
    feat_occ = np.array(feat_occ, dtype=np.float32)
    
    # 2. Feature Set: Occupancy + Entry/Exit Rates (12 features)
    feat_ee = []
    for aid in mouse_ids:
        row_occ = df_mice[df_mice["animal_id"] == aid].iloc[0]
        sub_ee = df_ee[df_ee["animal_id"] == aid]
        f_vec = [row_occ[f"P_Q{q+1}"] for q in range(K_STATES)]
        for q in range(K_STATES):
            q_row = sub_ee[sub_ee["state"] == f"Q{q+1}"]
            if len(q_row) > 0:
                f_vec.append(q_row["entry_rate"].iloc[0])
                ex = q_row["exit_rate"].iloc[0]
                f_vec.append(ex if not np.isnan(ex) else 0.0)
            else:
                f_vec.extend([0.0, 0.0])
        feat_ee.append(f_vec)
    feat_ee = np.array(feat_ee, dtype=np.float32)
    
    # 3. Feature Set: Routing Proportions (Supported exits: Q2 and Q4, 4 features)
    feat_route = []
    for aid in mouse_ids:
        sub_r = df_route[df_route["animal_id"] == aid]
        f_vec = []
        for from_s, to_s in [("Q2", "Q1"), ("Q2", "Q4"), ("Q4", "Q1"), ("Q4", "Q2")]:
            r_row = sub_r[(sub_r["from_state"] == from_s) & (sub_r["to_state"] == to_s)]
            if len(r_row) > 0 and not np.isnan(r_row["routing_prob"].iloc[0]):
                f_vec.append(r_row["routing_prob"].iloc[0])
            else:
                f_vec.append(0.0)
        feat_route.append(f_vec)
    feat_route = np.array(feat_route, dtype=np.float32)
    
    # 4. Feature Set: Local Syntax Summary (Transition entropy per state, 4 features)
    feat_syntax = []
    for a in animals:
        q_anc = a["q_curr"]
        s_anc = a["sylls"][a["anchors"]]
        targ = a["target_s"]
        f_vec = []
        for q in range(K_STATES):
            mask = (q_anc == q)
            if np.sum(mask) >= 20:
                counts = np.bincount(targ[mask], minlength=NUM_SYLLS).astype(float)
                p = counts / np.sum(counts)
                p = p[p > 0]
                ent = -np.sum(p * np.log2(p))
                f_vec.append(ent)
            else:
                f_vec.append(0.0)
        feat_syntax.append(f_vec)
    feat_syntax = np.array(feat_syntax, dtype=np.float32)
    
    # 5. Combined Feature Set
    feat_combined = np.hstack([feat_occ, feat_ee, feat_route, feat_syntax])
    
    feature_dict = {
        "1. Occupancy Only (P(Q))": feat_occ,
        "2. Occupancy + Entry/Exit": feat_ee,
        "3. Routing Proportions": feat_route,
        "4. Local Syntax Summary": feat_syntax,
        "5. Combined Hierarchy": feat_combined
    }
    
    decoder_results = []
    
    for feat_name, X in feature_dict.items():
        y_true = labels
        y_pred = np.zeros(N_mice, dtype=int)
        y_prob = np.zeros(N_mice, dtype=float)
        
        for i in range(N_mice):
            train_idx = [j for j in range(N_mice) if j != i]
            X_tr, y_tr = X[train_idx], labels[train_idx]
            X_te, y_te = X[[i]], labels[[i]]
            
            mu = np.mean(X_tr, axis=0, keepdims=True)
            sig = np.std(X_tr, axis=0, keepdims=True) + 1e-6
            
            X_tr_sc = (X_tr - mu) / sig
            X_te_sc = (X_te - mu) / sig
            
            clf = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=200)
            clf.fit(X_tr_sc, y_tr)
            
            prob = clf.predict_proba(X_te_sc)[0, 1]
            pred = int(prob >= 0.5)
            
            y_pred[i] = pred
            y_prob[i] = prob
            
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        loss = log_loss(y_true, y_prob)
        
        n_perm = 1000
        perm_accs = []
        for _ in range(n_perm):
            shuff_y = np.random.permutation(labels)
            shuff_pred = np.zeros(N_mice, dtype=int)
            for i in range(N_mice):
                train_idx = [j for j in range(N_mice) if j != i]
                X_tr, y_tr = X[train_idx], shuff_y[train_idx]
                X_te = X[[i]]
                mu = np.mean(X_tr, axis=0, keepdims=True)
                sig = np.std(X_tr, axis=0, keepdims=True) + 1e-6
                clf = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=200)
                clf.fit((X_tr - mu)/sig, y_tr)
                shuff_pred[i] = int(clf.predict_proba((X_te - mu)/sig)[0, 1] >= 0.5)
            perm_accs.append(accuracy_score(shuff_y, shuff_pred))
            
        p_val = np.mean(np.array(perm_accs) >= acc)
        
        decoder_results.append({
            "feature_set": feat_name,
            "n_features": X.shape[1],
            "loao_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": auc,
            "log_loss": loss,
            "perm_p_val": p_val
        })
        print(f"{feat_name:30s} | N={X.shape[1]:2d} feats | Acc={acc*100:5.1f}% | "
              f"BalAcc={bal_acc*100:5.1f}% | AUC={auc:5.3f} | Perm p={p_val:.4f}")
              
    df_decoder = pd.DataFrame(decoder_results)
    df_decoder.to_csv(os.path.join(TAB_DIR, "table_comp8_genotype_decoder.csv"), index=False)
    return df_decoder


# ---------------------------------------------------------------------------
# PUBLICATION FIGURE GENERATION
# ---------------------------------------------------------------------------
def plot_simple_genotype_comparisons(df_mice, df_occ, df_time, df_ee, df_route, df_decoder):
    print("\nGenerating publication figure: fig_simple_genotype_comparisons.png...")
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Physical Time Occupancy Boxplot + Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    wt_data = [df_mice[df_mice["genotype"] == "WT"][f"time_pct_Q{q+1}"].values * 100 for q in range(K_STATES)]
    ki_data = [df_mice[df_mice["genotype"] == "KI"][f"time_pct_Q{q+1}"].values * 100 for q in range(K_STATES)]
    
    pos_wt = np.arange(K_STATES) * 2.0 - 0.35
    pos_ki = np.arange(K_STATES) * 2.0 + 0.35
    
    bp_wt = ax1.boxplot(wt_data, positions=pos_wt, widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=COLORS["WT"], alpha=0.3, color=COLORS["WT"]),
                        medianprops=dict(color=COLORS["WT"], linewidth=2))
    bp_ki = ax1.boxplot(ki_data, positions=pos_ki, widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=COLORS["KI"], alpha=0.3, color=COLORS["KI"]),
                        medianprops=dict(color=COLORS["KI"], linewidth=2))
                        
    for q in range(K_STATES):
        jitter_wt = np.random.normal(0, 0.05, size=len(wt_data[q]))
        jitter_ki = np.random.normal(0, 0.05, size=len(ki_data[q]))
        ax1.scatter(pos_wt[q] + jitter_wt, wt_data[q], color=COLORS["WT"], edgecolors="k", s=40, zorder=3)
        ax1.scatter(pos_ki[q] + jitter_ki, ki_data[q], color=COLORS["KI"], edgecolors="k", s=40, zorder=3)
        
    ax1.set_xticks(np.arange(K_STATES) * 2.0)
    ax1.set_xticklabels(Q_NAMES, fontweight="bold", fontsize=11)
    ax1.set_ylabel("Physical Time Occupancy (%)", fontsize=11)
    
    perm_p_omnibus = df_occ["omnibus_permanova_p"].iloc[0] if "omnibus_permanova_p" in df_occ else np.nan
    q4_p = df_occ[df_occ["state"] == "Q4"]["p_exact_time"].iloc[0] if "p_exact_time" in df_occ else np.nan
    ax1.set_title(f"1. Physical Time Occupancy by Genotype\n(PERMANOVA p = {perm_p_omnibus:.4f}, Q4 p = {q4_p:.4f})", fontsize=11, fontweight="bold")
    ax1.legend([bp_wt["boxes"][0], bp_ki["boxes"][0]], ["WT Saline (N=8)", "KI Saline (N=7)"], loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # 2. Time-Resolved Occupancy (Decile Curves for Q1 and Q4)
    ax2 = fig.add_subplot(gs[0, 1])
    deciles = np.arange(10) + 1
    for q, color, ls in [(0, "#e41a1c", "-"), (3, "#984ea3", "--")]:
        sub_q = df_time[df_time["state"] == f"Q{q+1}"]
        wt_means = [sub_q[(sub_q["genotype"] == "WT") & (sub_q["decile"] == d)]["occupancy"].mean() * 100 for d in range(10)]
        ki_means = [sub_q[(sub_q["genotype"] == "KI") & (sub_q["decile"] == d)]["occupancy"].mean() * 100 for d in range(10)]
        ax2.plot(deciles, wt_means, color=COLORS["WT"], linestyle=ls, linewidth=2, label=f"WT {Q_NAMES[q]}")
        ax2.plot(deciles, ki_means, color=COLORS["KI"], linestyle=ls, linewidth=2, label=f"KI {Q_NAMES[q]}")
        
    ax2.set_xlabel("Session Decile (Time)", fontsize=11)
    ax2.set_ylabel("Occupancy (%)", fontsize=11)
    ax2.set_title("2. Time-Resolved Occupancy\n(Persistent Genotype Divergence)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # 3. Entry vs Exit Rate Decomposition (Q4 Focus)
    ax3 = fig.add_subplot(gs[0, 2])
    sub_q4 = df_ee[df_ee["state"] == "Q4"]
    wt_q4 = sub_q4[sub_q4["genotype"] == "WT"]
    ki_q4 = sub_q4[sub_q4["genotype"] == "KI"]
    
    ent_wt = wt_q4["entry_rate"].mean() * 100
    ent_ki = ki_q4["entry_rate"].mean() * 100
    exit_wt = wt_q4["exit_rate"].dropna().mean() * 100
    exit_ki = ki_q4["exit_rate"].dropna().mean() * 100
    
    df_ee_res = pd.read_csv(os.path.join(TAB_DIR, "table_comp3_entry_exit_decomposition.csv"))
    q4_p_exit = df_ee_res[df_ee_res["state"] == "Q4"]["p_exit_rate"].iloc[0] if len(df_ee_res)>0 else np.nan
    p_exit_title = f"p = {q4_p_exit:.4f}" if not np.isnan(q4_p_exit) else "Insufficient Support"
    
    x_pos = np.array([0, 1])
    ax3.bar(x_pos - 0.2, [ent_wt, exit_wt], width=0.35, color=COLORS["WT"], alpha=0.7, label="WT")
    ax3.bar(x_pos + 0.2, [ent_ki, exit_ki], width=0.35, color=COLORS["KI"], alpha=0.7, label="KI")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(["Entry Rate (%/blk)", "Exit Hazard (%/blk)"], fontweight="bold", fontsize=11)
    ax3.set_ylabel("Rate per Block (%)", fontsize=11)
    ax3.set_title(f"3. Q4 Occupancy Decomposition\n(Exit Hazard: {p_exit_title})", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.3)
    
    # 4. Routing Conditional on Exit from Q2
    ax4 = fig.add_subplot(gs[1, 0])
    sub_r_q2 = df_route[df_route["from_state"] == "Q2"]
    r_wt_q1 = sub_r_q2[(sub_r_q2["to_state"] == "Q1") & (sub_r_q2["genotype"] == "WT")]["routing_prob"].mean() * 100
    r_ki_q1 = sub_r_q2[(sub_r_q2["to_state"] == "Q1") & (sub_r_q2["genotype"] == "KI")]["routing_prob"].mean() * 100
    r_wt_q4 = sub_r_q2[(sub_r_q2["to_state"] == "Q4") & (sub_r_q2["genotype"] == "WT")]["routing_prob"].mean() * 100
    r_ki_q4 = sub_r_q2[(sub_r_q2["to_state"] == "Q4") & (sub_r_q2["genotype"] == "KI")]["routing_prob"].mean() * 100
    
    x_r = np.array([0, 1])
    ax4.bar(x_r - 0.2, [r_wt_q1, r_wt_q4], width=0.35, color=COLORS["WT"], alpha=0.7, label="WT Exits")
    ax4.bar(x_r + 0.2, [r_ki_q1, r_ki_q4], width=0.35, color=COLORS["KI"], alpha=0.7, label="KI Exits")
    ax4.set_xticks(x_r)
    ax4.set_xticklabels(["Route to Q1", "Route to Q4"], fontweight="bold", fontsize=11)
    ax4.set_ylabel("Routing Probability (%)", fontsize=11)
    ax4.set_title(f"4. Routing After Exiting Q2\n(WT->Q4: {r_wt_q4:.1f}% vs KI->Q1: {r_ki_q1:.1f}%)", fontsize=11, fontweight="bold")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.3)
    
    # 5. Within-State Syllable Composition (JSD)
    ax5 = fig.add_subplot(gs[1, 1])
    df_jsd = pd.read_csv(os.path.join(TAB_DIR, "table_comp5_within_state_syllables.csv"))
    valid_jsd = df_jsd[df_jsd["support_adequate"] == True]
    
    x_jsd = np.arange(len(valid_jsd))
    ax5.bar(x_jsd - 0.2, valid_jsd["within_wt_jsd"], width=0.2, color=COLORS["WT"], label="Within WT")
    ax5.bar(x_jsd, valid_jsd["within_ki_jsd"], width=0.2, color=COLORS["KI"], label="Within KI")
    ax5.bar(x_jsd + 0.2, valid_jsd["between_cohort_jsd"], width=0.2, color="#e6550d", label="Between Cohorts")
    ax5.set_xticks(x_jsd)
    ax5.set_xticklabels(valid_jsd["state"], fontweight="bold", fontsize=11)
    ax5.set_ylabel("Jensen-Shannon Divergence", fontsize=11)
    ax5.set_title("5. Within-State Syllable Profiles\n(Within vs Between Divergence)", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.3)
    
    # 6. Within-State Local Transitions (Matched Gain vs Cross Penalty)
    ax6 = fig.add_subplot(gs[1, 2])
    df_trans = pd.read_csv(os.path.join(TAB_DIR, "table_comp6_within_state_local_transitions.csv"))
    x_tr = np.arange(len(df_trans))
    ax6.bar(x_tr - 0.2, df_trans["mean_gain_matched_bits"], yerr=df_trans["sem_gain_matched"],
            width=0.35, color="#31a354", capsize=5, label="Matched Gain (bits)")
    ax6.bar(x_tr + 0.2, df_trans["mean_cross_penalty_bits"], yerr=df_trans["sem_cross_penalty"],
            width=0.35, color="#de2d26", capsize=5, label="Cross Penalty (bits)")
    ax6.set_xticks(x_tr)
    ax6.set_xticklabels(df_trans["state"], fontweight="bold", fontsize=11)
    ax6.set_ylabel("NLL Difference (bits/bout)", fontsize=11)
    ax6.set_title("6. Within-State Syntax Cross-Transfer\n(Targeted Policy Differences)", fontsize=11, fontweight="bold")
    ax6.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax6.legend(fontsize=9)
    ax6.grid(True, linestyle="--", alpha=0.3)
    
    # 7. State-Age Trajectory Gain (G x Age Interaction)
    ax7 = fig.add_subplot(gs[2, 0])
    df_age = pd.read_csv(os.path.join(TAB_DIR, "table_comp7_state_age_trajectories.csv"))
    x_age = np.arange(len(df_age))
    ax7.bar(x_age, df_age["mean_gain_gt_age_bits"], yerr=df_age["sem_gain"],
            width=0.4, color="#756bb1", capsize=5)
    ax7.set_xticks(x_age)
    ax7.set_xticklabels(df_age["state"], fontweight="bold", fontsize=11)
    ax7.set_ylabel("Incremental Gain (bits/bout)", fontsize=11)
    ax7.set_title("7. State-Age Trajectory Gain (G x Age)\n(Seq Evolution with Dwell)", fontsize=11, fontweight="bold")
    ax7.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax7.grid(True, linestyle="--", alpha=0.3)
    
    # 8. Genotype Decoder Feature Ablation
    ax8 = fig.add_subplot(gs[2, 1:])
    y_pos = np.arange(len(df_decoder))
    short_names = [f.split(" (")[0] for f in df_decoder["feature_set"]]
    accs = df_decoder["loao_accuracy"] * 100
    
    bars = ax8.barh(y_pos, accs, color="#2b8cbe", alpha=0.8, height=0.55)
    ax8.axvline(53.3, color="red", linestyle="--", linewidth=1.5, label="Chance Level (8/15 = 53.3%)")
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(short_names, fontweight="bold", fontsize=11)
    ax8.set_xlabel("Leave-One-Mouse-Out (LOMO) Accuracy (%)", fontsize=11)
    
    occ_row = df_decoder[df_decoder["feature_set"].str.contains("Occupancy")]
    acc_occ = occ_row["loao_accuracy"].iloc[0] * 100 if len(occ_row) > 0 else 0
    p_occ = occ_row["perm_p_val"].iloc[0] if len(occ_row) > 0 else 1.0
    ax8.set_title(f"8. Where Genotype Information Lives (Mouse-Level Decoder)\n(Occupancy achieves {acc_occ:.1f}% LOAO Accuracy, p = {p_occ:.3f})", fontsize=11, fontweight="bold")
    ax8.set_xlim(0, 105)
    
    for idx, (bar, p_v, auc_v) in enumerate(zip(bars, df_decoder["perm_p_val"], df_decoder["roc_auc"])):
        ax8.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2.0,
                 f"{accs.iloc[idx]:.1f}% (AUC={auc_v:.2f}, p={p_v:.3f})",
                 va="center", fontsize=10, fontweight="bold")
                 
    ax8.legend(loc="lower right")
    ax8.grid(True, linestyle="--", alpha=0.3)
    
    out_fig_path = os.path.join(OUT_DIR, "fig_simple_genotype_comparisons.png")
    fig.savefig(out_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    art_path = os.path.join(ARTIFACT_DIR, "fig_simple_genotype_comparisons.png")
    shutil.copyfile(out_fig_path, art_path)
    print(f"Saved figure to: {out_fig_path} and {art_path}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 80)
    print("STARTING SIMPLE GENOTYPE COMPARISON SUITE (8 ANALYSES)")
    print("=" * 80)
    
    animals = load_and_prepare_cohort_data()
    hmm = fit_canonical_shared_hmm(animals)
    
    df_mice, df_occ = run_comparison_1_plain_occupancy(animals)
    df_time, df_time_summ = run_comparison_2_time_resolved_occupancy(animals)
    df_ee, df_ee_summ = run_comparison_3_entry_exit_decomposition(animals)
    df_route, df_route_summ = run_comparison_4_routing_after_exit(animals)
    df_jsd_summ = run_comparison_5_within_state_syllable_composition(animals)
    df_trans, df_trans_summ = run_comparison_6_within_state_local_transitions(animals)
    df_age, df_age_summ = run_comparison_7_state_age_trajectories(animals)
    df_decoder = run_comparison_8_genotype_decoder(animals, df_mice, df_ee, df_route)
    
    plot_simple_genotype_comparisons(df_mice, df_occ, df_time, df_ee, df_route, df_decoder)
    
    elapsed = time.time() - start_time
    print(f"\nSimple Genotype Comparison Suite completed in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
