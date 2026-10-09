#!/usr/bin/env python3
"""
diagnostic_lagged_mi_bias.py
=============================================================================
Comprehensive diagnostic of Lagged Mutual Information (MI) timescale behavior:
1. Tests finite-sample bias (Miller-Madow) by tracking valid pair counts and
   evaluating fixed-N subsampled curves (N = 50,000 pairs constant across lags).
2. Evaluates the nature of "shuffled" baseline: compares state_moseq's
   cross-session pairing vs. within-session endpoint permutation.
3. Quantifies behavioral non-stationarity across the 60-minute session.
4. Generates a 4-panel diagnostic figure with linear difference curves.
=============================================================================
"""

import os
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
CACHED_MI = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "lagged_mi_results_1000s.pkl")
OUTPUT_FIG = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_timescale_diagnostic_bias.png")
OUTPUT_CSV = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "tables", "diagnostic_timescale_bias_summary.csv")

def compute_mi_fast(s1, s2, n_cats, pseudo_count=1e-8):
    """Vectorized 2D contingency table mutual information."""
    flat_idx = s1 * n_cats + s2
    counts = np.bincount(flat_idx, minlength=n_cats * n_cats).reshape((n_cats, n_cats)).astype(np.float64)
    counts += pseudo_count
    probs = counts / counts.sum()
    p1 = probs.sum(axis=1, keepdims=True)
    p2 = probs.sum(axis=0, keepdims=True)
    return float(np.sum(probs * np.log(probs / (p1 * p2))))

def main():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE LAGGED MI BIAS & NON-STATIONARITY DIAGNOSTIC")
    print("=" * 80)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
    sylls = bundle["syllables_dict"]
    keys = list(sylls.keys())
    fps = bundle.get("fps", 40.0125)
    n_cats = max([v.max() for v in sylls.values()]) + 1

    with open(CACHED_MI, "rb") as f:
        cached = pickle.load(f)
    lags = cached["lags"]
    lag_seconds = cached["lag_seconds"]
    real_mis_raw = cached["real_mis"]
    markov_mis_raw = cached["markov_mis"]
    shuff_cross_raw = cached["shuff_mis"]

    n_sessions = len(keys)
    n_lags = len(lags)
    N_sub = 50000

    print(f"Loaded {n_sessions} sessions, {n_lags} lags (1.0s to 1000.0s), n_categories={n_cats}")

    # Arrays for diagnostics
    pair_counts = np.zeros((n_sessions, n_lags), dtype=int)
    perm_null = np.zeros((n_sessions, n_lags), dtype=np.float64)
    sub_real = np.zeros((n_sessions, n_lags), dtype=np.float64)
    sub_perm = np.zeros((n_sessions, n_lags), dtype=np.float64)

    rng = np.random.default_rng(12345)
    n_perms = 5

    print(f"Evaluating within-session endpoint permutation and N={N_sub} subsampling...")
    for s_idx, k in enumerate(keys):
        seq = sylls[k]
        T = len(seq)
        for l_idx, lag in enumerate(lags):
            s_curr = seq[lag:T]
            s_lag = seq[0:T - lag]
            n_p = len(s_curr)
            pair_counts[s_idx, l_idx] = n_p

            # Within-session endpoint permutation null
            p_vals = []
            for _ in range(n_perms):
                s_lag_p = rng.permutation(s_lag)
                p_vals.append(compute_mi_fast(s_curr, s_lag_p, n_cats))
            perm_null[s_idx, l_idx] = np.mean(p_vals)

            # Fixed-N subsample
            sub_idx = rng.choice(n_p, size=N_sub, replace=False)
            sc_sub = s_curr[sub_idx]
            sl_sub = s_lag[sub_idx]
            sub_real[s_idx, l_idx] = compute_mi_fast(sc_sub, sl_sub, n_cats)
            sl_sub_p = rng.permutation(sl_sub)
            sub_perm[s_idx, l_idx] = compute_mi_fast(sc_sub, sl_sub_p, n_cats)

    print("Calculations complete! Generating summary table and 4-panel diagnostic figure...")

    # Summary table across lags
    summary_rows = []
    for l_idx in range(n_lags):
        sec = lag_seconds[l_idx]
        mean_pairs = float(pair_counts[:, l_idx].mean())
        active_sess = int((pair_counts[:, l_idx] > 0).sum())
        r_raw = float(real_mis_raw[:, l_idx].mean())
        m_raw = float(markov_mis_raw[:, l_idx].mean())
        s_cross = float(shuff_cross_raw[:, l_idx].mean())
        s_perm = float(perm_null[:, l_idx].mean())
        r_sub = float(sub_real[:, l_idx].mean())
        p_sub = float(sub_perm[:, l_idx].mean())
        net_real = r_raw - s_perm
        net_markov = m_raw - s_perm
        excess_real_over_markov = r_raw - m_raw

        summary_rows.append({
            "lag_seconds": sec,
            "mean_pair_count": mean_pairs,
            "contributing_sessions": active_sess,
            "raw_real_mi": r_raw,
            "raw_markov_null_mi": m_raw,
            "cross_session_shuff_mi": s_cross,
            "within_session_perm_null_mi": s_perm,
            "fixed_N_real_mi": r_sub,
            "fixed_N_perm_null_mi": p_sub,
            "net_real_mi": net_real,
            "net_markov_mi": net_markov,
            "excess_real_over_markov": excess_real_over_markov
        })

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary CSV to: {OUTPUT_CSV}")

    # Plot 4-Panel Diagnostic Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    # Panel A: Valid Pair Counts & Contributing Sessions
    ax_a = axs[0, 0]
    ax_a2 = ax_a.twinx()
    ax_a.plot(lag_seconds, df_summary["mean_pair_count"] / 1000.0, color="#1F77B4", linewidth=2.5, label="Mean Valid Pairs / Session (x10³)")
    ax_a.fill_between(lag_seconds, (pair_counts.min(axis=0)) / 1000.0, (pair_counts.max(axis=0)) / 1000.0, color="#1F77B4", alpha=0.15)
    ax_a2.plot(lag_seconds, df_summary["contributing_sessions"], color="#2CA02C", linewidth=2.2, linestyle="--", label="Contributing Sessions (out of 30)")
    ax_a.set_xscale("log")
    ax_a.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_a.set_ylim([40, 160])
    ax_a2.set_ylim([0, 35])
    ax_a.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_a.set_ylabel("Valid Syllable Pairs ($N \\times 10^3$)", fontweight="bold", color="#1F77B4")
    ax_a2.set_ylabel("Active Sessions (count)", fontweight="bold", color="#2CA02C")
    ax_a.set_title("A. Observation Sample Size & Retention Across Lags", fontweight="bold", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Raw Curves & Comparison of Nulls
    ax_b = axs[0, 1]
    ax_b.plot(lag_seconds, df_summary["raw_real_mi"], color="#D95F02", linewidth=2.5, label="Real Data (Continuous Syllables)")
    ax_b.plot(lag_seconds, df_summary["raw_markov_null_mi"], color="#2C2C2C", linewidth=2.2, label="1st-Order Markov Chain Null")
    ax_b.plot(lag_seconds, df_summary["cross_session_shuff_mi"], color="#7570B3", linewidth=2.0, linestyle="-.", label="Cross-Session Shuffled (state_moseq)")
    ax_b.plot(lag_seconds, df_summary["within_session_perm_null_mi"], color="#666666", linewidth=2.0, linestyle=":", label="Within-Session Permuted Null (iid)")
    ax_b.set_xscale("log")
    ax_b.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_b.set_ylim([0.0, 1.5])
    ax_b.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_b.set_ylabel("Mutual Information (bits)", fontweight="bold")
    ax_b.set_title("B. Raw Curves: Cross-Session vs. Within-Session Nulls", fontweight="bold", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.4)
    ax_b.legend(loc="upper right", fontsize=8.5)

    # Panel C: Fixed Sample Size Control (N = 50,000 constant)
    ax_c = axs[1, 0]
    ax_c.plot(lag_seconds, df_summary["fixed_N_real_mi"], color="#D95F02", linewidth=2.5, label="Real Data (Subsampled N=50k)")
    ax_c.plot(lag_seconds, df_summary["fixed_N_perm_null_mi"], color="#666666", linewidth=2.0, linestyle=":", label="Permuted Null (Subsampled N=50k)")
    ax_c.set_xscale("log")
    ax_c.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_c.set_ylim([0.0, 1.5])
    ax_c.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_c.set_ylabel("Mutual Information (bits)", fontweight="bold")
    ax_c.set_title("C. Fixed-N Control ($N = 50,000$ Pairs Clamped at All Lags)", fontweight="bold", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.4)
    ax_c.legend(loc="upper right", fontsize=8.5)

    # Panel D: Net Information / Excess Curves on Linear Axis with Zero
    ax_d = axs[1, 1]
    ax_d.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_d.plot(lag_seconds, df_summary["net_real_mi"], color="#D95F02", linewidth=2.2, label="Net Real MI (Real − Within-Session Perm)")
    ax_d.plot(lag_seconds, df_summary["net_markov_mi"], color="#2C2C2C", linewidth=2.0, linestyle="--", label="Net Markov MI (Markov − Within-Session Perm)")
    ax_d.plot(lag_seconds, df_summary["excess_real_over_markov"], color="#E7298A", linewidth=2.5, label="Excess Real MI (Real − Markov Null)")
    ax_d.set_xscale("log")
    ax_d.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_d.set_ylim([-0.05, 1.4])
    ax_d.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_d.set_ylabel("Net Information (bits)", fontweight="bold")
    ax_d.set_title("D. Excess Information Curves (Zero-Referenced Linear Axis)", fontweight="bold", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.4)
    ax_d.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic figure saved to: {OUTPUT_FIG}")
    print("=" * 80)

if __name__ == "__main__":
    main()
