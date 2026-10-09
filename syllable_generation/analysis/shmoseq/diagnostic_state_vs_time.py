#!/usr/bin/env python3
"""
diagnostic_state_vs_time.py
=============================================================================
Rigorous 4-way information-theoretic decomposition:
1. 1st-Order Markov Null (Syllable transition inertia)
2. shMoSeq Latent State (Z_t alone)
3. Time in Arena (t alone, 12 5-minute epochs or continuous kernel)
4. Time + State Joint (t + Z_t)

Tests whether the multi-second behavioral memory (~6.3s peak) is merely
a proxy for time in arena (habituation drift) or reflects genuine internal
macro-state organization.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "models", "selected_exploratory_model.pkl")
CACHED_MI = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "lagged_mi_results_1000s.pkl")
OUTPUT_FIG = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_timescale_state_vs_time_decomposition.png")
OUTPUT_CSV = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "tables", "diagnostic_state_vs_time_summary.csv")

def compute_mi_2d(x, y, nx, ny, pseudo=1e-8):
    """Vectorized 2D mutual information in bits."""
    flat = x * ny + y
    counts = np.bincount(flat, minlength=nx * ny).reshape((nx, ny)).astype(np.float64) + pseudo
    probs = counts / counts.sum()
    px = probs.sum(axis=1, keepdims=True)
    py = probs.sum(axis=0, keepdims=True)
    mi_nats = np.sum(probs * np.log(probs / (px * py)))
    return float(mi_nats / np.log(2.0))

def main():
    print("=" * 80)
    print("EVALUATING 4-WAY DECOMPOSITION: 1ST-ORDER MARKOV, STATE, TIME, TIME+STATE")
    print("=" * 80)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
    sylls = bundle["syllables_dict"]
    keys = list(sylls.keys())
    fps = bundle.get("fps", 40.0125)
    n_cats = max([v.max() for v in sylls.values()]) + 1

    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)
    states_dict = model_bundle["states_dict"]
    n_states = 8

    with open(CACHED_MI, "rb") as f:
        cached = pickle.load(f)
    lags = cached["lags"]
    lag_seconds = cached["lag_seconds"]
    # Convert cached nats to bits
    real_mis = cached["real_mis"] / np.log(2.0)
    markov_mis = cached["markov_mis"] / np.log(2.0)
    shuff_mis = cached["shuff_mis"] / np.log(2.0)

    n_sessions = len(keys)
    n_lags = len(lags)
    n_time_bins = 12 # 5-minute epochs across 60 minutes

    print(f"Cohort: {n_sessions} sessions, {n_lags} lags (1s to 1000s), {n_states} states, {n_time_bins} time bins")

    mi_state_arr = np.zeros((n_sessions, n_lags), dtype=np.float64)
    mi_time_arr = np.zeros((n_sessions, n_lags), dtype=np.float64)
    mi_time_state_arr = np.zeros((n_sessions, n_lags), dtype=np.float64)

    for l_idx, lag in enumerate(lags):
        for s_idx, k in enumerate(keys):
            s = sylls[k]
            z = states_dict[k]
            T = len(s)
            time_bins = np.linspace(0, n_time_bins, T, endpoint=False).astype(int)

            s_past = s[:-lag]
            s_future = s[lag:]
            z_past = z[:-lag]
            t_past = time_bins[:-lag]
            tz_past = t_past * n_states + z_past

            mi_state_arr[s_idx, l_idx] = compute_mi_2d(z_past, s_future, n_states, n_cats)
            mi_time_arr[s_idx, l_idx] = compute_mi_2d(t_past, s_future, n_time_bins, n_cats)
            mi_time_state_arr[s_idx, l_idx] = compute_mi_2d(tz_past, s_future, n_time_bins * n_states, n_cats)

    # Compute cohort statistics (mean ± SEM in bits)
    r_mean = real_mis.mean(axis=0)
    m_mean = markov_mis.mean(axis=0)
    z_mean = mi_state_arr.mean(axis=0)
    t_mean = mi_time_arr.mean(axis=0)
    tz_mean = mi_time_state_arr.mean(axis=0)

    # Independent information of state beyond time: I(Z; S | Time) = I(Z, Time; S) - I(Time; S)
    state_given_time = tz_mean - t_mean
    # Excess of real data over 1st order Markov
    excess_real_markov = r_mean - m_mean

    # Save summary CSV
    summary_df = pd.DataFrame({
        "lag_seconds": lag_seconds,
        "real_data_bits": r_mean,
        "markov_1st_null_bits": m_mean,
        "state_alone_bits": z_mean,
        "time_in_arena_bits": t_mean,
        "time_plus_state_bits": tz_mean,
        "state_given_time_bits": state_given_time,
        "excess_real_over_markov_bits": excess_real_markov
    })
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary CSV to: {OUTPUT_CSV}")

    # =========================================================================
    # CREATE EXTENDED 3-PANEL FIGURE (Raw, Baseline-Removed, & Net Gains)
    # =========================================================================
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), dpi=250)

    # Color Palette
    c_real = "#D95F02"       # Orange (Real Data)
    c_markov = "#252525"     # Charcoal (1st-Order Markov)
    c_state = "#1B9E77"      # Teal (shMoSeq State Z_t)
    c_time = "#7570B3"       # Purple (Time in Arena)
    c_timestate = "#E6AB02"  # Gold (Time + State)
    c_excess = "#E7298A"     # Magenta

    # -------------------------------------------------------------------------
    # Panel A: Raw Mutual Information Across Predictors
    # -------------------------------------------------------------------------
    ax_a = axs[0]
    ax_a.plot(lag_seconds, r_mean, color=c_real, linewidth=2.6, label="Real Syllables ($S_t$)")
    ax_a.plot(lag_seconds, m_mean, color=c_markov, linewidth=2.2, linestyle="--", label="1st-Order Markov Null ($M_t$)")
    ax_a.plot(lag_seconds, tz_mean, color=c_timestate, linewidth=2.4, label="Time + State ($t, Z_t$)")
    ax_a.plot(lag_seconds, t_mean, color=c_time, linewidth=2.0, linestyle=":", label="Time in Arena ($t$)")
    ax_a.plot(lag_seconds, z_mean, color=c_state, linewidth=2.2, label="shMoSeq State ($Z_t$)")

    ax_a.set_xscale("log")
    ax_a.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_a.set_ylim([0.0, 2.0])
    ax_a.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_a.set_ylabel("Predictive Information $I(X_t; S_{t+\\tau})$ (bits)", fontweight="bold")
    ax_a.set_title("A. Raw Predictive Information by Model", fontweight="bold", fontsize=11)
    ax_a.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_a.legend(loc="upper right", frameon=True, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel B: Baseline-Removed (Time-in-Arena Baseline Subtracted)
    # -------------------------------------------------------------------------
    ax_b = axs[1]
    # Subtract Time-in-Arena baseline from each model
    ax_b.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)
    ax_b.plot(lag_seconds, np.maximum(r_mean - t_mean, 0), color=c_real, linewidth=2.6, label="Real Data − Time ($S_t \\mid t$)")
    ax_b.plot(lag_seconds, np.maximum(m_mean - t_mean, 0), color=c_markov, linewidth=2.2, linestyle="--", label="1st-Order Markov − Time")
    ax_b.plot(lag_seconds, state_given_time, color=c_timestate, linewidth=2.4, label="State Information Beyond Time ($Z_t \\mid t$)")
    ax_b.plot(lag_seconds, np.zeros_like(lag_seconds), color=c_time, linewidth=2.0, linestyle=":", label="Time Baseline (Zero Reference)")

    # Shade excess of real over markov
    ax_b.fill_between(lag_seconds, np.maximum(m_mean - t_mean, 0), np.maximum(r_mean - t_mean, 0), color="#FDBF6F", alpha=0.45, label="Excess Behavioral Memory")

    ax_b.set_xscale("log")
    ax_b.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_b.set_ylim([-0.05, 1.6])
    ax_b.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_b.set_ylabel("Net Information Beyond Time (bits)", fontweight="bold")
    ax_b.set_title("B. Non-Stationary Time Baseline Removed", fontweight="bold", fontsize=11)
    ax_b.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", frameon=True, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel C: Net Incremental Information (Proving State != Time)
    # -------------------------------------------------------------------------
    ax_c = axs[2]
    ax_c.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)
    ax_c.plot(lag_seconds, excess_real_markov, color=c_excess, linewidth=2.6, label="Real Excess Memory (Real − 1st Markov)")
    ax_c.plot(lag_seconds, state_given_time, color=c_state, linewidth=2.4, linestyle="-.", label="State Info Beyond Time ($I(Z; S \\mid t)$)")

    peak_idx = np.argmax(excess_real_markov)
    ax_c.scatter([lag_seconds[peak_idx]], [excess_real_markov[peak_idx]], color="#C51B7D", s=70, zorder=5)
    ax_c.annotate(f"Excess Peak: +{excess_real_markov[peak_idx]:.3f} bits\n(at $\\tau = {lag_seconds[peak_idx]:.1f}$s)",
                  xy=(lag_seconds[peak_idx], excess_real_markov[peak_idx]),
                  xytext=(lag_seconds[peak_idx] * 2.2, excess_real_markov[peak_idx] * 1.15),
                  arrowprops=dict(facecolor="#C51B7D", arrowstyle="->", lw=1.5),
                  fontweight="bold", fontsize=8.5)

    ax_c.set_xscale("log")
    ax_c.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_c.set_ylim([-0.03, 0.25])
    ax_c.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_c.set_ylabel("Incremental Information (bits)", fontweight="bold")
    ax_c.set_title("C. Independent State Contribution vs. Memory", fontweight="bold", fontsize=11)
    ax_c.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved extended 3-panel figure to: {OUTPUT_FIG}")
    print("=" * 80)

if __name__ == "__main__":
    main()
