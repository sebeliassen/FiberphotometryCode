#!/usr/bin/env python3
"""
plot_lagged_mi_clean_baseline_removed.py
=============================================================================
Generates clean publication figures containing ONLY Real Data and 1st-Order Null:
1. Raw Lagged MI (Real Data vs. 1st-Order Markov Null).
2. Baseline-Removed Lagged MI (Subtracting the shared non-stationary baseline),
   showing both Real and Markov curves decaying to 0 with excess shaded.
3. Net Excess Information curve (Real Data minus 1st-Order Markov Null).
=============================================================================
"""

import os
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "lagged_mi_results_1000s.pkl")
OUTPUT_FIG_2PANEL = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_real_vs_markov_baseline_removed.png")
OUTPUT_FIG_3PANEL = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_real_vs_markov_3panel.png")

def main():
    print("=" * 80)
    print("GENERATING CLEAN REAL VS. MARKOV (BASELINE-REMOVED) FIGURES")
    print("=" * 80)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    lags = bundle["lags"]
    lag_seconds = bundle["lag_seconds"]
    real_mis = bundle["real_mis"]      # (30, 35)
    markov_mis = bundle["markov_mis"]  # (30, 35)
    shuff_mis = bundle["shuff_mis"]    # (30, 35)

    n_sessions, n_lags = real_mis.shape
    print(f"Loaded {n_sessions} sessions across {n_lags} lags (1.0s to 1000.0s)")

    # Baseline-subtracted curves (subtracting the shared non-stationary baseline per session)
    real_sub = real_mis - shuff_mis
    markov_sub = markov_mis - shuff_mis
    excess_mi = real_mis - markov_mis

    # Colors
    c_real = "#D95F02"     # Burnt orange
    c_markov = "#252525"   # Charcoal black
    c_excess = "#E7298A"   # Vibrant magenta
    c_shade = "#FDBF6F"    # Light orange shading

    # =========================================================================
    # 1. GENERATE 2-PANEL FIGURE (Raw vs. Baseline-Removed)
    # =========================================================================
    fig2, axs = plt.subplots(1, 2, figsize=(14, 5.2), dpi=250)

    # Panel A: Raw Lagged MI (Real vs. Markov only)
    ax_a = axs[0]
    for s in range(n_sessions):
        ax_a.plot(lag_seconds, real_mis[s], color=c_real, alpha=0.10, linewidth=0.7)
        ax_a.plot(lag_seconds, markov_mis[s], color=c_markov, alpha=0.08, linewidth=0.7)

    r_mean = real_mis.mean(axis=0)
    r_sem = real_mis.std(axis=0) / np.sqrt(n_sessions)
    m_mean = markov_mis.mean(axis=0)
    m_sem = markov_mis.std(axis=0) / np.sqrt(n_sessions)

    ax_a.plot(lag_seconds, r_mean, color=c_real, linewidth=2.8, label="Real Data (Continuous Syllables)")
    ax_a.fill_between(lag_seconds, r_mean - r_sem, r_mean + r_sem, color=c_real, alpha=0.25)
    ax_a.plot(lag_seconds, m_mean, color=c_markov, linewidth=2.4, label="1st-Order Markov Chain Null")
    ax_a.fill_between(lag_seconds, m_mean - m_sem, m_mean + m_sem, color=c_markov, alpha=0.20)

    ax_a.set_xscale("log")
    ax_a.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_a.set_ylim([0.0, 1.5])
    ax_a.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Lagged Mutual Information (bits)", fontsize=11, fontweight="bold")
    ax_a.set_title("A. Raw Lagged MI: Real Data vs. 1st-Order Markov Null", fontsize=12, fontweight="bold")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)

    # Panel B: Baseline-Removed Lagged MI
    ax_b = axs[1]
    for s in range(n_sessions):
        ax_b.plot(lag_seconds, real_sub[s], color=c_real, alpha=0.10, linewidth=0.7)
        ax_b.plot(lag_seconds, markov_sub[s], color=c_markov, alpha=0.08, linewidth=0.7)

    r_sub_mean = real_sub.mean(axis=0)
    r_sub_sem = real_sub.std(axis=0) / np.sqrt(n_sessions)
    m_sub_mean = markov_sub.mean(axis=0)
    m_sub_sem = markov_sub.std(axis=0) / np.sqrt(n_sessions)

    ax_b.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)
    ax_b.plot(lag_seconds, r_sub_mean, color=c_real, linewidth=2.8, label="Real Data (Baseline Removed)")
    ax_b.fill_between(lag_seconds, r_sub_mean - r_sub_sem, r_sub_mean + r_sub_sem, color=c_real, alpha=0.25)
    ax_b.plot(lag_seconds, m_sub_mean, color=c_markov, linewidth=2.4, linestyle="--", label="1st-Order Markov Null (Baseline Removed)")
    ax_b.fill_between(lag_seconds, m_sub_mean - m_sub_sem, m_sub_mean + m_sub_sem, color=c_markov, alpha=0.20)

    # Shade the excess behavioral memory between Real and Markov
    ax_b.fill_between(lag_seconds, m_sub_mean, r_sub_mean, color=c_shade, alpha=0.45, label=r"Excess Behavioral Memory ($\Delta I > 0$)")

    ax_b.set_xscale("log")
    ax_b.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_b.set_ylim([-0.05, 1.15])
    ax_b.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel(r"Net Mutual Information $\Delta I$ (bits)", fontsize=11, fontweight="bold")
    ax_b.set_title("B. Baseline Removed: Excess Behavioral Memory Highlighted", fontsize=12, fontweight="bold")
    ax_b.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG_2PANEL), exist_ok=True)
    plt.savefig(OUTPUT_FIG_2PANEL, bbox_inches="tight")
    plt.close()
    print(f"Saved 2-panel figure to: {OUTPUT_FIG_2PANEL}")

    # =========================================================================
    # 2. GENERATE 3-PANEL FIGURE (Raw, Baseline-Removed, Excess Curve Alone)
    # =========================================================================
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5.0), dpi=250)

    # 3-Panel A: Raw
    axs3[0].plot(lag_seconds, r_mean, color=c_real, linewidth=2.8, label="Real Data")
    axs3[0].fill_between(lag_seconds, r_mean - r_sem, r_mean + r_sem, color=c_real, alpha=0.25)
    axs3[0].plot(lag_seconds, m_mean, color=c_markov, linewidth=2.4, label="1st-Order Markov Null")
    axs3[0].fill_between(lag_seconds, m_mean - m_sem, m_mean + m_sem, color=c_markov, alpha=0.20)
    for s in range(n_sessions):
        axs3[0].plot(lag_seconds, real_mis[s], color=c_real, alpha=0.08, linewidth=0.6)
        axs3[0].plot(lag_seconds, markov_mis[s], color=c_markov, alpha=0.06, linewidth=0.6)
    axs3[0].set_xscale("log")
    axs3[0].set_xlim([lag_seconds[0], lag_seconds[-1]])
    axs3[0].set_ylim([0.0, 1.5])
    axs3[0].set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    axs3[0].set_ylabel("Mutual Information (bits)", fontweight="bold")
    axs3[0].set_title("A. Raw Lagged MI", fontweight="bold", fontsize=11)
    axs3[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axs3[0].legend(loc="upper right", frameon=True, fontsize=9)

    # 3-Panel B: Baseline Removed
    axs3[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    axs3[1].plot(lag_seconds, r_sub_mean, color=c_real, linewidth=2.8, label="Real Data (Net)")
    axs3[1].fill_between(lag_seconds, r_sub_mean - r_sub_sem, r_sub_mean + r_sub_sem, color=c_real, alpha=0.25)
    axs3[1].plot(lag_seconds, m_sub_mean, color=c_markov, linewidth=2.4, linestyle="--", label="1st-Order Markov Null (Net)")
    axs3[1].fill_between(lag_seconds, m_sub_mean - m_sub_sem, m_sub_mean + m_sub_sem, color=c_markov, alpha=0.20)
    axs3[1].fill_between(lag_seconds, m_sub_mean, r_sub_mean, color=c_shade, alpha=0.45, label="Excess Memory")
    for s in range(n_sessions):
        axs3[1].plot(lag_seconds, real_sub[s], color=c_real, alpha=0.08, linewidth=0.6)
        axs3[1].plot(lag_seconds, markov_sub[s], color=c_markov, alpha=0.06, linewidth=0.6)
    axs3[1].set_xscale("log")
    axs3[1].set_xlim([lag_seconds[0], lag_seconds[-1]])
    axs3[1].set_ylim([-0.05, 1.15])
    axs3[1].set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    axs3[1].set_ylabel("Net Mutual Information (bits)", fontweight="bold")
    axs3[1].set_title("B. Baseline Removed (Floor Corrected)", fontweight="bold", fontsize=11)
    axs3[1].grid(True, which="both", linestyle="--", alpha=0.35)
    axs3[1].legend(loc="upper right", frameon=True, fontsize=9)

    # 3-Panel C: Excess Information Alone (Real − Markov)
    ex_mean = excess_mi.mean(axis=0)
    ex_sem = excess_mi.std(axis=0) / np.sqrt(n_sessions)
    axs3[2].axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)
    axs3[2].plot(lag_seconds, ex_mean, color=c_excess, linewidth=2.8, label="Excess MI (Real − Markov Null)")
    axs3[2].fill_between(lag_seconds, ex_mean - ex_sem, ex_mean + ex_sem, color=c_excess, alpha=0.25)
    for s in range(n_sessions):
        axs3[2].plot(lag_seconds, excess_mi[s], color=c_excess, alpha=0.08, linewidth=0.6)

    peak_idx = np.argmax(ex_mean)
    axs3[2].scatter([lag_seconds[peak_idx]], [ex_mean[peak_idx]], color="#C51B7D", s=70, zorder=5)
    axs3[2].annotate(f"Peak: +{ex_mean[peak_idx]:.3f} bits\n(at $\\tau = {lag_seconds[peak_idx]:.1f}$s)",
                     xy=(lag_seconds[peak_idx], ex_mean[peak_idx]),
                     xytext=(lag_seconds[peak_idx] * 2.5, ex_mean[peak_idx] * 1.15),
                     arrowprops=dict(facecolor="#C51B7D", arrowstyle="->", lw=1.5),
                     fontweight="bold", fontsize=9)

    axs3[2].set_xscale("log")
    axs3[2].set_xlim([lag_seconds[0], lag_seconds[-1]])
    axs3[2].set_ylim([-0.03, 0.12])
    axs3[2].set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    axs3[2].set_ylabel("Excess Mutual Information (bits)", fontweight="bold")
    axs3[2].set_title("C. Net Behavioral Memory ($\\Delta I = I_{\\mathrm{real}} - I_{\\mathrm{markov}}$)", fontweight="bold", fontsize=11)
    axs3[2].grid(True, which="both", linestyle="--", alpha=0.35)
    axs3[2].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG_3PANEL), exist_ok=True)
    plt.savefig(OUTPUT_FIG_3PANEL, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-panel figure to: {OUTPUT_FIG_3PANEL}")
    print("=" * 80)

if __name__ == "__main__":
    main()
