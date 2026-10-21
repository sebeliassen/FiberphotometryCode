#!/usr/bin/env python3
"""
generate_presentation_lag_figures.py
=============================================================================
Generates publication-quality presentation figures for:
1. WT Saline vs. 1st-Order Markov Null (2-panel: Raw & Baseline-Removed with bout peak).
2. All 4 Experimental Groups vs. 1st-Order Markov Null (4-panel 2x2 layout).
3. Direct Overlay Comparison of Excess Behavioral Memory (Delta I) across all 4 groups.

Outputs are saved directly to:
- /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/
- /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/
- /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations/
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

OUTPUT_DIRS = [
    os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output"),
    os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures"),
    os.path.join(BASE_DIR, "syllable_generation", "presentations"),
]

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
    })

def main():
    print("=" * 80)
    print("GENERATING PRESENTATION LAG TEST FIGURES")
    print("=" * 80)

    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    lags = bundle["lags"]
    lag_seconds = bundle["lag_seconds"]
    real_mis = bundle["real_mis"]      # (30, 35)
    markov_mis = bundle["markov_mis"]  # (30, 35)
    shuff_mis = bundle["shuff_mis"]    # (30, 35)
    keys = bundle["keys"]
    meta = bundle["metadata_dict"]

    excess_mis = real_mis - markov_mis
    real_sub = real_mis - shuff_mis
    markov_sub = markov_mis - shuff_mis

    groups = {
        "WT_Saline": [],
        "WT_Amphetamine": [],
        "KI_Saline": [],
        "KI_Amphetamine": []
    }
    for i, k in enumerate(keys):
        c = meta[k]["cohort"]
        groups[c].append(i)

    group_colors = {
        "WT_Saline": "#1F77B4",        # Royal Blue
        "WT_Amphetamine": "#D62728",   # Crimson Red
        "KI_Saline": "#2CA02C",        # Forest Green
        "KI_Amphetamine": "#FF7F0E",   # Amber Orange
    }
    c_markov = "#252525"              # Charcoal

    set_style()

    # =========================================================================
    # FIGURE 1: WT Saline vs. 1st-Order Markov Null (1:1 Aspect Ratio for Slides)
    # =========================================================================
    wt_sal_idx = groups["WT_Saline"]
    r_wts = real_mis[wt_sal_idx]
    m_wts = markov_mis[wt_sal_idx]
    s_wts = shuff_mis[wt_sal_idx]

    r_mean = r_wts.mean(axis=0)
    r_sem = r_wts.std(axis=0) / np.sqrt(len(wt_sal_idx))
    m_mean = m_wts.mean(axis=0)
    m_sem = m_wts.std(axis=0) / np.sqrt(len(wt_sal_idx))

    ex_wts = excess_mis[wt_sal_idx]
    ex_mean = ex_wts.mean(axis=0)
    peak_idx = int(np.argmax(ex_mean))

    fig1, ax1 = plt.subplots(figsize=(7.8, 7.8), dpi=300)

    # Individual session traces
    for s in range(len(wt_sal_idx)):
        ax1.plot(lag_seconds, r_wts[s], color="#1F77B4", alpha=0.15, linewidth=1.0)
        ax1.plot(lag_seconds, m_wts[s], color=c_markov, alpha=0.12, linewidth=1.0)

    # Shading between real and markov curves (Excess behavioral memory)
    ax1.fill_between(
        lag_seconds, m_mean, r_mean, where=(r_mean >= m_mean),
        color="#FDBF6F", alpha=0.45, label=r"Excess Behavioral Memory ($\Delta I > 0$)"
    )

    # Real and Markov mean curves
    ax1.plot(lag_seconds, r_mean, color="#1F77B4", linewidth=3.2, label="Real Mouse Syllables (WT Saline, $N=8$)")
    ax1.fill_between(lag_seconds, r_mean - r_sem, r_mean + r_sem, color="#1F77B4", alpha=0.25)
    ax1.plot(lag_seconds, m_mean, color=c_markov, linewidth=2.6, linestyle="--", label="1st-Order Markov Chain Null")
    ax1.fill_between(lag_seconds, m_mean - m_sem, m_mean + m_sem, color=c_markov, alpha=0.18)

    # Bout peak annotation at tau = 6.2s
    peak_lag = lag_seconds[peak_idx]
    peak_val = r_mean[peak_idx]
    ax1.scatter([peak_lag], [peak_val], color="#D95F02", s=90, zorder=6)
    ax1.annotate(
        f"Natural Bout Peak\n+{ex_mean[peak_idx]:.3f} bits ($\\tau = {peak_lag:.1f}$s)",
        xy=(peak_lag, peak_val),
        xytext=(peak_lag * 2.6, peak_val * 1.15),
        arrowprops=dict(facecolor="#D95F02", edgecolor="#D95F02", arrowstyle="->", lw=1.8),
        fontweight="bold", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF3E0", edgecolor="#D95F02", lw=1.2)
    )

    ax1.set_xscale("log")
    ax1.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax1.set_ylim([0.0, 1.55])
    ax1.set_xlabel("Time Lag $\\tau$ (seconds, log scale)", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Mutual Information $I(S_t; S_{t+\\tau})$ (bits)", fontweight="bold", fontsize=13)
    ax1.set_title("Wild-Type Saline ($N=8$ Mice):\nReal Syllables vs. 1st-Order Markov Null", fontweight="bold", fontsize=14, pad=14)
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95, fontsize=10.5)

    plt.tight_layout()

    fig1_name = "pres_wt_saline_vs_markov_null.png"
    for d in OUTPUT_DIRS:
        out_path = os.path.join(d, fig1_name)
        fig1.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: 4 Groups vs. 1st-Order Markov Null (2x2 Grid)
    # =========================================================================
    fig2, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    cohort_configs = [
        ("WT_Saline", "A. Wild-Type Saline (Normative Baseline, $N=8$)", group_colors["WT_Saline"], axs[0, 0]),
        ("WT_Amphetamine", "B. Wild-Type Amphetamine (Hyper-Persistence, $N=8$)", group_colors["WT_Amphetamine"], axs[0, 1]),
        ("KI_Saline", "C. DAT-KI Saline (Mutant Baseline, $N=7$)", group_colors["KI_Saline"], axs[1, 0]),
        ("KI_Amphetamine", "D. DAT-KI Amphetamine (Syntax Disruption / Looping, $N=7$)", group_colors["KI_Amphetamine"], axs[1, 1]),
    ]

    for g_key, title, col, ax in cohort_configs:
        idxs = groups[g_key]
        g_real = real_mis[idxs]
        g_markov = markov_mis[idxs]
        g_excess = excess_mis[idxs]

        r_m = g_real.mean(axis=0)
        r_s = g_real.std(axis=0) / np.sqrt(len(idxs))
        m_m = g_markov.mean(axis=0)
        m_s = g_markov.std(axis=0) / np.sqrt(len(idxs))
        ex_m = g_excess.mean(axis=0)

        # Traces for individual animals
        for s in range(len(idxs)):
            ax.plot(lag_seconds, g_real[s], color=col, alpha=0.15, linewidth=0.8)
            ax.plot(lag_seconds, g_markov[s], color=c_markov, alpha=0.10, linewidth=0.8)

        # Real Mean
        ax.plot(lag_seconds, r_m, color=col, linewidth=3.0, label=f"Real Data ({g_key.replace('_', ' ')})")
        ax.fill_between(lag_seconds, r_m - r_s, r_m + r_s, color=col, alpha=0.25)

        # Markov Null Mean
        ax.plot(lag_seconds, m_m, color=c_markov, linewidth=2.2, linestyle="--", label="1st-Order Markov Null")
        ax.fill_between(lag_seconds, m_m - m_s, m_m + m_s, color=c_markov, alpha=0.18)

        # Shading between real and markov
        if g_key != "KI_Amphetamine":
            ax.fill_between(lag_seconds, m_m, r_m, where=(r_m >= m_m), color=col, alpha=0.12, label=r"Excess Memory ($\Delta I > 0$)")
        else:
            # Highlight negative excess memory for DAT-KI Amphetamine
            ax.fill_between(lag_seconds, r_m, m_m, where=(m_m >= r_m), color="#888888", alpha=0.15, label=r"Negative Excess ($\Delta I < 0$)")

        # Peak annotation
        p_idx = int(np.argmax(ex_m))
        if ex_m[p_idx] > 0.03:
            p_lag = lag_seconds[p_idx]
            p_val = r_m[p_idx]
            ax.scatter([p_lag], [p_val], color=col, s=70, zorder=5)
            ax.annotate(
                f"Peak: +{ex_m[p_idx]:.3f} bits\n($\\tau = {p_lag:.1f}$s)",
                xy=(p_lag, p_val),
                xytext=(p_lag * 2.5, p_val * 1.15),
                arrowprops=dict(facecolor=col, edgecolor=col, arrowstyle="->", lw=1.5),
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, lw=1.0)
            )

        ax.set_xscale("log")
        ax.set_xlim([lag_seconds[0], lag_seconds[-1]])
        ax.set_ylim([0.0, 1.65])
        ax.set_xlabel("Time Lag $\\tau$ (seconds, log scale)", fontweight="bold")
        ax.set_ylabel("Mutual Information (bits)", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.92, fontsize=9.5)

    fig2.suptitle("Behavioral Timescales Across Cohorts: Real Data vs. 1st-Order Markov Null", fontweight="bold", y=0.995)
    plt.tight_layout()

    fig2_name = "pres_4_groups_vs_markov_null.png"
    for d in OUTPUT_DIRS:
        out_path = os.path.join(d, fig2_name)
        fig2.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3: Bonus Slide-Ready Direct Overlay Comparison of Excess Memory (Delta I)
    # =========================================================================
    fig3, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)

    ax_c1.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    for g_key in ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        idxs = groups[g_key]
        g_ex = excess_mis[idxs]
        mean_ex = g_ex.mean(axis=0)
        sem_ex = g_ex.std(axis=0) / np.sqrt(len(idxs))
        color = group_colors[g_key]
        lbl = g_key.replace("_", " ")

        ax_c1.plot(lag_seconds, mean_ex, color=color, linewidth=2.8, label=f"{lbl} ($N={len(idxs)}$)")
        ax_c1.fill_between(lag_seconds, mean_ex - sem_ex, mean_ex + sem_ex, color=color, alpha=0.18)

    # Annotate WT Saline peak
    ax_c1.scatter([lag_seconds[peak_idx]], [ex_mean[peak_idx]], color="#1F77B4", s=80, zorder=6)
    ax_c1.annotate(
        f"WT Saline Peak\n+{ex_mean[peak_idx]:.3f} bits (6.2s)",
        xy=(lag_seconds[peak_idx], ex_mean[peak_idx]),
        xytext=(lag_seconds[peak_idx] * 2.3, ex_mean[peak_idx] * 1.2),
        arrowprops=dict(facecolor="#1F77B4", edgecolor="#1F77B4", arrowstyle="->", lw=1.8),
        fontweight="bold", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1F77B4", lw=1.0)
    )

    ax_c1.set_xscale("log")
    ax_c1.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_c1.set_ylim([-0.18, 0.35])
    ax_c1.set_xlabel("Time Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_c1.set_ylabel(r"Excess Mutual Information $\Delta I$ (bits)", fontweight="bold")
    ax_c1.set_title("A. Excess Behavioral Memory Across Lags ($\\Delta I = I_{\\mathrm{real}} - I_{\\mathrm{markov}}$)", fontweight="bold", pad=12)
    ax_c1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_c1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.95)

    # Boxplot of individual mouse peak memory at tau = 6.2s
    ax_c2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    box_data = []
    box_labels = ["WT\nSaline", "WT\nAmph", "DAT-KI\nSaline", "DAT-KI\nAmph"]
    box_cols = [group_colors["WT_Saline"], group_colors["WT_Amphetamine"], group_colors["KI_Saline"], group_colors["KI_Amphetamine"]]

    for g_key in ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        idxs = groups[g_key]
        box_data.append(excess_mis[idxs, peak_idx])

    bplot = ax_c2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.55, showmeans=True,
                          meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6.5))
    for patch, c in zip(bplot["boxes"], box_cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for pos, vals in enumerate(box_data, 1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax_c2.scatter(np.full_like(vals, pos) + jitter, vals, color="black", s=32, alpha=0.8, zorder=5)

    ax_c2.set_ylabel("Excess Memory at Bout Scale $\\tau = 6.2$s (bits)", fontweight="bold")
    ax_c2.set_title("B. Individual Mouse Bout Memory at $\\tau = 6.2$s", fontweight="bold", pad=12)
    ax_c2.grid(True, linestyle="--", alpha=0.35, axis="y")

    fig3.suptitle("Dopaminergic Disruption of Behavioral Memory: Hyper-Persistence vs. Syntax Breakdown", fontweight="bold", y=0.99)
    plt.tight_layout()

    fig3_name = "pres_4_groups_excess_memory_comparison.png"
    for d in OUTPUT_DIRS:
        out_path = os.path.join(d, fig3_name)
        fig3.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig3)

    print("=" * 80)
    print("ALL PRESENTATION FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
