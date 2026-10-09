#!/usr/bin/env python3
"""
plot_lagged_mi_by_group.py
=============================================================================
Evaluates and visualizes Lagged Mutual Information and Excess Behavioral Memory
across the 4 experimental groups:
1. WT Saline (Normative baseline, N=8)
2. WT Amphetamine (Psychostimulant hyperlocomotion/stereotypy, N=8)
3. DAT-KI Saline (Hyperdopaminergic mutant baseline, N=7)
4. DAT-KI Amphetamine (Mutant drug challenge, N=7)
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
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "lagged_mi_results_1000s.pkl")
OUTPUT_FIG = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_timescale_by_group_comparison.png")
OUTPUT_CSV = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "tables", "diagnostic_timescale_by_group_summary.csv")

def main():
    print("=" * 80)
    print("ANALYZING BEHAVIORAL TIMESCALES & EXCESS MEMORY ACROSS EXPERIMENTAL GROUPS")
    print("=" * 80)

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
    n_sessions, n_lags = real_mis.shape

    groups = {
        "WT_Saline": [],
        "WT_Amphetamine": [],
        "KI_Saline": [],
        "KI_Amphetamine": []
    }
    for i, k in enumerate(keys):
        c = meta[k]["cohort"]
        groups[c].append(i)

    group_labels = {
        "WT_Saline": "WT Saline ($N=8$)",
        "WT_Amphetamine": "WT Amphetamine ($N=8$)",
        "KI_Saline": "DAT-KI Saline ($N=7$)",
        "KI_Amphetamine": "DAT-KI Amphetamine ($N=7$)"
    }
    group_colors = {
        "WT_Saline": "#2B83BA",       # Blue
        "WT_Amphetamine": "#D7191C",  # Crimson Red
        "KI_Saline": "#ABDDA4",       # Mint Green
        "KI_Amphetamine": "#FDAE61"   # Orange/Gold
    }

    # Summary table
    rows = []
    for g_key, idxs in groups.items():
        g_real = real_mis[idxs]
        g_markov = markov_mis[idxs]
        g_excess = excess_mis[idxs]
        g_mean_excess = g_excess.mean(axis=0)
        peak_idx = int(np.argmax(g_mean_excess))

        for l_idx in range(n_lags):
            rows.append({
                "group": g_key,
                "lag_seconds": lag_seconds[l_idx],
                "real_mi_mean": float(g_real[:, l_idx].mean()),
                "real_mi_sem": float(g_real[:, l_idx].std() / np.sqrt(len(idxs))),
                "markov_mi_mean": float(g_markov[:, l_idx].mean()),
                "markov_mi_sem": float(g_markov[:, l_idx].std() / np.sqrt(len(idxs))),
                "excess_mi_mean": float(g_excess[:, l_idx].mean()),
                "excess_mi_sem": float(g_excess[:, l_idx].std() / np.sqrt(len(idxs))),
                "is_peak": (l_idx == peak_idx)
            })

    df_summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved group summary CSV to: {OUTPUT_CSV}")

    # =========================================================================
    # GENERATE 4-PANEL PUBLICATION FIGURE
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=250)

    # -------------------------------------------------------------------------
    # Panel A: WT Cohort (Saline vs. Amphetamine Raw & Markov)
    # -------------------------------------------------------------------------
    ax_a = axs[0, 0]
    wt_sal_idx = groups["WT_Saline"]
    wt_amph_idx = groups["WT_Amphetamine"]

    r_wts = real_mis[wt_sal_idx].mean(axis=0)
    m_wts = markov_mis[wt_sal_idx].mean(axis=0)
    r_wta = real_mis[wt_amph_idx].mean(axis=0)
    m_wta = markov_mis[wt_amph_idx].mean(axis=0)

    ax_a.plot(lag_seconds, r_wts, color="#2B83BA", linewidth=2.5, label="WT Saline (Real Data)")
    ax_a.plot(lag_seconds, m_wts, color="#2B83BA", linewidth=1.8, linestyle="--", label="WT Saline (1st Markov Null)")
    ax_a.plot(lag_seconds, r_wta, color="#D7191C", linewidth=2.5, label="WT Amphetamine (Real Data)")
    ax_a.plot(lag_seconds, m_wta, color="#D7191C", linewidth=1.8, linestyle="--", label="WT Amphetamine (1st Markov Null)")

    ax_a.set_xscale("log")
    ax_a.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_a.set_ylim([0.0, 1.6])
    ax_a.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_a.set_ylabel("Mutual Information (bits)", fontweight="bold")
    ax_a.set_title("A. Wild-Type (WT): Saline vs. Amphetamine Timescales", fontweight="bold", fontsize=11)
    ax_a.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_a.legend(loc="upper right", frameon=True, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel B: DAT-KI Cohort (Saline vs. Amphetamine Raw & Markov)
    # -------------------------------------------------------------------------
    ax_b = axs[0, 1]
    ki_sal_idx = groups["KI_Saline"]
    ki_amph_idx = groups["KI_Amphetamine"]

    r_kis = real_mis[ki_sal_idx].mean(axis=0)
    m_kis = markov_mis[ki_sal_idx].mean(axis=0)
    r_kia = real_mis[ki_amph_idx].mean(axis=0)
    m_kia = markov_mis[ki_amph_idx].mean(axis=0)

    ax_b.plot(lag_seconds, r_kis, color="#1B7837", linewidth=2.5, label="DAT-KI Saline (Real Data)")
    ax_b.plot(lag_seconds, m_kis, color="#1B7837", linewidth=1.8, linestyle="--", label="DAT-KI Saline (1st Markov Null)")
    ax_b.plot(lag_seconds, r_kia, color="#E66101", linewidth=2.5, label="DAT-KI Amphetamine (Real Data)")
    ax_b.plot(lag_seconds, m_kia, color="#E66101", linewidth=1.8, linestyle="--", label="DAT-KI Amphetamine (1st Markov Null)")

    ax_b.set_xscale("log")
    ax_b.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_b.set_ylim([0.0, 1.6])
    ax_b.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_b.set_ylabel("Mutual Information (bits)", fontweight="bold")
    ax_b.set_title("B. DAT-KI Mutants: Saline vs. Amphetamine Timescales", fontweight="bold", fontsize=11)
    ax_b.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", frameon=True, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel C: Excess Behavioral Memory Curves Across All 4 Groups
    # -------------------------------------------------------------------------
    ax_c = axs[1, 0]
    ax_c.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)

    for g_key in ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        idxs = groups[g_key]
        g_ex = excess_mis[idxs]
        mean_ex = g_ex.mean(axis=0)
        sem_ex = g_ex.std(axis=0) / np.sqrt(len(idxs))
        color = group_colors[g_key]
        label = group_labels[g_key]

        ax_c.plot(lag_seconds, mean_ex, color=color, linewidth=2.6, label=label)
        ax_c.fill_between(lag_seconds, mean_ex - sem_ex, mean_ex + sem_ex, color=color, alpha=0.18)

    # Highlight peak of WT Saline
    ex_wts = excess_mis[wt_sal_idx].mean(axis=0)
    peak_wts = int(np.argmax(ex_wts))
    ax_c.scatter([lag_seconds[peak_wts]], [ex_wts[peak_wts]], color="#2B83BA", s=70, zorder=5)
    ax_c.annotate(f"WT Saline Peak\n+{ex_wts[peak_wts]:.3f} bits (6.2s)",
                  xy=(lag_seconds[peak_wts], ex_wts[peak_wts]),
                  xytext=(lag_seconds[peak_wts] * 2.2, ex_wts[peak_wts] * 1.15),
                  arrowprops=dict(facecolor="#2B83BA", arrowstyle="->", lw=1.5),
                  fontweight="bold", fontsize=8.5)

    ax_c.set_xscale("log")
    ax_c.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_c.set_ylim([-0.15, 0.35])
    ax_c.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontweight="bold")
    ax_c.set_ylabel("Excess Mutual Information $\Delta I$ (bits)", fontweight="bold")
    ax_c.set_title("C. Excess Behavioral Memory Across Groups ($\Delta I = I_{\\mathrm{real}} - I_{\\mathrm{markov}}$)", fontweight="bold", fontsize=11)
    ax_c.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper right", frameon=True, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel D: Individual Mouse Memory at the Peak Timescale (~6.2s)
    # -------------------------------------------------------------------------
    ax_d = axs[1, 1]
    peak_idx = 11 # 6.2s
    ax_d.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)

    box_data = []
    box_labels = ["WT\nSaline", "WT\nAmph", "KI\nSaline", "KI\nAmph"]
    box_colors = ["#2B83BA", "#D7191C", "#ABDDA4", "#FDAE61"]

    for g_key in ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]:
        idxs = groups[g_key]
        vals = excess_mis[idxs, peak_idx]
        box_data.append(vals)

    bplot = ax_d.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.55, showmeans=True,
                         meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6))
    for patch, c in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Individual jittered points
    rng = np.random.default_rng(42)
    for pos, vals in enumerate(box_data, 1):
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        ax_d.scatter(np.full_like(vals, pos) + jitter, vals, color="black", s=28, alpha=0.75, zorder=4)

    ax_d.set_ylabel("Excess Memory at $\\tau = 6.2$s (bits)", fontweight="bold")
    ax_d.set_title("D. Per-Mouse Bout Memory at $\\tau = 6.2$ Seconds", fontweight="bold", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.35, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved group timescale figure to: {OUTPUT_FIG}")
    print("=" * 80)

if __name__ == "__main__":
    main()
