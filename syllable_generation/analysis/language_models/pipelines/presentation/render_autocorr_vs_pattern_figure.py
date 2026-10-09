#!/usr/bin/env python3
"""
Figure: Autocorrelation Decay vs. Predictive Pattern Horizon
Visualizes the fundamental difference between:
  1. Pairwise Autocorrelation Decay R(Delta t): Linear persistence of a single behavior.
  2. Predictive Pattern Horizon W*: The temporal depth of multi-syllable contextual patterns required
     to forecast future transitions h seconds later (extracted from the 2D surface).
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"

COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#d95f02",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#7570b3"
}
COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

def plot_autocorr_vs_pattern():
    # 1. Load 2D surface table to extract predictive pattern profile across W
    df_2d = pd.read_csv(os.path.join(TAB_DIR, "table_2d_memory_timescale_surface.csv"))

    # 2. Compute average pairwise autocorrelation decay curve per cohort
    CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(CACHE_PATH, "rb") as f:
        sb = pickle.load(f)["session_bouts"]
    with open(RAW_PATH, "rb") as f:
        raw_s = pickle.load(f)

    FPS = 80.025
    lags_sec = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0])
    lags_frames = (lags_sec * FPS).astype(int)

    autocorr_curves = {c: [] for c in COHORT_ORDER}

    for i, bouts in enumerate(sb):
        if len(bouts) < 100: continue
        meta = getattr(raw_s[i], "metadata", {})
        c = f"{meta.get('genotype', 'WT')}_{meta.get('drug', 'Saline')}"
        if c not in COHORT_ORDER: continue
        inj_f = meta.get("injection_frame", 0)

        # Build 1-second binned syllable presence
        N = bouts[-1]["end_frame"] + 1
        fs = np.full(N, -1, dtype=np.int32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            fs[sf:ef+1] = b["syllable"]

        post_fs = fs[inj_f:]
        if len(post_fs) < lags_frames[-1] + 1000: continue

        # Sample points to compute probability that S_{t+dt} == S_t
        t_sample = np.arange(0, len(post_fs) - lags_frames[-1], int(5.0 * FPS))
        s_t = post_fs[t_sample]
        valid_t = (s_t >= 0)
        t_sample = t_sample[valid_t]
        s_t = s_t[valid_t]

        ac_vals = []
        for lf in lags_frames:
            s_fut = post_fs[t_sample + lf]
            # Match probability
            match_p = np.mean(s_fut == s_t)
            ac_vals.append(match_p)

        autocorr_curves[c].append(ac_vals)

    # Render Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5), dpi=300)

    # Panel 1: Pairwise Autocorrelation (Single-Behavior Persistence)
    for c in COHORT_ORDER:
        arr = np.array(autocorr_curves[c])
        mean_ac = np.mean(arr, axis=0) * 100.0
        sem_ac = stats.sem(arr, axis=0) * 100.0
        ax1.plot(lags_sec, mean_ac, color=COLORS[c], linewidth=2.4, marker="o", label=c.replace("_", " "))
        ax1.fill_between(lags_sec, mean_ac - sem_ac, mean_ac + sem_ac, color=COLORS[c], alpha=0.18)

    ax1.set_xscale("log")
    ax1.set_xlabel("Time Lag $\\Delta t$ (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Pairwise Syllable Persistence ($P(S_{t+\\Delta t} = S_t)$, %)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A. Pairwise Autocorrelation Persistence\n(How long does an individual ongoing behavior self-sustain?)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=9.0)

    # Panel 2: Predictive Pattern Horizon W* at h=120s (matching Fig 2 forecast horizon)
    sub_h120 = df_2d[df_2d["h_sec"] == 120.0]
    w_vals = sorted(sub_h120["W_sec"].unique())

    for c in COHORT_ORDER:
        sub_c = sub_h120[sub_h120["cohort"] == c].sort_values("W_sec")
        ax2.plot(sub_c["W_sec"], sub_c["delta_history"], color=COLORS[c], linewidth=2.4, marker="s", label=c.replace("_", " "))

    ax2.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="No History Gain (Markov S_t Only)")
    ax2.set_xlabel("History Window Depth $W$ (Seconds of Past Pattern Context)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Incremental Memory Gain $\\Delta_{\\text{hist}}$ (% above $S_t$)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B. Multi-Syllable Predictive Pattern Horizon (Forecast Horizon $h = 120\\text{s}$)\n(How many past seconds of pattern context are integrated to predict future transitions?)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xticks(w_vals)
    ax2.set_xticklabels([f"{int(w)}s" for w in w_vals], fontsize=10, fontweight="bold")
    ax2.set_ylim(-26, 11.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower left", fontsize=9.0)

    # Annotate WT peak vs KI collapse
    ax2.annotate("WT Saline Peaks at W=15-30s\n(Sustained Multi-Minute Context)",
                 xy=(15, 2.3), xytext=(20, 8.5),
                 arrowprops=dict(facecolor=COLORS["WT_Saline"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9, color=COLORS["WT_Saline"])

    ax2.annotate("KI Saline Memory Collapse\n(Context Dilution at Long W)",
                 xy=(60.0, -12.6), xytext=(45, -18.0),
                 arrowprops=dict(facecolor=COLORS["KI_Saline"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9, color=COLORS["KI_Saline"])

    fig.suptitle("Decoupling Pairwise Autocorrelation from Multi-Syllable Pattern Integration",
                 fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_png = os.path.join(OUT_DIR, "fig_autocorr_vs_predictive_pattern_horizon.png")
    fig_pdf = os.path.join(OUT_DIR, "fig_autocorr_vs_predictive_pattern_horizon.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Saved: {fig_png} and {fig_pdf}")
    import shutil
    shutil.copy(fig_png, os.path.join(ARTIFACT_DIR, "fig_autocorr_vs_predictive_pattern_horizon.png"))

if __name__ == "__main__":
    plot_autocorr_vs_pattern()
