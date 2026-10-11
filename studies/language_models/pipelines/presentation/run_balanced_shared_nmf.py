#!/usr/bin/env python3
"""
Module 5: Hierarchically-Balanced Shared NMF Dictionary & Compositional Occupancy
Constructs a common behavioral coordinate space to compare component occupancy across cohorts
without bias or mathematical scaling artifacts:
  1. Hierarchically balanced sampling: Equal number of windows sampled per animal (N=30 mice).
  2. NMF basis normalization: ||H_k||_2 = 1 for each component; Z absorbs the magnitude.
  3. Non-negative projection via NMF.transform() (strict non-negative least squares).
  4. Compositional usage metric: p_ik = sum_t Z_itk / sum_{j,t} Z_itj.
  5. Component usage entropy: H(p_i) = -sum_k p_ik * log2(p_ik) per animal.
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "post_injection_df_cache.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#d95f02",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#7570b3"
}
COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

def run_balanced_shared_nmf():
    print("="*75)
    print("RUNNING MODULE 5: HIERARCHICALLY-BALANCED SHARED NMF (K=16)")
    print("="*75)

    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)

    unique_sessions = df["session_idx"].unique()
    n_animals = len(unique_sessions)
    print(f"Total dataset: {len(df)} windows across N={n_animals} animals.")

    # Hierarchically balanced sampling: sample min_win from each animal
    np.random.seed(42)
    win_per_mouse = min(df["session_idx"].value_counts())
    print(f"Sampling {win_per_mouse} windows per animal (Total = {win_per_mouse * n_animals} balanced windows)...")

    balanced_indices = []
    for s in unique_sessions:
        s_indices = df[df["session_idx"] == s].index.values
        chosen = np.random.choice(s_indices, size=win_per_mouse, replace=False)
        balanced_indices.extend(chosen)

    df_balanced = df.loc[balanced_indices]
    W_SEC = 20.0 # Optimal biological memory window (replaces 68.3s pooled artifact)
    print(f"Extracting past counts at optimal biological memory window W = {W_SEC}s...")
    X_balanced = np.array([w[W_SEC] for w in df_balanced["past_w"]], dtype=np.float32)

    # Fit Shared NMF with K=16
    K = 16
    print(f"Fitting NMF (K={K}) on balanced data matrix {X_balanced.shape}...")
    nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=300)
    nmf.fit(X_balanced)

    # Normalize dictionary rows: ||H_k||_2 = 1
    H = nmf.components_.copy()
    H_norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    H_normalized = H / H_norms
    nmf.components_ = H_normalized

    # Project all windows using strict non-negative transform
    print("Projecting all animals into normalized shared dictionary via non-negative transform...")
    X_all = np.array([w[W_SEC] for w in df["past_w"]], dtype=np.float32)
    Z_all = nmf.transform(X_all) # [N_total, K]

    # Compute compositional occupancy p_ik per animal
    df["Z"] = list(Z_all)
    animal_results = []

    for s in unique_sessions:
        sub = df[df["session_idx"] == s]
        c = sub["cohort"].iloc[0]
        sid = sub["sid"].iloc[0]

        Z_mouse = np.array(sub["Z"].tolist(), dtype=np.float32) # [N_mouse, K]

        # Total activation allocated to each component
        sum_Z_k = np.sum(Z_mouse, axis=0) # [K]
        total_Z = np.sum(sum_Z_k) + 1e-12
        p_ik = sum_Z_k / total_Z # Compositional usage distribution [K]

        # Shannon entropy of component usage (bits)
        p_nz = p_ik[p_ik > 0]
        H_usage = -np.sum(p_nz * np.log2(p_nz))

        row = {
            "session_idx": s,
            "sid": sid,
            "cohort": c,
            "component_usage_entropy_bits": H_usage
        }
        for k in range(K):
            row[f"comp_{k+1}_pct"] = p_ik[k] * 100.0

        animal_results.append(row)

    df_animals = pd.DataFrame(animal_results)
    csv_path = os.path.join(TAB_DIR, "table_balanced_shared_nmf_occupancy.csv")
    df_animals.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Summary of occupancy by cohort
    comp_cols = [f"comp_{k+1}_pct" for k in range(K)]
    cohort_occupancy = df_animals.groupby("cohort")[comp_cols].mean()
    cohort_entropy = df_animals.groupby("cohort")["component_usage_entropy_bits"].agg(["mean", "sem"])

    print("\n--- COMPONENT USAGE ENTROPY BY COHORT (BITS, MAX = 4.0) ---")
    print(cohort_entropy)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.5), dpi=300, gridspec_kw={"width_ratios": [2.2, 1.0]})
    fig.suptitle(f"Hierarchically-Balanced Shared Behavioral Coordinate Space ($K=16$, Optimal History Window $W = {int(W_SEC)}\\text{{s}}$)\n(Strict Animal-Balanced Sampling & Unit-Norm Dictionary Basis Projection)", fontsize=13.5, fontweight="bold", y=0.99)

    # Panel 1: Component Usage Heatmap
    im = ax1.imshow(cohort_occupancy.loc[COHORT_ORDER].values, cmap="magma", aspect="auto")
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f"C{k+1}" for k in range(K)], fontsize=9.5, fontweight="bold")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([c.replace("_", "\n") for c in COHORT_ORDER], fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Shared NMF Component ($k$)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Compositional Component Occupancy ($p_{ik}$, % of Total Latent Activity)\n(Hierarchically-Balanced Shared Dictionary, $\\|H_k\\|_2=1$)", fontsize=11.5, fontweight="bold")

    for i in range(4):
        for j in range(K):
            val = cohort_occupancy.loc[COHORT_ORDER].values[i, j]
            col = "white" if val < 12 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=col, fontsize=8.0, fontweight="bold")

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Mean Occupancy (%)")

    # Panel 2: Component Usage Entropy
    ent_means = [cohort_entropy.loc[c, "mean"] for c in COHORT_ORDER]
    ent_sems = [cohort_entropy.loc[c, "sem"] for c in COHORT_ORDER]

    bars = ax2.bar(range(4), ent_means, yerr=ent_sems, capsize=5,
                   color=[COLORS[c] for c in COHORT_ORDER], edgecolor="black", linewidth=1.2, alpha=0.85)
    for b, m in zip(bars, ent_means):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{m:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels([c.replace("_", "\n") for c in COHORT_ORDER], fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Component Usage Entropy (Bits)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Diversity of Component Usage\n(Entropy $H = -\\sum p_k \\log_2 p_k$)", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, 4.2)
    ax2.axhline(np.log2(K), color="#666666", linestyle="--", label=f"Max Uniform ({np.log2(K):.1f} bits)")
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig_png = os.path.join(OUT_DIR, "fig_balanced_shared_nmf_occupancy.png")
    fig_pdf = os.path.join(OUT_DIR, "fig_balanced_shared_nmf_occupancy.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Saved: {fig_png} and {fig_pdf}")
    import shutil
    shutil.copy(fig_png, os.path.join(ARTIFACT_DIR, "fig_balanced_shared_nmf_occupancy.png"))

if __name__ == "__main__":
    t0 = time.time()
    run_balanced_shared_nmf()
    print(f"Total execution time: {time.time() - t0:.1f}s")
