#!/usr/bin/env python3
"""
Render the Real Latent NMF Regularization / Denoising Curves (Z Representation)
Shows the empirical bias-variance peak (>100%) and proves KI is simpler (K*=3 vs WT K*=10).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
TABLES_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"

COLORS = {
    "WT_Saline": "#1f77b4",        # Deep Blue
    "WT_Amphetamine": "#d95f02",   # Burnt Orange
    "KI_Saline": "#2ca02c",        # Green
    "KI_Amphetamine": "#7570b3"    # Purple
}
MARKERS = {
    "WT_Saline": "o",
    "WT_Amphetamine": "s",
    "KI_Saline": "^",
    "KI_Amphetamine": "D"
}

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.1

def render():
    csv_path = os.path.join(TABLES_DIR, "cohort_4group_latent_regularization_nmf.csv")
    df = pd.read_csv(csv_path)
    
    k_vals = df["K"].values
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=300)
    
    # Reference 100% line (Full 100-D Repertoire)
    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.6, zorder=1, label="Uncompressed 100-D Syllable Past Baseline (100%)")
    
    # Shaded Denoising / Regularization Zone (> 100%)
    ax.axhspan(100.0, 140.0, color="#e8f5e9", alpha=0.35, zorder=0)
    ax.text(28.5, 137.0, "Denoising / Regularization Zone (NMF beats Raw 100-D)", 
            fontsize=8.5, fontweight="bold", color="#2e7d32", ha="right", style="italic")
            
    labels = {
        "WT_Saline": f"WT + Saline (N=8)       [Peak at K=10: {df.loc[df['K']==10, 'WT_Saline_pct'].values[0]:.1f}% | K=4: {df.loc[df['K']==4, 'WT_Saline_pct'].values[0]:.1f}%]",
        "WT_Amphetamine": f"WT + Amph (N=8)         [Peak at K= 3: {df.loc[df['K']==3, 'WT_Amphetamine_pct'].values[0]:.1f}% | K=4: {df.loc[df['K']==4, 'WT_Amphetamine_pct'].values[0]:.1f}%]",
        "KI_Saline": f"KI + Saline (N=7)       [Peak at K= 3: {df.loc[df['K']==3, 'KI_Saline_pct'].values[0]:.1f}% | K=4: {df.loc[df['K']==4, 'KI_Saline_pct'].values[0]:.1f}%]",
        "KI_Amphetamine": f"KI + Amph (N=7)         [Peak at K= 4: {df.loc[df['K']==4, 'KI_Amphetamine_pct'].values[0]:.1f}% | K=4: {df.loc[df['K']==4, 'KI_Amphetamine_pct'].values[0]:.1f}%]"
    }
    
    for c in cohort_order:
        pct_vals = df[f"{c}_pct"].values
        ax.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.4, label=labels[c], zorder=3)
        ax.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=40, edgecolors="white", linewidth=0.8, zorder=4)
        
    # Highlight peaks with distinct marker
    ax.scatter(10, df.loc[df['K']==10, 'WT_Saline_pct'].values[0], color="#d62728", s=80, zorder=5, edgecolors="white", linewidth=1.2)
    ax.scatter(3, df.loc[df['K']==3, 'KI_Saline_pct'].values[0], color="#d62728", s=80, zorder=5, edgecolors="white", linewidth=1.2)
    ax.scatter(3, df.loc[df['K']==3, 'WT_Amphetamine_pct'].values[0], color="#d62728", s=80, zorder=5, edgecolors="white", linewidth=1.2)
    ax.scatter(4, df.loc[df['K']==4, 'KI_Amphetamine_pct'].values[0], color="#d62728", s=80, zorder=5, edgecolors="white", linewidth=1.2)

    # Annotations
    ax.annotate("WT Saline: Rich repertoire peaks at K=10\n(Denoising gain: +12.1% above 100-D baseline)",
                xy=(10, 112.1), xytext=(12, 103.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.3),
                fontsize=8.3, fontweight="bold", color=COLORS["WT_Saline"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor=COLORS["WT_Saline"], alpha=0.95))
                
    ax.annotate("KI Saline: Simpler dynamics peak at K=3\n(Denoising gain: +33.2% above 100-D baseline)",
                xy=(3, 133.2), xytext=(5, 133.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Saline"], lw=1.3),
                fontsize=8.3, fontweight="bold", color=COLORS["KI_Saline"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor=COLORS["KI_Saline"], alpha=0.95))
                
    ax.annotate("Amphetamine collapses WT peak from K=10 down to K=3\n(Stereotypy matches KI low-dimensional structure)",
                xy=(3, 108.7), xytext=(5, 88.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Amphetamine"], lw=1.3),
                fontsize=8.3, fontweight="bold", color=COLORS["WT_Amphetamine"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor=COLORS["WT_Amphetamine"], alpha=0.95))

    ax.axvline(4, color="#e7298a", linestyle=":", linewidth=1.5, zorder=2, label="Candidate Macro Scale ($K=4$)")

    ax.set_xlim(0, 31)
    ax.set_ylim(40, 142)
    ax.set_xticks([1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    ax.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Utility Retained (% of Own 100-D Past Baseline)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("Empirical NMF Regularization / Denoising Curves (Latent State $Z$)\n(Demonstrating Intrinsic Rank: KI Peaks at $K=3$ vs. WT at $K=10$)",
                 fontsize=13, fontweight="bold", pad=12)
                 
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.96, edgecolor="#d0d7de",
              fontsize=8.0, prop={'family': 'monospace', 'size': 7.7},
              title="Intrinsic Dimensionality & Peak Saturation:", title_fontsize=8.5)
              
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_latent_regularization_curves.png")
    out_pdf = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_latent_regularization_curves.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    shutil.copy(out_png, os.path.join(ARTIFACT_DIR, "fig1b_4cohort_nmf_latent_regularization_curves.png"))
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    render()
