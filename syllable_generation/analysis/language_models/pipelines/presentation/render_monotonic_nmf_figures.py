#!/usr/bin/env python3
"""
Render Monotonic NMF Retention Curves and Diagnostic Investigation
Demonstrates why raw Z plateaued at 89% and why X_hat = Z @ H reaches 100% monotonically.
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
    clean_csv = os.path.join(TABLES_DIR, "cohort_4group_reconstructed_nmf_retention.csv")
    ext_csv = os.path.join(TABLES_DIR, "cohort_4group_extended_nmf_retention_k100.csv")
    
    df_clean = pd.read_csv(clean_csv)
    df_ext = pd.read_csv(ext_csv)
    
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    k_clean = df_clean["K"].values
    
    # -------------------------------------------------------------------------
    # 1. PRIMARY PUBLICATION FIGURE: Clean Strictly Monotonic Retention to 100%
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=300)
    
    ax.axhline(100.0, color="#555555", linestyle=":", linewidth=1.5, zorder=1, label="Full 100-D Repertoire Ceiling (100%)")
    ax.axhline(90.0, color="#888888", linestyle="--", linewidth=1.1, zorder=1, label="90% Informational Saturation Threshold")
    ax.axhline(95.0, color="#bbbbbb", linestyle="-.", linewidth=0.9, zorder=1, label="95% Informational Saturation Threshold")
    
    labels_clean = {
        "WT_Saline": f"WT + Saline (N=8)       [K₉₀:   5 | K₉₅:   9 | K=4: {df_clean.loc[df_clean['K']==4, 'WT_Saline_pct'].values[0]:.1f}% | K=89: 100.0%]",
        "WT_Amphetamine": f"WT + Amph (N=8)         [K₉₀:   7 | K₉₅:  10 | K=4: {df_clean.loc[df_clean['K']==4, 'WT_Amphetamine_pct'].values[0]:.1f}% | K=89: 100.0%]",
        "KI_Saline": f"KI + Saline (N=7)       [K₉₀:   4 | K₉₅:  25 | K=4: {df_clean.loc[df_clean['K']==4, 'KI_Saline_pct'].values[0]:.1f}% | K=89:  99.8%]",
        "KI_Amphetamine": f"KI + Amph (N=7)         [K₉₀:   7 | K₉₅:  28 | K=4: {df_clean.loc[df_clean['K']==4, 'KI_Amphetamine_pct'].values[0]:.1f}% | K=89:  99.7%]"
    }
    
    for c in cohort_order:
        pct_vals = df_clean[f"{c}_pct"].values
        ax.plot(k_clean, pct_vals, color=COLORS[c], linewidth=2.4, label=labels_clean[c], zorder=3)
        ax.scatter(k_clean, pct_vals, color=COLORS[c], marker=MARKERS[c], s=35, edgecolors="white", linewidth=0.7, zorder=4)
        
    ax.axvline(4, color="#d95f02", linestyle="--", linewidth=1.6, zorder=2, label="Candidate Macro Scale ($K = 4$)")
    ax.axvline(89, color="#666666", linestyle=":", linewidth=1.4, zorder=2, label="WT Repertoire Rank Ceiling ($K = 89$)")
    
    # Annotations
    ax.annotate("All 4 cohorts converge monotonically to 100%\nat WT rank ceiling (K = 89)",
                xy=(89, 100.0), xytext=(52, 92.5),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.3),
                fontsize=8.5, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#444444", alpha=0.96))
                
    ax.annotate("KI Saline: 90.9% at K=4\n(Highest macro efficiency)",
                xy=(4, 90.9), xytext=(7, 85.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Saline"], lw=1.2),
                fontsize=8.3, fontweight="bold", color=COLORS["KI_Saline"],
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#e8f5e9", edgecolor=COLORS["KI_Saline"], alpha=0.92))
                
    ax.annotate("WT Saline: 89.4% at K=4\n(Hits 95% at K=9)",
                xy=(4, 89.4), xytext=(12, 80.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.2),
                fontsize=8.3, fontweight="bold", color=COLORS["WT_Saline"],
                bbox=dict(boxstyle="round,pad=0.26", facecolor="#e1f5fe", edgecolor=COLORS["WT_Saline"], alpha=0.92))

    ax.set_xlim(-1, 92)
    ax.set_ylim(65, 104)
    ax.set_xticks([1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 89])
    ax.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Retention (% of Full 100-D Past Ceiling)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("NMF Predictive Retention to Full Empirical Rank ($K = 1 \\dots 89$)\n(Reconstructed Syllable Past $\\hat{X} = Z \\cdot H$ Relative to Uncompressed Ceiling)",
                 fontsize=13, fontweight="bold", pad=12)
                 
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.96, edgecolor="#d0d7de",
              fontsize=8.1, prop={'family': 'monospace', 'size': 7.8},
              title="Monotonic Saturation Hierarchy:", title_fontsize=8.6)
              
    plt.tight_layout()
    png_mono = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_reconstructed_retention_curves.png")
    pdf_mono = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_reconstructed_retention_curves.pdf")
    plt.savefig(png_mono, dpi=300)
    plt.savefig(pdf_mono)
    plt.close()
    shutil.copy(png_mono, os.path.join(ARTIFACT_DIR, "fig1b_4cohort_nmf_reconstructed_retention_curves.png"))
    print(f"Saved: {png_mono}")

    # -------------------------------------------------------------------------
    # 2. DIAGNOSTIC FIGURE: Raw Latent Z (with Artifacts) vs Reconstructed X_hat
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 6.4), dpi=300)
    
    k_ext = df_ext["K"].values
    
    # Panel A: Raw Latent Z
    ax1.axhline(100.0, color="#555555", linestyle=":", linewidth=1.4, zorder=1)
    ax1.axhline(90.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
    for c in cohort_order:
        ax1.plot(k_ext, df_ext[f"{c}_pct_100d"].values, color=COLORS[c], linewidth=2.2, label=c.replace('_', ' '), zorder=3)
        ax1.scatter(k_ext, df_ext[f"{c}_pct_100d"].values, color=COLORS[c], marker=MARKERS[c], s=28, edgecolors="white", linewidth=0.6, zorder=4)
    ax1.axvline(89, color="#666666", linestyle=":", linewidth=1.3, zorder=2)
    ax1.set_xlim(-1, 103)
    ax1.set_ylim(68, 107)
    ax1.set_xticks([1, 10, 20, 30, 50, 70, 89, 100])
    ax1.set_xlabel("Number of Components ($K$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predictive Retention (% of Ceiling)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Raw Latent Coordinates $Z$\n(Artifacts: Ridge Shrinkage Plateau & K>89 Ill-Conditioning Cliff)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower left", fontsize=8.2)
    
    ax1.annotate("WT Saline plateaus at 89.2%\n(Fixed $\\lambda=500$ over-regularizes Z)",
                 xy=(30, 88.0), xytext=(22, 78.0),
                 arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.2),
                 fontsize=8.0, fontweight="bold", color=COLORS["WT_Saline"],
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#e1f5fe", edgecolor=COLORS["WT_Saline"], alpha=0.92))
                 
    ax1.annotate("Collinear Cliff at K > 89\n(Only 89 non-zero syllables exist!)",
                 xy=(90, 72.0), xytext=(55, 72.0),
                 arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
                 fontsize=8.0, fontweight="bold", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9c4", edgecolor="#fbc02d", alpha=0.95))

    # Panel B: Reconstructed Syllable Past X_hat = Z @ H
    ax2.axhline(100.0, color="#555555", linestyle=":", linewidth=1.4, zorder=1, label="100-D Ceiling")
    ax2.axhline(90.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1, label="90% Threshold")
    for c in cohort_order:
        ax2.plot(k_clean, df_clean[f"{c}_pct"].values, color=COLORS[c], linewidth=2.2, label=c.replace('_', ' '), zorder=3)
        ax2.scatter(k_clean, df_clean[f"{c}_pct"].values, color=COLORS[c], marker=MARKERS[c], s=28, edgecolors="white", linewidth=0.6, zorder=4)
    ax2.axvline(89, color="#666666", linestyle=":", linewidth=1.3, zorder=2, label="Rank Ceiling (K=89)")
    ax2.set_xlim(-1, 92)
    ax2.set_ylim(68, 103)
    ax2.set_xticks([1, 10, 20, 30, 50, 70, 89])
    ax2.set_xlabel("Number of Components ($K$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predictive Retention (% of Ceiling)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Reconstructed Past $\\hat{X} = Z \\cdot H$\n(Resolved: Strictly Monotonic Convergence to 100.0% at Rank 89)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.2)

    plt.tight_layout()
    diag_png = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_diagnostic_comparison.png")
    diag_pdf = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_diagnostic_comparison.pdf")
    plt.savefig(diag_png, dpi=300)
    plt.savefig(diag_pdf)
    plt.close()
    shutil.copy(diag_png, os.path.join(ARTIFACT_DIR, "fig1b_4cohort_nmf_diagnostic_comparison.png"))
    print(f"Saved: {diag_png}")

if __name__ == "__main__":
    render()
