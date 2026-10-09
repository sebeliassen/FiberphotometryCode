#!/usr/bin/env python3
"""
===============================================================================
Render Publication-Quality Percentage-Wise NMF Retention Curves
(fig1b_4cohort_nmf_relative_retention_curves.png & Dual-Panel)
===============================================================================
"""

import os
import shutil

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(LM_DIR, "output", "presentation", "tables", "cohort_4group_dense_nmf_retention.csv")

# Artifact dirs
ARTIFACT_DIR_1 = "/home/fsp585/.gemini/antigravity-ide/brain/acfd96c6-639a-4c71-be4a-46cc2b7a0386"
ARTIFACT_DIR_CURRENT = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
for d in [ARTIFACT_DIR_1, ARTIFACT_DIR_CURRENT]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    "WT_Saline": "#1f77b4",        # Deep Blue
    "WT_Amphetamine": "#d95f02",   # Burnt Orange
    "KI_Saline": "#2ca02c",        # Green
    "KI_Amphetamine": "#7570b3"    # Purple
}
LABELS = {
    "WT_Saline": "WT + Saline (N=8)",
    "WT_Amphetamine": "WT + Amphetamine (N=8)",
    "KI_Saline": "KI + Saline (N=7)",
    "KI_Amphetamine": "KI + Amphetamine (N=7)"
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

def render_relative_retention_figure():
    df = pd.read_csv(CSV_PATH)
    k_vals = df["K"].values
    cohort_order = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

    # -------------------------------------------------------------------------
    # 1. STANDALONE FIGURE: Relative Predictive Capacity Retained (% of 100-D Ceiling)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=300)

    # Reference lines
    ax.axhline(100.0, color="#555555", linestyle=":", linewidth=1.5, zorder=1, label="Uncompressed 100-D Past Ceiling (100%)")
    ax.axhline(90.0, color="#888888", linestyle="--", linewidth=1.1, zorder=1, label="90% Informational Saturation Threshold")

    # Dynamic labels with saturation statistics
    enhanced_labels = {
        "WT_Saline": f"WT + Saline (N=8)       [K₉₀: >30 | K=4: {df.loc[df['K']==4, 'WT_Saline_pct_100d'].values[0]:.1f}%]",
        "WT_Amphetamine": f"WT + Amph (N=8)         [K₉₀:  14 | K=4: {df.loc[df['K']==4, 'WT_Amphetamine_pct_100d'].values[0]:.1f}%]",
        "KI_Saline": f"KI + Saline (N=7)       [K₉₀:   2 | K=4: {df.loc[df['K']==4, 'KI_Saline_pct_100d'].values[0]:.1f}%]",
        "KI_Amphetamine": f"KI + Amph (N=7)         [K₉₀:   1 | K=4: {df.loc[df['K']==4, 'KI_Amphetamine_pct_100d'].values[0]:.1f}%]"
    }

    for c in cohort_order:
        pct_vals = df[f"{c}_pct_100d"].values
        ax.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.5, label=enhanced_labels[c], zorder=3)
        ax.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=42, edgecolors="white", linewidth=0.8, zorder=4)

    # Candidate Macro Scale line at K=4
    ax.axvline(4, color="#d95f02", linestyle="--", linewidth=1.6, zorder=2, label="Candidate Macro Scale ($K = 4$)")

    # Clean Callout Annotations at K=4
    ax.annotate("KI Amph: 98.5% at K=4\n(Saturated at K=1: 91.0%)",
                xy=(4, 98.51), xytext=(2.0, 102.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Amphetamine"], lw=1.2),
                fontsize=8.4, fontweight="bold", color=COLORS["KI_Amphetamine"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#f3e5f5", edgecolor=COLORS["KI_Amphetamine"], alpha=0.92))

    ax.annotate("KI Saline: 95.2% at K=4\n(Rapid saturation: K₉₀=2)",
                xy=(4, 95.20), xytext=(7.0, 96.8),
                arrowprops=dict(arrowstyle="->", color=COLORS["KI_Saline"], lw=1.2),
                fontsize=8.4, fontweight="bold", color=COLORS["KI_Saline"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#e8f5e9", edgecolor=COLORS["KI_Saline"], alpha=0.92))

    ax.annotate("WT Saline: 82.6% at K=4\n(Gradual accumulation: K₉₀>30)",
                xy=(4, 82.57), xytext=(5.5, 77.2),
                arrowprops=dict(arrowstyle="->", color=COLORS["WT_Saline"], lw=1.2),
                fontsize=8.4, fontweight="bold", color=COLORS["WT_Saline"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#e1f5fe", edgecolor=COLORS["WT_Saline"], alpha=0.92))

    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(74, 106)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    ax.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Retention (% of Full 100-D Past Ceiling)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("NMF Predictive Retention vs. Number of Components ($K$)\n(Normalized % of Full History Repertoire Across 4 Cohorts)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d7de",
              fontsize=8.5, prop={'family': 'monospace', 'size': 8.2},
              title="Dimensional Saturation Hierarchy:", title_fontsize=9.0)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_relative_retention_curves.png")
    pdf_path = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_relative_retention_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    shutil.copy(png_path, os.path.join(ARTIFACT_DIR_1, "fig1b_4cohort_nmf_relative_retention_curves.png"))
    shutil.copy(png_path, os.path.join(ARTIFACT_DIR_CURRENT, "fig1b_4cohort_nmf_relative_retention_curves.png"))
    print(f"  [SAVED] {png_path}")

    # -------------------------------------------------------------------------
    # 2. DUAL-PANEL COMPREHENSIVE FIGURE: (A) Absolute Gain, (B) Relative Retention %
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300)

    # Panel A: Absolute Predictive Gain (K=1..30)
    for c in cohort_order:
        g_vals = df[c].values
        ax1.plot(k_vals, g_vals, color=COLORS[c], linewidth=2.3, label=LABELS[c], zorder=3)
        ax1.scatter(k_vals, g_vals, color=COLORS[c], marker=MARKERS[c], s=35, edgecolors="white", linewidth=0.7, zorder=4)

    ax1.axvline(4, color="#d95f02", linestyle="--", linewidth=1.5, zorder=2, label="Candidate Macro Scale ($K = 4$)")
    ax1.set_xlim(0.5, 30.5)
    ax1.set_ylim(8.0, 23.0)
    ax1.set_xticks([1, 2, 4, 6, 8, 10, 15, 20, 25, 30])
    ax1.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$ vs. Prior)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax1.set_title("A. Absolute Predictive Utility across Dimensions ($K=1..30$)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d0d7de", fontsize=8.5)

    # Panel B: Relative Retention %
    ax2.axhline(100.0, color="#555555", linestyle=":", linewidth=1.4, zorder=1, label="100-D Full Ceiling (100%)")
    ax2.axhline(90.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1, label="90% Threshold")

    for c in cohort_order:
        pct_vals = df[f"{c}_pct_100d"].values
        ax2.plot(k_vals, pct_vals, color=COLORS[c], linewidth=2.3, label=enhanced_labels[c], zorder=3)
        ax2.scatter(k_vals, pct_vals, color=COLORS[c], marker=MARKERS[c], s=35, edgecolors="white", linewidth=0.7, zorder=4)

    ax2.axvline(4, color="#d95f02", linestyle="--", linewidth=1.5, zorder=2, label="Macro Scale ($K=4$)")
    ax2.set_xlim(0.5, 30.5)
    ax2.set_ylim(74, 106)
    ax2.set_xticks([1, 2, 4, 6, 8, 10, 15, 20, 25, 30])
    ax2.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Relative Information Retained (% of 100-D Ceiling)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax2.set_title("B. Dimensional Saturation (% of Repertoire Ceiling)", fontsize=12.5, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d0d7de",
               fontsize=7.8, prop={'family': 'monospace', 'size': 7.6},
               title="Saturation Hierarchy:", title_fontsize=8.5)

    plt.tight_layout()
    dual_png = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_dual_panel_retention.png")
    dual_pdf = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_dual_panel_retention.pdf")
    plt.savefig(dual_png, dpi=300)
    plt.savefig(dual_pdf)
    plt.close()

    shutil.copy(dual_png, os.path.join(ARTIFACT_DIR_1, "fig1b_4cohort_nmf_dual_panel_retention.png"))
    shutil.copy(dual_png, os.path.join(ARTIFACT_DIR_CURRENT, "fig1b_4cohort_nmf_dual_panel_retention.png"))
    print(f"  [SAVED] {dual_png}")

    # Also update the canonical fig1b_4cohort_nmf_retention_curves.png with dense K=1..30 points
    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=300)
    for c in cohort_order:
        g_vals = df[c].values
        ax.plot(k_vals, g_vals, color=COLORS[c], linewidth=2.4, label=LABELS[c], zorder=3)
        ax.scatter(k_vals, g_vals, color=COLORS[c], marker=MARKERS[c], s=42, edgecolors="white", linewidth=0.8, zorder=4)

    ax.axvline(4, color="#d95f02", linestyle="--", linewidth=1.5, zorder=2, label="Candidate Macro Scale ($K = 4$)")
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(8, 23.0)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 15, 20, 25, 30])
    ax.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$ vs. Reference Prior)", fontsize=11.5, fontweight="bold", labelpad=9)
    ax.set_title("NMF Predictive Retention vs. Number of Components ($K$)\n(Dense K=1..30 Leak-Free Evaluation across 4 Cohorts)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d7de", fontsize=9.2)

    plt.tight_layout()
    dense_canon_png = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_retention_curves.png")
    dense_canon_pdf = os.path.join(OUT_DIR, "fig1b_4cohort_nmf_retention_curves.pdf")
    plt.savefig(dense_canon_png, dpi=300)
    plt.savefig(dense_canon_pdf)
    plt.close()
    shutil.copy(dense_canon_png, os.path.join(ARTIFACT_DIR_1, "fig1b_4cohort_nmf_retention_curves.png"))
    shutil.copy(dense_canon_png, os.path.join(ARTIFACT_DIR_CURRENT, "fig1b_4cohort_nmf_retention_curves.png"))
    print(f"  [UPDATED CANONICAL] {dense_canon_png}")

if __name__ == "__main__":
    render_relative_retention_figure()
