#!/usr/bin/env python3
"""
===============================================================================
Slide 7 Figure: Predictive Utility vs. Number of NMF Components
(With Dual Axis: Absolute Gain + Normalized Retention %)
===============================================================================
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/f7fc7de2-e8ac-4e84-bfb7-4ed3043f7346"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Data points (Strict leakage-free 5-fold CV by session)
K_vals = np.array([1, 2, 3, 4, 6, 8, 10, 15, 20, 30])
gain_vals = np.array([-2.30, 6.42, 15.28, 17.55, 19.37, 21.97, 24.54, 29.02, 32.12, 33.37])
ref_100d = 29.06

# Professional presentation styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax1 = plt.subplots(figsize=(9.2, 5.4), dpi=300)

# Reference line: Full 100-D Repertoire
ax1.axhline(ref_100d, color='#7f7f7f', linestyle='--', linewidth=1.8, zorder=1, label=f'Full 100-D repertoire ({ref_100d:.1f}%)')

# Plot continuous line connecting points with smooth shading below
ax1.plot(K_vals, gain_vals, color='#1f77b4', linewidth=2.8, zorder=2, label='NMF Components (CV Held-Out)')
ax1.fill_between(K_vals, np.maximum(gain_vals, 0), 0, color='#1f77b4', alpha=0.08, zorder=1)

# Scatter points
ax1.scatter(K_vals, gain_vals, color='#08519c', s=55, zorder=3, edgecolors='white', linewidth=1.0)

# Highlighted annotated points
annot_K = [4, 10, 15]
annot_indices = [np.where(K_vals == k)[0][0] for k in annot_K]
ax1.scatter(K_vals[annot_indices], gain_vals[annot_indices], color='#d62728', s=85, zorder=4, edgecolors='white', linewidth=1.2)

# Text Callout Annotations with both Compression Ratio & Relative Retention %
# 1. At K=4
ax1.annotate("4 components (25×)\n60% of full prediction",
             xy=(4, 17.55),
             xytext=(4, 7.5),
             arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2, shrinkA=0, shrinkB=4),
             fontsize=9.2, fontweight='bold', color='#111111', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#d0d7de', alpha=0.95))

# 2. At K=10
ax1.annotate("10 components (10×)\n84% of full prediction",
             xy=(10, 24.54),
             xytext=(9, 14.5),
             arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2, shrinkA=0, shrinkB=4),
             fontsize=9.2, fontweight='bold', color='#111111', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#d0d7de', alpha=0.95))

# 3. At K=15
ax1.annotate("15 components\n100% of full prediction",
             xy=(15, 29.02),
             xytext=(18.0, 22.0),
             arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2, shrinkA=0, shrinkB=4),
             fontsize=9.2, fontweight='bold', color='#111111', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#d0d7de', alpha=0.95))

# Left Axis Limits & Labels
ax1.set_xlim(-0.5, 31.5)
y_min, y_max = -6.0, 38.0
ax1.set_ylim(y_min, y_max)
ax1.set_xticks([1, 2, 4, 6, 8, 10, 15, 20, 30])

ax1.set_xlabel("Number of NMF Components ($K$)", fontsize=11.5, fontweight='bold', labelpad=9)
ax1.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$)", fontsize=11.5, fontweight='bold', labelpad=9, color='#111111')
ax1.set_title("Predictive Utility vs. Number of NMF Components", fontsize=13.5, fontweight='bold', pad=14)

# Grid
ax1.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#d0d7de', fontsize=9.5)

# --- TWIN AXIS (Right: Relative Retention %) ---
ax2 = ax1.twinx()
ax2.set_ylim(y_min / ref_100d * 100.0, y_max / ref_100d * 100.0)
ax2.set_ylabel("Relative Utility Retained (% of 100-D Repertoire)", fontsize=11.5, fontweight='bold', labelpad=10, color='#333333')

# Ticks on right axis at 0%, 25%, 50%, 75%, 100%, 125%
right_ticks = [0, 25, 50, 75, 100, 125]
ax2.set_yticks(right_ticks)
ax2.set_yticklabels([f"{t}%" for t in right_ticks], fontsize=10)
ax2.spines['right'].set_color('#444444')
ax2.spines['right'].set_linewidth(1.2)

plt.tight_layout()

# Save
png_path = os.path.join(OUT_DIR, "fig1b_nmf_predictive_utility_curve_dual_axis.png")
pdf_path = os.path.join(OUT_DIR, "fig1b_nmf_predictive_utility_curve_dual_axis.pdf")
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

# Copy to artifact dir
artifact_png = os.path.join(ARTIFACT_DIR, "fig1b_nmf_predictive_utility_curve_dual_axis.png")
shutil.copy(png_path, artifact_png)

# Overwrite default figure path too
shutil.copy(png_path, os.path.join(OUT_DIR, "fig1b_nmf_predictive_utility_curve.png"))
shutil.copy(png_path, os.path.join(ARTIFACT_DIR, "fig1b_nmf_predictive_utility_curve.png"))

print(f"[SUCCESS] Dual axis figure saved to:\n  - {png_path}\n  - {artifact_png}")
