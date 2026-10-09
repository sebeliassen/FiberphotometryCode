#!/usr/bin/env python3
"""
===============================================================================
Minimalist Slide 4 Figure: Empirical History Timescale Curve
===============================================================================
A clean, minimal presentation figure:
  - Empirical data points
  - Smooth biexponential fit curve
  - Red dashed vertical line at operational window (68.3s)
  - No arrows, no callout boxes, no error bars
===============================================================================
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Paths
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
DATA_PATH = os.path.join(LM_DIR, "output", "01_horizon_dynamics", "tables", "past_memory_capacity_results.csv")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/518dcf13-db16-4fec-bc0b-b0a6ea54a502"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load empirical data
df = pd.read_csv(DATA_PATH)
w_vals = df["past_window_sec"].values
pmax_vals = df["gain_p5_pmax_mean"].values

# Fit biexponential model
def biexp_func(w, P0, A, tau_rise, tau_decay):
    return P0 + A * (np.exp(-w / tau_decay) - np.exp(-w / tau_rise))

popt, _ = curve_fit(biexp_func, w_vals, pmax_vals, p0=[19.0, 12.0, 17.0, 800.0], maxfev=20000)

w_dense = np.linspace(0.0, 365.0, 500)
p_dense = biexp_func(w_dense, *popt)

# Minimalist styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# Smooth biexponential fit
ax.plot(w_dense, p_dense, color='#1f77b4', linewidth=2.8, label='Biexponential Fit', zorder=2)

# Empirical data points
ax.scatter(w_vals, pmax_vals, color='#08519c', s=45, zorder=3, edgecolors='white', linewidth=0.8, label='Empirical Measurements')

# Red dashed vertical line at operational window (68.3s)
ax.axvline(68.3, color='#d62728', linestyle='--', linewidth=2.0, zorder=2, label='Operational Window (68.3 s)')

# Labels & Limits
ax.set_xlim(-5, 375)
ax.set_ylim(15, 33)

ax.set_xlabel("Past Memory Window $W$ (seconds)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel("Predictive Gain (% $\\Delta D_{\\text{JS}}$)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_title("Predictive Gain vs. Past History Horizon", fontsize=13, fontweight='bold', pad=12)

# Clean minimalist grid & legend
ax.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#e0e0e0', fontsize=9.5)

plt.tight_layout()

# Save
png_path = os.path.join(OUT_DIR, "fig2_timescale_memory_curve_slide4.png")
pdf_path = os.path.join(OUT_DIR, "fig2_timescale_memory_curve_slide4.pdf")
fig_canon_path = os.path.join(OUT_DIR, "fig2_empirical_timescale_memory_curve.png")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.savefig(fig_canon_path, dpi=300)
plt.close()

# Copy to artifact dir for instant view
artifact_png = os.path.join(ARTIFACT_DIR, "fig2_timescale_memory_curve_slide4.png")
shutil.copy(png_path, artifact_png)

print(f"[SUCCESS] Minimalist Slide 4 figure saved to:\n  - {png_path}\n  - {artifact_png}")
