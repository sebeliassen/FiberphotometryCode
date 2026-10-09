#!/usr/bin/env python3
"""
===============================================================================
Slide 7 Figure: What Do the Components Capture?
(Syllable Loadings, Spatial Arena Distribution, and Mixture Representation)
===============================================================================
A clean, publication-quality presentation slide figure:
  - Header: 1-minute window -> Continuous mixture representation z_t
  - 4 Columns: Components 1 to 4
    - Top subpanel: Top Syllable Loadings (Bar plot, neutral Syllable IDs)
    - Bottom subpanel: Spatial Arena Occupancy (2D Density / Heatmap in Circular Arena)
  - Neutral, objective terminology (no biological overclaims)
===============================================================================
"""

import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# Paths
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/f7fc7de2-e8ac-4e84-bfb7-4ed3043f7346"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

past_counts = []
spatial_features = []

for s_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    frame_spatial = np.zeros((N, 2), dtype=np.float32)
    
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 2:
            frame_spatial[sf : ef + 1, 0] = vec[0]
            frame_spatial[sf : ef + 1, 1] = vec[1]
            
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    cum_spatial = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
    
    for t in range(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        sp_mean = (cum_spatial[t] - cum_spatial[t - W_FRAMES]) / float(W_FRAMES)
        past_counts.append(w_b)
        spatial_features.append(sp_mean)

X_counts = np.array(past_counts, dtype=np.float32)
Spatial = np.array(spatial_features, dtype=np.float32)

# Robust spatial coordinate normalization into unit arena [-1, 1]
valid_sp = np.abs(Spatial[:, 0]) > 1e-4
if np.sum(valid_sp) > 500:
    x_raw = Spatial[valid_sp, 0]
    y_raw = Spatial[valid_sp, 1]
    x_p2, x_p98 = np.percentile(x_raw, 1.5), np.percentile(x_raw, 98.5)
    y_p2, y_p98 = np.percentile(y_raw, 1.5), np.percentile(y_raw, 98.5)
    Spatial[:, 0] = np.clip((Spatial[:, 0] - x_p2) / (x_p98 - x_p2 + 1e-6) * 2.0 - 1.0, -1.15, 1.15)
    Spatial[:, 1] = np.clip((Spatial[:, 1] - y_p2) / (y_p98 - y_p2 + 1e-6) * 2.0 - 1.0, -1.15, 1.15)

# Fit NMF K=4
K = 4
nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
W = nmf.fit_transform(X_counts)
H = nmf.components_
W_probs = W / np.clip(np.sum(W, axis=1, keepdims=True), 1e-8, None)
mean_pop = np.mean(X_counts, axis=0) + 1e-4

# Colors for components
comp_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
comp_names = [f"Component {k+1}" for k in range(K)]
cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']

# Presentation Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.1

fig = plt.figure(figsize=(16.5, 9.8), dpi=300)

# Layout: Title & Subtitle + Top Banner (Mixture representation) + 4 Component Columns (Top Syllables + Spatial Arena)
gs_main = gridspec.GridSpec(3, 1, height_ratios=[0.09, 0.12, 0.79], hspace=0.28,
                            left=0.06, right=0.96, top=0.93, bottom=0.07)

# --- Top Header Block ---
ax_header = fig.add_subplot(gs_main[0])
ax_header.axis('off')
ax_header.text(0.5, 0.82, "What Do the Components Capture?",
               fontsize=19, fontweight='bold', ha='center', va='center', color='#111111')
ax_header.text(0.5, 0.18, "Each ~1-minute behavioral window can be expressed as a mixture of a small number of recurring syllable combinations.",
               fontsize=11.5, style='italic', ha='center', va='center', color='#444444')

# --- Mixture Schematic Banner ---
ax_mix = fig.add_subplot(gs_main[1])
ax_mix.axis('off')

from matplotlib.patches import FancyBboxPatch

# Draw a clean conceptual container box
rect = FancyBboxPatch((0.00, 0.05), 1.00, 0.90, boxstyle="round,pad=0.02",
                      facecolor='#f8f9fa', edgecolor='#d0d7de',
                      linewidth=1.2, transform=ax_mix.transAxes, zorder=1)
ax_mix.add_patch(rect)

ax_mix.text(0.04, 0.50, "Example Behavioral Window ($W_t = 68.3\\text{s}$):",
            fontsize=11.5, fontweight='bold', va='center', transform=ax_mix.transAxes, color='#222222')

ax_mix.text(0.38, 0.50, "$\\mathbf{z}_t = [0.62, \\; 0.21, \\; 0.11, \\; 0.06]$",
            fontsize=12.5, fontweight='bold', va='center', transform=ax_mix.transAxes, color='#111111')

# Draw miniature stacked weight bar
bar_x = 0.65
bar_w = 0.31
bar_h = 0.42
bar_y = 0.29
weights_ex = [0.62, 0.21, 0.11, 0.06]
cum_w = 0.0
for k, (w, col) in enumerate(zip(weights_ex, comp_colors)):
    seg_rect = plt.Rectangle((bar_x + cum_w * bar_w, bar_y), w * bar_w, bar_h,
                             facecolor=col, edgecolor='white', linewidth=1.0,
                             transform=ax_mix.transAxes, zorder=2)
    ax_mix.add_patch(seg_rect)
    if w >= 0.10:
        ax_mix.text(bar_x + (cum_w + w/2) * bar_w, bar_y + bar_h/2, f"{int(w*100)}%",
                    fontsize=9.0, fontweight='bold', color='white', ha='center', va='center',
                    transform=ax_mix.transAxes, zorder=3)
    cum_w += w

ax_mix.text(bar_x + bar_w/2, bar_y - 0.22, "Continuous Mixture Vector (Soft Allocation)",
            fontsize=8.5, color='#555555', ha='center', va='center', transform=ax_mix.transAxes)

# --- Main Grid: 4 Component Columns ---
gs_cols = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_main[2], height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.48)

for k in range(K):
    # --- Top Row: Syllable Loadings ---
    ax_bar = fig.add_subplot(gs_cols[0, k])
    
    enrichment = H[k] / mean_pop
    top_sylls = np.argsort(enrichment)[::-1][:5]
    top_enrich = enrichment[top_sylls]
    
    y_pos = np.arange(len(top_sylls))[::-1]
    bars = ax_bar.barh(y_pos, top_enrich, color=comp_colors[k], alpha=0.88, edgecolor='#333333', linewidth=0.8, height=0.62)
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Syllable {s}" for s in top_sylls], fontsize=9.5, fontweight='bold', color='#222222')
    ax_bar.set_xlabel("Fold Enrichment (vs. Pop Mean)", fontsize=8.5, labelpad=5)
    
    max_enr = max(top_enrich)
    ax_bar.set_xlim(0, max_enr * 1.35)
    for idx, (s_id, val) in enumerate(zip(top_sylls, top_enrich)):
        ax_bar.text(val + max_enr * 0.03, y_pos[idx], f"{val:.1f}×",
                    va='center', fontsize=9.0, fontweight='bold', color='#111111')
        
    ax_bar.set_title(f"{comp_names[k]} Loadings", fontsize=11.5, fontweight='bold', color=comp_colors[k], pad=8)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.35)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    
    # --- Bottom Row: Spatial Arena Occupancy (True Square Arena) ---
    ax_sp = fig.add_subplot(gs_cols[1, k])
    
    # Compute 2D weighted spatial histogram over square arena
    weights_k = W_probs[:, k]
    x_pts = np.clip(Spatial[:, 0], -1.0, 1.0)
    y_pts = np.clip(Spatial[:, 1], -1.0, 1.0)
    
    # 2D Binned density with Gaussian smoothing across square extent
    bins = 40
    H_sp, xedges, yedges = np.histogram2d(x_pts, y_pts, bins=bins, range=[[-1.0, 1.0], [-1.0, 1.0]], weights=weights_k)
    H_smooth = gaussian_filter(H_sp, sigma=1.4)
    
    X_grid, Y_grid = np.meshgrid(0.5*(xedges[:-1]+xedges[1:]), 0.5*(yedges[:-1]+yedges[1:]))
    
    # Render contourf density across full square arena
    cf = ax_sp.contourf(X_grid, Y_grid, H_smooth.T, levels=15, cmap=cmaps[k], alpha=0.92)
    
    # Draw true square arena boundary
    square = plt.Rectangle((-1.0, -1.0), 2.0, 2.0, color='#222222', fill=False, linewidth=1.6, linestyle='-', zorder=3)
    ax_sp.add_patch(square)
    
    ax_sp.set_xlim(-1.08, 1.08)
    ax_sp.set_ylim(-1.08, 1.08)
    ax_sp.set_aspect('equal')
    ax_sp.set_title(f"{comp_names[k]} Arena Density", fontsize=10.5, fontweight='bold', color='#333333', pad=6)
    ax_sp.set_xlabel("Normalized $x$", fontsize=8.5, labelpad=3)
    ax_sp.set_ylabel("Normalized $y$", fontsize=8.5, labelpad=3)
    ax_sp.set_xticks([-1, 0, 1])
    ax_sp.set_yticks([-1, 0, 1])
    ax_sp.tick_params(labelsize=8)
    ax_sp.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()

# Save paths
png_path = os.path.join(OUT_DIR, "fig1_nmf_component_interpretation.png")
pdf_path = os.path.join(OUT_DIR, "fig1_nmf_component_interpretation.pdf")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

# Copy to brain artifact directory for viewing
artifact_png = os.path.join(ARTIFACT_DIR, "fig1_nmf_component_interpretation.png")
shutil.copy(png_path, artifact_png)

print(f"[SUCCESS] Component interpretation figure saved to:\n  - {png_path}\n  - {artifact_png}")
