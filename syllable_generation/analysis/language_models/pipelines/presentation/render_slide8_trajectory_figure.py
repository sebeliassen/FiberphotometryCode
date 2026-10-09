#!/usr/bin/env python3
"""
===============================================================================
Slide 8 Figure: Continuous Behavioral Trajectories Over Time
(Representative Session and Group Average across WT / Saline animals)
===============================================================================
Renders:
  Panel A: Group Average Timecourse (Mean ± SEM across WT Saline animals)
  Panel B: Single Representative Session Continuous Mixture Trajectory
  Panel C: 2D Projection of Continuous Behavioral Trajectory (State Manifold)
===============================================================================
"""

import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/f7fc7de2-e8ac-4e84-bfb7-4ed3043f7346"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. Load Bout Cache (instant load)
cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

# Exact WT Saline session metadata
wt_saline_sessions = [
    {"index": 1, "sid": "2025-04-28T09_31_08_bottom_right", "entry_f": 66135, "inj_f": 162135},
    {"index": 2, "sid": "2025-04-28T09_31_08_top_left", "entry_f": 28222, "inj_f": 124222},
    {"index": 6, "sid": "2025-04-28T12_46_50_bottom_left", "entry_f": 12174, "inj_f": 108174},
    {"index": 7, "sid": "2025-04-28T14_17_17_bottom_left", "entry_f": 44073, "inj_f": 140073},
    {"index": 8, "sid": "2025-04-28T14_17_17_bottom_right", "entry_f": 57610, "inj_f": 153610},
    {"index": 10, "sid": "2025-04-28T15_54_32_bottom_left", "entry_f": 29275, "inj_f": 125275},
    {"index": 12, "sid": "2025-04-28T15_54_32_top_left", "entry_f": 14044, "inj_f": 110044},
    {"index": 14, "sid": "2025-04-28T17_28_17_top_left", "entry_f": 11161, "inj_f": 107161}
]

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

# 2. Fit NMF K=4 across all sessions
all_past_counts = []
for s_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
            
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    
    for t in range(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        all_past_counts.append(w_b)

X_all = np.array(all_past_counts, dtype=np.float32)
K = 4
nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
nmf.fit(X_all)

# 3. Extract timecourses for WT Saline sessions up to 45 min (standard trial: 20 min baseline + 25 min post-saline)
MAX_T_MIN = 45.0
common_time_min = np.linspace(1.5, MAX_T_MIN, 300)  # Common 45-min grid
session_timecourses = []

for item in wt_saline_sessions:
    s_idx = item["index"]
    entry_f = item["entry_f"]
    inj_f = item["inj_f"]
    bouts = session_bouts[s_idx]
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
            
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    
    sess_w_counts = []
    sess_times = []
    
    start_t = entry_f + W_FRAMES
    max_eval_frame = int(entry_f + (MAX_T_MIN + 1.0) * 60.0 * FPS)
    end_t = min(N, max_eval_frame)
    
    for t in range(start_t, end_t, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        sess_w_counts.append(w_b)
        sess_times.append((t - entry_f) / FPS / 60.0)  # minutes from entry
        
    sess_times = np.array(sess_times)
    if len(sess_times) > 20:
        W_sess = nmf.transform(np.array(sess_w_counts, dtype=np.float32))
        W_probs = W_sess / np.clip(np.sum(W_sess, axis=1, keepdims=True), 1e-8, None)
        
        # Smooth slightly over adjacent 10s windows for clean presentation
        from scipy.ndimage import uniform_filter1d
        W_probs_smooth = uniform_filter1d(W_probs, size=3, axis=0)
        W_probs_smooth = W_probs_smooth / np.clip(np.sum(W_probs_smooth, axis=1, keepdims=True), 1e-8, None)
        
        # Interpolate onto common time grid
        interp_probs = np.zeros((len(common_time_min), K))
        for k in range(K):
            f_interp = interp1d(sess_times, W_probs_smooth[:, k], bounds_error=False, fill_value="extrapolate")
            interp_probs[:, k] = np.clip(f_interp(common_time_min), 0, 1)
        interp_probs = interp_probs / np.clip(np.sum(interp_probs, axis=1, keepdims=True), 1e-8, None)
        
        session_timecourses.append({
            "sid": item["sid"],
            "raw_times": sess_times,
            "raw_probs": W_probs_smooth,
            "interp_probs": interp_probs,
            "inj_min": (inj_f - entry_f) / FPS / 60.0
        })

print(f"Extracted {len(session_timecourses)} WT Saline timecourses.")

# Stack across sessions for group statistics
all_interp = np.stack([s["interp_probs"] for s in session_timecourses], axis=0) # [n_sessions, n_times, K]
mean_probs = np.nanmean(all_interp, axis=0) # [n_times, K]
sem_probs = np.nanstd(all_interp, axis=0) / np.sqrt(len(session_timecourses))

# 4. Rendering Presentation Figure
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.1

comp_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
comp_names = [f"Component {k+1}" for k in range(K)]

fig = plt.figure(figsize=(15.8, 9.6), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.26,
                       left=0.07, right=0.96, top=0.90, bottom=0.08)

# Figure Header
fig.suptitle("Continuous Behavioral Trajectories Over Time (Wildtype / Saline Control)",
             fontsize=17, fontweight='bold', y=0.96, color='#111111')

# --- Panel A: Group Average Timecourse (Mean ± SEM) ---
ax_grp = fig.add_subplot(gs[0, 0])
for k in range(K):
    ax_grp.plot(common_time_min, mean_probs[:, k], color=comp_colors[k],
                linewidth=2.5, label=comp_names[k])
    ax_grp.fill_between(common_time_min,
                        np.maximum(0, mean_probs[:, k] - sem_probs[:, k]),
                        np.minimum(1.0, mean_probs[:, k] + sem_probs[:, k]),
                        color=comp_colors[k], alpha=0.18)

ax_grp.axvline(20.0, color='#444444', linestyle='--', linewidth=1.3, alpha=0.8)
ax_grp.text(20.5, 0.78, "Saline Injection\n(t = 20 min)", fontsize=9, color='#444444', va='top')

ax_grp.set_title("A. Cohort Average Timecourse (n = 8 WT/Saline)", fontsize=12.5, fontweight='bold', pad=10)
ax_grp.set_xlabel("Time from Arena Entry (minutes)", fontsize=10.5)
ax_grp.set_ylabel("Mixture Weight ($\mathbf{z}_t$)", fontsize=10.5)
ax_grp.set_xlim(1.5, MAX_T_MIN)
ax_grp.set_ylim(0, 0.85)
ax_grp.grid(True, linestyle=':', alpha=0.4)
ax_grp.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
ax_grp.spines['top'].set_visible(False)
ax_grp.spines['right'].set_visible(False)

# --- Panel B: Single Representative Session Timecourse ---
rep_sess = session_timecourses[0]
valid_mask = (rep_sess["raw_times"] >= 1.5) & (rep_sess["raw_times"] <= MAX_T_MIN)
t_rep = rep_sess["raw_times"][valid_mask]
probs_rep = np.nan_to_num(rep_sess["raw_probs"][valid_mask], nan=0.0)

ax_rep = fig.add_subplot(gs[0, 1])

for k in range(K):
    ax_rep.plot(t_rep, probs_rep[:, k],
                color=comp_colors[k], linewidth=2.2, label=comp_names[k])

inj_m = rep_sess["inj_min"]
ax_rep.axvline(inj_m, color='#444444', linestyle='--', linewidth=1.3, alpha=0.8)
ax_rep.text(inj_m + 0.5, 0.95, f"Saline Injection\n(t = {inj_m:.1f} min)", fontsize=9, color='#444444', va='top')

ax_rep.set_title("B. Single Representative Session Trajectory", fontsize=12.5, fontweight='bold', pad=10)
ax_rep.set_xlabel("Time from Arena Entry (minutes)", fontsize=10.5)
ax_rep.set_ylabel("Mixture Weight ($\mathbf{z}_t$)", fontsize=10.5)
ax_rep.set_xlim(1.5, MAX_T_MIN)
ax_rep.set_ylim(0, 1.05)
ax_rep.grid(True, linestyle=':', alpha=0.4)
ax_rep.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
ax_rep.spines['top'].set_visible(False)
ax_rep.spines['right'].set_visible(False)

# --- Panel C: Equal-AUC Normalized Relative Prominence ---
# Normalize each component so each has equal total area under curve (AUC = 1.0)
ax_norm = fig.add_subplot(gs[1, 0])

auc_norm_probs = np.zeros_like(probs_rep)
for k in range(K):
    total_auc = np.trapz(probs_rep[:, k], t_rep)
    if total_auc > 1e-6:
        auc_norm_probs[:, k] = probs_rep[:, k] / total_auc
    else:
        auc_norm_probs[:, k] = probs_rep[:, k]

# Scale for visual comparability so average height is 1.0
auc_norm_scaled = auc_norm_probs * (MAX_T_MIN - 1.5)

for k in range(K):
    ax_norm.plot(t_rep, auc_norm_scaled[:, k],
                 color=comp_colors[k], linewidth=2.3, label=comp_names[k])
    ax_norm.fill_between(t_rep, 0, auc_norm_scaled[:, k],
                         color=comp_colors[k], alpha=0.12)

ax_norm.axvline(inj_m, color='#444444', linestyle='--', linewidth=1.3, alpha=0.8)
ax_norm.axhline(1.0, color='#888888', linestyle=':', linewidth=1.1, alpha=0.7)
ax_norm.text(MAX_T_MIN - 0.5, 1.05, "Session Average (1.0×)", fontsize=8.5, color='#666666', ha='right')

ax_norm.set_title("C. Relative State Prominence (Equal AUC Normalization)", fontsize=12.5, fontweight='bold', pad=10)
ax_norm.set_xlabel("Time from Arena Entry (minutes)", fontsize=10.5)
ax_norm.set_ylabel("Relative Prominence (vs. Component Mean)", fontsize=10.5)
ax_norm.set_xlim(1.5, MAX_T_MIN)
ax_norm.set_ylim(0, np.percentile(auc_norm_scaled, 99.5) * 1.25)
ax_norm.grid(True, linestyle=':', alpha=0.4)
ax_norm.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
ax_norm.spines['top'].set_visible(False)
ax_norm.spines['right'].set_visible(False)

# --- Panel D: Smooth Continuous Trajectory in Behavioral Space ---
ax_man = fig.add_subplot(gs[1, 1])

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection

pca_2d = PCA(n_components=2, random_state=42)
probs_clean = np.nan_to_num(probs_rep, nan=0.25)
z_proj_raw = pca_2d.fit_transform(probs_clean)

# Apply Gaussian filter for a smooth, readable continuous flow
z_proj_smooth = gaussian_filter1d(z_proj_raw, sigma=4.0, axis=0)

# Create multicolored line segments colored by time
points = z_proj_smooth.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(t_rep[0], t_rep[-1])
lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3.5, alpha=0.95, zorder=3)
lc.set_array(t_rep[:-1])
line = ax_man.add_collection(lc)

# Add periodic directional markers (every 5 minutes)
sample_interval = 25  # ~2 minutes
for idx in range(10, len(t_rep) - 10, sample_interval):
    dx = z_proj_smooth[idx+1, 0] - z_proj_smooth[idx-1, 0]
    dy = z_proj_smooth[idx+1, 1] - z_proj_smooth[idx-1, 1]
    dist = np.hypot(dx, dy)
    if dist > 1e-4:
        ax_man.annotate('', xy=(z_proj_smooth[idx, 0] + dx/dist*0.02, z_proj_smooth[idx, 1] + dy/dist*0.02),
                        xytext=(z_proj_smooth[idx, 0], z_proj_smooth[idx, 1]),
                        arrowprops=dict(arrowstyle="->", color='#222222', lw=1.5), zorder=4)

# Start and End prominent markers
ax_man.scatter(z_proj_smooth[0, 0], z_proj_smooth[0, 1], color='#00e676', edgecolor='black',
               s=160, marker='o', label='Start (0 min)', zorder=5)
ax_man.scatter(z_proj_smooth[-1, 0], z_proj_smooth[-1, 1], color='#ff1744', edgecolor='black',
               s=180, marker='X', label='End (45 min)', zorder=5)

# Injection waypoint marker
inj_idx = np.argmin(np.abs(t_rep - inj_m))
ax_man.scatter(z_proj_smooth[inj_idx, 0], z_proj_smooth[inj_idx, 1], color='white', edgecolor='#222222',
               linewidth=1.8, s=120, marker='D', label='Injection (20 min)', zorder=5)

cbar = plt.colorbar(line, ax=ax_man, pad=0.03, fraction=0.045)
cbar.set_label("Time in Session (min)", fontsize=9.5)

ax_man.set_xlim(np.min(z_proj_smooth[:, 0]) - 0.08, np.max(z_proj_smooth[:, 0]) + 0.08)
ax_man.set_ylim(np.min(z_proj_smooth[:, 1]) - 0.08, np.max(z_proj_smooth[:, 1]) + 0.08)

ax_man.set_title("D. Smooth Trajectory in Behavioral Space (PCA Manifold)", fontsize=12.5, fontweight='bold', pad=10)
ax_man.set_xlabel("Behavioral Component Dimension 1", fontsize=10.5)
ax_man.set_ylabel("Behavioral Component Dimension 2", fontsize=10.5)
ax_man.grid(True, linestyle=':', alpha=0.4)
ax_man.legend(loc='lower left', framealpha=0.92, fontsize=8.8)
ax_man.spines['top'].set_visible(False)
ax_man.spines['right'].set_visible(False)

# Save figure
png_path = os.path.join(OUT_DIR, "fig1d_continuous_trajectories_wt_saline.png")
pdf_path = os.path.join(OUT_DIR, "fig1d_continuous_trajectories_wt_saline.pdf")
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

# Copy to brain artifact directory
artifact_png = os.path.join(ARTIFACT_DIR, "fig1d_continuous_trajectories_wt_saline.png")
shutil.copy(png_path, artifact_png)

print(f"[SUCCESS] Continuous trajectory figure saved to:\n  - {png_path}\n  - {artifact_png}")
