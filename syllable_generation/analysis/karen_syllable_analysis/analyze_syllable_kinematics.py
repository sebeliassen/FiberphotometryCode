#!/usr/bin/env python3
"""
analyze_syllable_kinematics.py

Step 1: Quantify within-syllable centroid speed, forward velocity,
and angular velocity distributions across bouts to test whether
syllables (especially regulated syllables 6 & 72, and top frequent syllables)
exhibit genuine bimodal multimodality across locomotor contexts.
"""

import os
import pickle
from pathlib import Path
import numpy as np
np.bool8 = np.bool_
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import keypoint_moseq as kpms

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2")
MODEL_NAME = "final_karen_model_k240k"
OUT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_syllable_analysis/output/kinematic_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📈 STEP 1: QUANTITATIVE WITHIN-SYLLABLE KINEMATIC & SPEED ANALYSIS")
print("=" * 80)

# 1. Load results and config
print("[1/4] Loading model results and metadata...")
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)
fps = config.get("fps", 40)
dt = 1.0 / fps

# 2. Extract syllable sequences, centroids, and headings
print(f"[2/4] Processing {len(results)} recordings at {fps} fps...")
centroids = {k: v["centroid"] for k, v in results.items()}
headings = {k: v["heading"] for k, v in results.items()}
syllables = {k: v["syllable"] for k, v in results.items()}

# Pre-filter centroids and headings for clean velocities (window=5)
filt_centroids, filt_headings = kpms.viz.filter_centroids_headings(centroids, headings, filter_size=5)

# Calculate syllable frequencies to find top syllables
all_sylls = np.concatenate([v for v in syllables.values()])
counts = pd.Series(all_sylls).value_counts()
top_sylls = counts.index[:8].tolist()

# Target syllables: Regulated syllables 6 & 72 + Top 6 most frequent
target_syllables = [6, 72] + [s for s in top_sylls if s not in [6, 72]][:6]
print(f"Target syllables for kinematic profiling: {target_syllables}")

# 3. Extract bout-level kinematic metrics
print("[3/4] Extracting bout-level speed, forward velocity, and angular velocity...")
bout_records = []

for key in results.keys():
    syll_seq = syllables[key]
    c_seq = filt_centroids[key]  # (T, 2)
    h_seq = filt_headings[key]   # (T,)
    
    # Instantaneous kinematics
    # Centroid displacement per frame (px/frame)
    vel_vec = np.diff(c_seq, axis=0) * fps  # (T-1, 2) in px/s
    speed = np.linalg.norm(vel_vec, axis=-1) # (T-1,) in px/s
    
    # Forward velocity: projection of velocity vector onto heading direction
    heading_vec = np.stack([np.cos(h_seq[:-1]), np.sin(h_seq[:-1])], axis=-1)
    forward_vel = np.sum(vel_vec * heading_vec, axis=-1) # (T-1,) px/s
    
    # Angular velocity: wrap angle difference
    dh = np.diff(h_seq)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ang_vel = np.abs(dh) * fps * (180.0 / np.pi) # (T-1,) deg/s
    
    # Identify contiguous bouts of each syllable
    # Bout changes where diff != 0
    change_idx = np.where(np.diff(syll_seq) != 0)[0] + 1
    bout_starts = np.insert(change_idx, 0, 0)
    bout_ends = np.append(change_idx, len(syll_seq))
    
    for start, end in zip(bout_starts, bout_ends):
        duration = end - start
        if duration < 2:
            continue
        s_id = syll_seq[start]
        if s_id in target_syllables:
            # Bout slice for kinematics (up to T-1)
            e_k = min(end, len(speed))
            if start < e_k:
                m_speed = np.mean(speed[start:e_k])
                m_fwd = np.mean(forward_vel[start:e_k])
                m_ang = np.mean(ang_vel[start:e_k])
                
                bout_records.append({
                    "recording": key,
                    "syllable": s_id,
                    "duration_frames": duration,
                    "duration_sec": duration * dt,
                    "mean_speed_px_s": m_speed,
                    "mean_forward_vel_px_s": m_fwd,
                    "mean_angular_vel_deg_s": m_ang,
                })

df_bouts = pd.DataFrame(bout_records)
print(f"Extracted {len(df_bouts)} total bouts across target syllables.")

# Calculate Sarle's Bimodality Coefficient (BC)
# BC = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
# BC > 0.555 suggests a bimodal or multimodal distribution
def calc_bimodality_coeff(data):
    n = len(data)
    if n < 10:
        return 0.0
    skew = stats.skew(data)
    kurt = stats.kurtosis(data, fisher=False) # Pearson kurtosis (normal = 3)
    sample_factor = 3 * ((n - 1)**2) / ((n - 2) * (n - 3))
    bc = (skew**2 + 1) / (kurt + sample_factor)
    return bc

# Print summary table
print("\n" + "=" * 80)
print("📊 BOUT-LEVEL KINEMATIC METRICS & BIMODALITY DIAGNOSTIC")
print("=" * 80)
print(f"{'Syllable':<10} | {'Bouts':<8} | {'Median Dur':<12} | {'Mean Speed':<12} | {'Fwd Vel':<12} | {'Bimodality Coeff (BC)':<20} | {'Shape'}")
print("-" * 90)

bimodality_results = {}
for s in target_syllables:
    sub = df_bouts[df_bouts["syllable"] == s]
    speeds = sub["mean_speed_px_s"].values
    dur = sub["duration_sec"].values
    fwd = sub["mean_forward_vel_px_s"].values
    
    bc_speed = calc_bimodality_coeff(speeds)
    is_bimodal = bc_speed > 0.555
    bimodality_results[s] = bc_speed
    
    shape_str = "⚠️ Bimodal/Multi" if is_bimodal else "✅ Unimodal"
    tag = "★ SIG" if s in [6, 72] else ""
    s_label = f"Syllable {s} {tag}".strip()
    
    print(f"{s_label:<10} | {len(sub):<8} | {np.median(dur):.3f}s ({np.median(sub['duration_frames']):.0f}f) | {np.mean(speeds):>7.1f} px/s | {np.mean(fwd):>7.1f} px/s | {bc_speed:>10.3f} (crit: 0.555) | {shape_str}")

print("=" * 80)

# 4. Generate Publication-Quality Visualization
print("\n[4/4] Plotting kinematic distributions (Speed, Forward Velocity, Angular Velocity)...")
fig, axes = plt.subplots(len(target_syllables), 3, figsize=(15, 2.5 * len(target_syllables)), sharex="col")

for idx, s in enumerate(target_syllables):
    sub = df_bouts[df_bouts["syllable"] == s]
    is_sig = s in [6, 72]
    color = "#D32F2F" if is_sig else "#1976D2"
    
    # Col 1: Speed distribution
    ax1 = axes[idx, 0]
    sns.histplot(sub["mean_speed_px_s"], kde=True, ax=ax1, color=color, bins=35, stat="density", alpha=0.6)
    bc = bimodality_results[s]
    bc_tag = f"BC={bc:.2f} (Bimodal)" if bc > 0.555 else f"BC={bc:.2f} (Unimodal)"
    ax1.set_ylabel(f"Syllable {s}\n{'[SIG]' if is_sig else ''}", fontweight="bold", color=color)
    ax1.set_title(f"Speed Distribution ({bc_tag})", fontsize=10, loc="left")
    
    # Col 2: Forward Velocity (Signed)
    ax2 = axes[idx, 1]
    sns.histplot(sub["mean_forward_vel_px_s"], kde=True, ax=ax2, color=color, bins=35, stat="density", alpha=0.6)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_title("Forward vs Backward Velocity", fontsize=10, loc="left")
    ax2.set_ylabel("")
    
    # Col 3: Angular Velocity
    ax3 = axes[idx, 2]
    sns.histplot(sub["mean_angular_vel_deg_s"], kde=True, ax=ax3, color=color, bins=35, stat="density", alpha=0.6)
    ax3.set_title("Angular Turning Rate (deg/s)", fontsize=10, loc="left")
    ax3.set_ylabel("")

axes[-1, 0].set_xlabel("Mean Centroid Speed (px/s)", fontweight="bold")
axes[-1, 1].set_xlabel("Mean Forward Velocity (px/s)", fontweight="bold")
axes[-1, 2].set_xlabel("Mean Angular Velocity (deg/s)", fontweight="bold")

plt.suptitle("Quantitative Kinematic Distribution Analysis Across Syllable Bouts", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()

out_fig_png = OUT_DIR / "syllable_kinematic_distributions.png"
out_fig_pdf = OUT_DIR / "syllable_kinematic_distributions.pdf"
plt.savefig(out_fig_png, dpi=300, bbox_inches="tight")
plt.savefig(out_fig_pdf, bbox_inches="tight")
plt.close()

# Save bout statistics CSV
csv_out = OUT_DIR / "bout_kinematic_metrics.csv"
df_bouts.to_csv(csv_out, index=False)

print(f"Saved publication figures to:\n  • {out_fig_png}\n  • {out_fig_pdf}")
print(f"Saved bout-level kinematic table to: {csv_out}")
print("=" * 80)
