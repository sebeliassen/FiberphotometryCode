#!/usr/bin/env python3
"""
analyze_coordinate_jitter.py

Empirically analyzes the high-frequency spectral components (jitter/noise)
across three levels of coordinates in the Karen dataset:
1. Raw DeepLabCut keypoint predictions (directly from .h5 output)
2. Preprocessed keypoint coordinates entering Keypoint-MoSeq (data['Y'])
3. Reconstructed continuous model estimates (Y_est = estimate_coordinates(x, v, h, Cd))

Also isolates whether the jitter in Y_est originates from:
- Postural latents (Cd * x)
- Heading rotation (h)
- Centroid translation (v)
"""

import sys

sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")
import pickle
from pathlib import Path

import numpy as np

np.bool8 = np.bool_
import jax.numpy as jnp
import keypoint_moseq as kpms
import pandas as pd
from jax_moseq.models.keypoint_slds import estimate_aligned, estimate_coordinates, rigid_transform
from jax_moseq.utils import unbatch
from scipy import signal

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2")
MODEL_NAME = "final_karen_model_k240k"
DATA_PKL = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl")

print("=" * 80)
print("🔬 EMPIRICAL JITTER & HIGH-FREQUENCY SPECTRAL ANALYSIS")
print("=" * 80)

# 1. Load model and metadata
print("[1/5] Loading Keypoint-MoSeq model and checkpoint data...")
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

key0 = str(metadata[0][0])
print(f"Analyzing session: {key0[:60]}...")

# 2. Extract model inputs and model estimates
print("[2/5] Extracting input coordinates and model estimates...")
Y_input_dict = unbatch(data["Y"], *metadata)
input_coords = np.array(Y_input_dict[key0])  # (T, 17, 2) at 40 Hz
T = len(input_coords)

# Reconstruct Y_est
Y_est_all = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
Y_est_dict = unbatch(Y_est_all, *metadata)
est_coords = np.array(Y_est_dict[key0])  # (T, 17, 2) at 40 Hz

# Also extract egocentric posture and heading/centroid
Cd = np.array(model["params"]["Cd"])
k_body = Cd.shape[0] // 2 + 1
x_dict = unbatch(model["states"]["x"], *metadata)
v_dict = unbatch(model["states"]["v"], *metadata)
h_dict = unbatch(model["states"]["h"], *metadata)

x_sess = np.array(x_dict[key0])
v_sess = np.array(v_dict[key0])
h_sess = np.array(h_dict[key0])

# 3. Load raw DeepLabCut coordinates
print("[3/5] Loading raw DeepLabCut .h5 tracking...")
with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

# Find session matching key0
s0 = None
for s in sessions:
    ref_path = s.tracking_refs_df["path"].iloc[0]
    if Path(ref_path).stem in key0 or key0 in Path(ref_path).stem:
        s0 = s
        break
if s0 is None:
    s0 = sessions[0]

raw_h5_path = s0.tracking_refs_df["path"].iloc[0]
dlc_df = pd.read_hdf(raw_h5_path)
scorer = dlc_df.columns.levels[0][0]
dlc_bodyparts = dlc_df.columns.levels[1].tolist()

# Extract a common keypoint: snout / nose
target_bp = "nose" if "nose" in dlc_bodyparts else dlc_bodyparts[0]
print(f"Target bodypart for detailed comparison: '{target_bp}'")

raw_x = dlc_df[scorer][target_bp]["x"].values
raw_y = dlc_df[scorer][target_bp]["y"].values
raw_coords_full = np.stack([raw_x, raw_y], axis=-1)

# Check sampling rate: if raw is 80 Hz, downsample to 40 Hz
if len(raw_coords_full) >= 2 * T:
    raw_coords_80 = raw_coords_full[:2*T]
    raw_coords_40 = raw_coords_80[::2]  # Direct 40 Hz subsample
else:
    raw_coords_40 = raw_coords_full[:T]
    raw_coords_80 = None

# Extract target keypoint from KPMS arrays
# Check bodyparts list in config or use index 0
inp_bp = input_coords[:, 0, :]
est_bp = est_coords[:, 0, :]

# Also create a filtered version of model estimates (smoothed h and v)
# Using a 5-frame Savitzky-Golay filter on heading and centroid
h_smooth = signal.savgol_filter(h_sess, window_length=7, polyorder=2)
v_smooth = signal.savgol_filter(v_sess, window_length=7, polyorder=2, axis=0)

Y_bar = np.array(estimate_aligned(jnp.array(x_sess), jnp.array(Cd), k_body))
Y_smooth_est = np.array(rigid_transform(jnp.array(Y_bar), jnp.array(v_smooth), jnp.array(h_smooth)))
est_smooth_bp = Y_smooth_est[:, 0, :]

# ==============================================================================
# 4. METRICS COMPUTATION
# ==============================================================================
print("[4/5] Computing Spectral Density and Kinematic Smoothness Metrics...")

def get_psd_metrics(trajectory_2d, fs=40.0):
    """Computes Welch PSD for 2D velocity and high-frequency power ratios."""
    # Velocity (px/s)
    vel = np.diff(trajectory_2d, axis=0) * fs
    speed = np.linalg.norm(vel, axis=-1)

    # Detrended speed PSD
    f, pxx = signal.welch(speed - np.mean(speed), fs=fs, nperseg=min(len(speed), 512))

    total_power = np.trapz(pxx, f)
    hf_10 = np.trapz(pxx[f >= 10.0], f[f >= 10.0]) / (total_power + 1e-12)
    hf_15 = np.trapz(pxx[f >= 15.0], f[f >= 15.0]) / (total_power + 1e-12)

    # Jerk (d^3 x / dt^3)
    acc = np.diff(vel, axis=0) * fs
    jerk = np.diff(acc, axis=0) * fs
    mean_jerk = np.mean(np.linalg.norm(jerk, axis=-1))

    return {
        "mean_jerk": mean_jerk,
        "hf_power_gt_10hz": hf_10 * 100,
        "hf_power_gt_15hz": hf_15 * 100,
        "f": f,
        "pxx": pxx
    }

m_raw40 = get_psd_metrics(raw_coords_40, fs=40.0)
m_inp   = get_psd_metrics(inp_bp, fs=40.0)
m_est   = get_psd_metrics(est_bp, fs=40.0)
m_sm_est = get_psd_metrics(est_smooth_bp, fs=40.0)

# Heading spectral analysis
f_h, pxx_h = signal.welch(np.diff(h_sess) * 40.0, fs=40.0, nperseg=512)
tot_h = np.trapz(pxx_h, f_h)
hf_h_10 = np.trapz(pxx_h[f_h >= 10.0], f_h[f_h >= 10.0]) / (tot_h + 1e-12)

# ==============================================================================
# 5. DISPLAY RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print(f"📊 COMPARATIVE FREQUENCY SPECTRUM & JITTER BREAKDOWN (Bodypart: {target_bp})")
print("=" * 80)

print(f"{'Data Stage':<32} | {'Mean Jerk (px/s³)':<18} | {'> 10 Hz Power %':<16} | {'> 15 Hz Power %':<16}")
print("-" * 88)
print(f"{'1. Raw DLC (Direct 40Hz)':<32} | {m_raw40['mean_jerk']:>16.2f}  | {m_raw40['hf_power_gt_10hz']:>14.2f}% | {m_raw40['hf_power_gt_15hz']:>14.2f}%")
label2 = "2. KPMS Input (data['Y'])"
print(f"{label2:<32} | {m_inp['mean_jerk']:>16.2f}  | {m_inp['hf_power_gt_10hz']:>14.2f}% | {m_inp['hf_power_gt_15hz']:>14.2f}%")
print(f"{'3. KPMS Model Estimate (Y_est)':<32} | {m_est['mean_jerk']:>16.2f}  | {m_est['hf_power_gt_10hz']:>14.2f}% | {m_est['hf_power_gt_15hz']:>14.2f}%")
print(f"{'4. Y_est with Filtered (h, v)':<32} | {m_sm_est['mean_jerk']:>16.2f}  | {m_sm_est['hf_power_gt_10hz']:>14.2f}% | {m_sm_est['hf_power_gt_15hz']:>14.2f}%")
print("=" * 80)

print("\n🔍 HEADING (h) & CENTROID (v) DECOMPOSITION:")
print(f"  • Heading angular velocity high-frequency power (>10 Hz): {hf_h_10 * 100:.2f}% of total angular variance!")
print(f"  • Raw model estimate jerk vs filtered heading/centroid jerk reduction: {((m_est['mean_jerk'] - m_sm_est['mean_jerk']) / m_est['mean_jerk']) * 100:.1f}% reduction!")
print("=" * 80)
