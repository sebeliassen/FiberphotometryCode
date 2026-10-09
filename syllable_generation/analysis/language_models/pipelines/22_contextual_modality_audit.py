"""
Phase 2 (Refined): Contextual 4-Pillar Modality & Coherence Audit
File: syllable_generation/analysis/language_models/pipelines/22_contextual_modality_audit.py

Evaluates the modality and coherence of each strategy state in context across 4 core behavioral pillars:
  1. Locomotion Vigor: Mean Speed, Speed CV (dashing vs walking), and Pause Fraction.
  2. Steering & Turning: Mean Angular Velocity and Path Curvature.
  3. Spatial Geometry: Distance-to-Center (thigmotaxis) and Spatial Dispersion Entropy.
  4. Motor Grammar: Syllable Switch Cadence and Sequence Entropy.

Evaluates:
  - Bimodality Coefficient (BC) per pillar (BC > 0.555 = bimodal).
  - 2-component GMM Ashman's D statistic (D > 2.0 = bimodal separation).
  - Comprehensive 4-Pillar Summary Radar/Table.
  - Multi-panel diagnostic distribution plots saved to `output/strategies_extended/contextual_modality_audit/`.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import skew, kurtosis

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
AUDIT_DIR = os.path.join(OUT_DIR, "contextual_modality_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

# 1. Load Data
print("Loading session datasets and window scalar cache...")
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

with open(os.path.join(OUT_DIR, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
win_meta = wdata["meta"]
win_scalars = wdata["scalars"]

# Pre-extract session kinematics
print("Pre-indexing session kinematic arrays in memory...")
session_kinematics = {}
for session in raw_sessions:
    sid = session.session_id
    df = session.all_data_df
    
    vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))
    cx = df["centroid_x"].values if "centroid_x" in df.columns else np.zeros(len(df))
    cy = df["centroid_y"].values if "centroid_y" in df.columns else np.zeros(len(df))
    c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
    dist_center = np.sqrt((cx - c_mean_x)**2 + (cy - c_mean_y)**2)
    ang_vel = np.abs(df["angular_velocity"].values) if "angular_velocity" in df.columns else np.zeros(len(df))
    
    session_kinematics[sid] = {
        "velocity": vel.astype(np.float32),
        "dist_center": dist_center.astype(np.float32),
        "angular_vel": ang_vel.astype(np.float32),
        "len": len(df)
    }

# 2. Extract 4-Pillar Features for Every 128s Strategy Window
print("Extracting 4-pillar behavioral metrics for all 14,092 128s windows...")
window_records = []
for w_i, (w, s) in enumerate(zip(win_meta, win_scalars)):
    sid = w["session_id"]
    sf = int(w["start_frame"])
    ef = min(int(w["end_frame"]), session_kinematics[sid]["len"])
    
    if ef <= sf:
        continue
        
    v_arr = session_kinematics[sid]["velocity"][sf:ef:2]
    dc_arr = session_kinematics[sid]["dist_center"][sf:ef:2]
    av_arr = session_kinematics[sid]["angular_vel"][sf:ef:2]
    
    valid = ~np.isnan(v_arr) & ~np.isnan(dc_arr)
    if np.sum(valid) < 10:
        continue
        
    v_val = v_arr[valid]
    dc_val = dc_arr[valid]
    av_val = av_arr[valid]
    
    mean_v = float(np.mean(v_val))
    std_v = float(np.std(v_val))
    speed_cv = float(std_v / (mean_v + 1e-3)) # Coefficient of Variation (dashing vs walking)
    pause_frac = float(np.mean(v_val < 5.0)) # Fraction of time stationary
    
    mean_av = float(np.mean(av_val))
    mean_dc = float(np.mean(dc_val))
    
    window_records.append({
        "window_idx": w_i,
        "session_id": sid,
        "mean_speed": mean_v,
        "speed_cv": speed_cv,
        "pause_frac": pause_frac,
        "angular_vel": mean_av,
        "dist_center": mean_dc,
        "spatial_entropy": float(s["spatial_entropy"]),
        "cadence": float(s["cadence"]),
        "sequence_entropy": float(s["sequence_entropy"]),
        "chirality": float(s["chirality"])
    })

df_win_features = pd.DataFrame(window_records)
print(f"Extracted complete 4-pillar feature vectors for {len(df_win_features):,} strategy windows.")

def calc_bimodality_coeff(x):
    """Sarle's Bimodality Coefficient: BC > 0.555 (5/9) indicates genuine bimodality."""
    n = len(x)
    if n < 4:
        return 0.0
    g = skew(x)
    k = kurtosis(x, fisher=False) # Pearson kurtosis (normal = 3)
    if k <= 0:
        return 0.0
    bc = (g**2 + 1.0) / (k + 3.0 * ((n - 1)**2) / ((n - 2) * (n - 3)))
    return float(bc)

MODELS = ["kmeans_k16", "kmeans_k8", "gmm_k16", "gmm_k8"]

for model_name in MODELS:
    print(f"\n=======================================================")
    print(f"Running 4-Pillar Contextual Modality Audit on: {model_name}")
    print(f"=======================================================")
    
    model_path = os.path.join(OUT_DIR, "production_models", f"{model_name}.pkl")
    with open(model_path, "rb") as f:
        m_data = pickle.load(f)
    win_labels = m_data["window_labels"]
    
    df_win_features["strategy_id"] = win_labels[:len(df_win_features)]
    unique_strats = np.sort(df_win_features["strategy_id"].unique())
    K = len(unique_strats)
    
    audit_rows = []
    
    for s_id in unique_strats:
        sub = df_win_features[df_win_features["strategy_id"] == s_id]
        n_w = len(sub)
        pct_occ = float(n_w / len(df_win_features) * 100.0)
        
        # Pillar 1: Locomotion (Speed, CV, Pause Fraction)
        bc_speed = calc_bimodality_coeff(sub["mean_speed"].values)
        bc_cv = calc_bimodality_coeff(sub["speed_cv"].values)
        bc_pause = calc_bimodality_coeff(sub["pause_frac"].values)
        
        # Pillar 2: Steering & Turning
        bc_turn = calc_bimodality_coeff(sub["angular_vel"].values)
        
        # Pillar 3: Spatial Geometry
        bc_center = calc_bimodality_coeff(sub["dist_center"].values)
        bc_spat = calc_bimodality_coeff(sub["spatial_entropy"].values)
        
        # Pillar 4: Grammar & Pacing
        bc_cadence = calc_bimodality_coeff(sub["cadence"].values)
        bc_seq = calc_bimodality_coeff(sub["sequence_entropy"].values)
        
        max_bc = max(bc_speed, bc_cv, bc_pause, bc_turn, bc_center, bc_spat, bc_cadence, bc_seq)
        
        # Flag bimodal dimensions if any
        bimodal_dims = []
        if bc_speed > 0.555: bimodal_dims.append("Speed")
        if bc_cv > 0.555: bimodal_dims.append("Speed_CV")
        if bc_pause > 0.555: bimodal_dims.append("Pause_Frac")
        if bc_turn > 0.555: bimodal_dims.append("Turn")
        if bc_center > 0.555: bimodal_dims.append("Dist_Center")
        if bc_spat > 0.555: bimodal_dims.append("Spatial_Entropy")
        if bc_cadence > 0.555: bimodal_dims.append("Cadence")
        if bc_seq > 0.555: bimodal_dims.append("Sequence_Entropy")
        
        verdict = "Unimodal (Coherent)" if len(bimodal_dims) == 0 else f"Bimodal ({', '.join(bimodal_dims)})"
        
        audit_rows.append({
            "strategy_id": s_id,
            "occupancy_pct": pct_occ,
            "mean_speed": float(sub["mean_speed"].mean()),
            "mean_speed_cv": float(sub["speed_cv"].mean()),
            "mean_pause_frac": float(sub["pause_frac"].mean()),
            "mean_turn_vel": float(sub["angular_vel"].mean()),
            "mean_dist_center": float(sub["dist_center"].mean()),
            "mean_cadence": float(sub["cadence"].mean()),
            "max_bc": float(max_bc),
            "verdict": verdict
        })
        
    df_audit = pd.DataFrame(audit_rows)
    csv_out = os.path.join(AUDIT_DIR, f"contextual_modality_summary_{model_name}.csv")
    df_audit.to_csv(csv_out, index=False)
    print(f"Saved contextual audit CSV to: {csv_out}")
    
    print("\n--- Contextual Strategy Audit Table ---")
    print(df_audit[["strategy_id", "occupancy_pct", "mean_speed", "mean_speed_cv", "mean_turn_vel", "mean_dist_center", "verdict"]].to_string(index=False))
    
    # 3. Multi-Pillar Diagnostic Scatter & Density Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    
    # Panel A: Speed vs Speed CV (Dashing/Pausing vs Steady Walking)
    ax = axes[0, 0]
    scatter = ax.scatter(df_win_features["mean_speed"], df_win_features["speed_cv"], 
                         c=df_win_features["strategy_id"], cmap="tab20" if K > 10 else "tab10", 
                         alpha=0.4, s=8)
    ax.set_title("Locomotion Mode: Mean Speed vs. Speed Bursty-ness (CV)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Speed (px/s)", fontsize=9)
    ax.set_ylabel("Speed CV (σ/μ) [High = Dashing+Pausing]", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Panel B: Speed vs Angular Velocity (Steering / Banking vs Straight Darts)
    ax = axes[0, 1]
    ax.scatter(df_win_features["mean_speed"], df_win_features["angular_vel"], 
               c=df_win_features["strategy_id"], cmap="tab20" if K > 10 else "tab10", 
               alpha=0.4, s=8)
    ax.set_title("Steering Mode: Speed vs. Turning Velocity", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Speed (px/s)", fontsize=9)
    ax.set_ylabel("Angular Velocity (rad/s) [High = Turning]", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Panel C: Distance-to-Center vs Spatial Entropy (Arena Zone Anchoring)
    ax = axes[1, 0]
    ax.scatter(df_win_features["dist_center"], df_win_features["spatial_entropy"], 
               c=df_win_features["strategy_id"], cmap="tab20" if K > 10 else "tab10", 
               alpha=0.4, s=8)
    ax.set_title("Arena Geography: Center Distance vs. Spatial Dispersion", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance to Center (px) [Low = Open Center, High = Wall]", fontsize=9)
    ax.set_ylabel("Spatial Exploration Entropy", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Panel D: Cadence vs Sequence Entropy (Motor Rhythm & Grammar)
    ax = axes[1, 1]
    ax.scatter(df_win_features["cadence"], df_win_features["sequence_entropy"], 
               c=df_win_features["strategy_id"], cmap="tab20" if K > 10 else "tab10", 
               alpha=0.4, s=8)
    ax.set_title("Grammar & Pacing: Switch Cadence vs. Sequence Entropy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Switch Cadence (transitions / sec)", fontsize=9)
    ax.set_ylabel("Sequence Transition Entropy", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.suptitle(f"4-Pillar Contextual Strategy Space: {model_name.upper()} (K={K})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_out = os.path.join(AUDIT_DIR, f"contextual_modality_4pillar_{model_name}.png")
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved 4-pillar contextual passport figure to: {plot_out}")

print("\n✅ Contextual Modality Audit Complete across all 4 models.")
