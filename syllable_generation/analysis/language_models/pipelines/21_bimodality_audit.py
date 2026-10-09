"""
Phase 2: Ultra-Fast Vectorized Bimodality & Unimodality Audit
File: syllable_generation/analysis/language_models/pipelines/21_bimodality_audit.py

Tests whether any of the discovered strategies in KMeans (K=8, 16) and GMM (K=8, 16)
are secretly bimodal or if they represent coherent, unimodal behavioral attractors.

Metrics evaluated per strategy state:
  1. GMM BIC Comparison: ΔBIC = BIC(1-component) - BIC(2-component).
     - ΔBIC < 0 or small: Unimodal.
     - ΔBIC >> 0: Potential sub-modes.
  2. Ashman's D Statistic: D = |μ1 - μ2| / sqrt(2 * (σ1² + σ2²)).
     - D > 2.0: Clean, well-separated bimodal peaks.
     - D < 2.0: Unimodal or single broad distribution.
  3. Diagnostic multi-panel summary plots for all strategy states.
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
AUDIT_DIR = os.path.join(OUT_DIR, "bimodality_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

# 1. Load Sessions & Metadata
print("Loading session datasets...")
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)
session_lookup = {s.session_id: s for s in raw_sessions}

# Pre-extract session kinematics once in memory for ultra-fast vectorized access
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

MODELS_TO_AUDIT = ["kmeans_k16", "kmeans_k8", "gmm_k16", "gmm_k8"]

def compute_ashmans_d(gmm2):
    """Compute Ashman's D statistic for 2-component GMM."""
    m1, m2 = gmm2.means_[0, 0], gmm2.means_[1, 0]
    s1 = np.sqrt(gmm2.covariances_[0, 0, 0]) if gmm2.covariances_.ndim == 3 else np.sqrt(gmm2.covariances_[0, 0])
    s2 = np.sqrt(gmm2.covariances_[1, 0, 0]) if gmm2.covariances_.ndim == 3 else np.sqrt(gmm2.covariances_[1, 0])
    denom = np.sqrt(2.0 * (s1**2 + s2**2))
    if denom <= 1e-6:
        return 0.0
    return float(np.abs(m1 - m2) / denom)

for model_name in MODELS_TO_AUDIT:
    print("\n=======================================================")
    print(f"Running Bimodality Audit on: {model_name}")
    print("=======================================================")

    intervals_csv = os.path.join(OUT_DIR, f"session_strategy_intervals_{model_name}.csv")
    if not os.path.exists(intervals_csv):
        print(f"Skipping {model_name}, intervals file not found: {intervals_csv}")
        continue
    df_intervals = pd.read_csv(intervals_csv)

    # Extract unique strategy IDs
    unique_strats = np.sort(df_intervals["strategy_id"].unique())
    K = len(unique_strats)

    # Collect arrays per strategy
    strat_vels = {s_id: [] for s_id in unique_strats}
    strat_dcs = {s_id: [] for s_id in unique_strats}
    strat_avs = {s_id: [] for s_id in unique_strats}

    for _, interval in df_intervals.iterrows():
        sid = interval["session_id"]
        if sid not in session_kinematics:
            continue

        s_frame = int(interval["start_frame"])
        e_frame = min(int(interval["end_frame"]), session_kinematics[sid]["len"])
        s_id = int(interval["strategy_id"])

        if e_frame <= s_frame:
            continue

        # Stride 2 (~40 Hz sampling) for ultra-fast performance
        v_sub = session_kinematics[sid]["velocity"][s_frame:e_frame:2]
        dc_sub = session_kinematics[sid]["dist_center"][s_frame:e_frame:2]
        av_sub = session_kinematics[sid]["angular_vel"][s_frame:e_frame:2]

        valid = ~np.isnan(v_sub) & ~np.isnan(dc_sub)
        if np.sum(valid) > 0:
            strat_vels[s_id].append(v_sub[valid])
            strat_dcs[s_id].append(dc_sub[valid])
            strat_avs[s_id].append(av_sub[valid])

    total_frames = sum(sum(len(a) for a in strat_vels[s_id]) for s_id in unique_strats)
    print(f"Aggregated {total_frames:,} sampled frames across {K} strategies.")

    audit_results = []

    # 2. Test Bimodality per Strategy
    for s_id in unique_strats:
        if len(strat_vels[s_id]) == 0:
            continue

        vel_arr = np.concatenate(strat_vels[s_id])
        dc_arr = np.concatenate(strat_dcs[s_id])
        n_f = len(vel_arr)
        pct_occupancy = float(n_f / total_frames * 100.0)

        vel_vals = vel_arr.reshape(-1, 1)
        dc_vals = dc_arr.reshape(-1, 1)

        # Subsample for fast GMM fitting (max 20,000 samples)
        if len(vel_vals) > 20000:
            idx_sub = np.random.choice(len(vel_vals), size=20000, replace=False)
            vel_sub = vel_vals[idx_sub]
            dc_sub = dc_vals[idx_sub]
        else:
            vel_sub = vel_vals
            dc_sub = dc_vals

        # Velocity Modality Test
        gmm1_v = GaussianMixture(n_components=1, random_state=42).fit(vel_sub)
        gmm2_v = GaussianMixture(n_components=2, random_state=42, n_init=3).fit(vel_sub)
        bic1_v, bic2_v = gmm1_v.bic(vel_sub), gmm2_v.bic(vel_sub)
        delta_bic_v = bic1_v - bic2_v
        ashman_d_v = compute_ashmans_d(gmm2_v)
        is_vel_bimodal = (delta_bic_v > 1000) and (ashman_d_v > 2.0)

        # Distance-to-Center Modality Test
        gmm1_dc = GaussianMixture(n_components=1, random_state=42).fit(dc_sub)
        gmm2_dc = GaussianMixture(n_components=2, random_state=42, n_init=3).fit(dc_sub)
        bic1_dc, bic2_dc = gmm1_dc.bic(dc_sub), gmm2_dc.bic(dc_sub)
        delta_bic_dc = bic1_dc - bic2_dc
        ashman_d_dc = compute_ashmans_d(gmm2_dc)
        is_dc_bimodal = (delta_bic_dc > 1000) and (ashman_d_dc > 2.0)

        status = "Bimodal" if (is_vel_bimodal or is_dc_bimodal) else "Unimodal (Coherent)"

        audit_results.append({
            "strategy_id": s_id,
            "occupancy_pct": pct_occupancy,
            "mean_velocity": float(np.mean(vel_arr)),
            "std_velocity": float(np.std(vel_arr)),
            "vel_delta_bic": float(delta_bic_v),
            "vel_ashman_d": float(ashman_d_v),
            "vel_bimodal": bool(is_vel_bimodal),
            "mean_dist_center": float(np.mean(dc_arr)),
            "std_dist_center": float(np.std(dc_arr)),
            "dc_delta_bic": float(delta_bic_dc),
            "dc_ashman_d": float(ashman_d_dc),
            "dc_bimodal": bool(is_dc_bimodal),
            "verdict": status
        })

    df_audit = pd.DataFrame(audit_results)
    csv_out = os.path.join(AUDIT_DIR, f"bimodality_summary_{model_name}.csv")
    df_audit.to_csv(csv_out, index=False)
    print(f"Saved audit CSV to: {csv_out}")

    print("\n--- Summary Audit Table ---")
    print(df_audit[["strategy_id", "occupancy_pct", "mean_velocity", "vel_ashman_d", "dc_ashman_d", "verdict"]].to_string(index=False))

    # 3. Plot Diagnostic Modality Grid
    n_cols = 4
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), dpi=200)
    axes = np.array(axes).reshape(-1)

    for idx, s_id in enumerate(unique_strats):
        ax = axes[idx]
        if len(strat_vels[s_id]) == 0:
            continue
        v_data = np.concatenate(strat_vels[s_id])
        v_data = v_data[v_data < np.percentile(v_data, 99)]

        row_stat = df_audit[df_audit["strategy_id"] == s_id].iloc[0]
        color = "#ef4444" if "Bimodal" in row_stat["verdict"] else "#10b981"

        ax.hist(v_data, bins=40, density=True, color=color, alpha=0.65, edgecolor="none")
        ax.set_title(f"Strategy {s_id} ({row_stat['occupancy_pct']:.1f}%)\n{row_stat['verdict']}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Speed (px/s)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(K, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Bimodality & Kinematic Coherence Audit: {model_name.upper()} (K={K})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_out = os.path.join(AUDIT_DIR, f"bimodality_grid_{model_name}.png")
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved diagnostic grid plot to: {plot_out}")

print("\n✅ Phase 2 Complete: All 4 models audited with zero bottlenecks.")
