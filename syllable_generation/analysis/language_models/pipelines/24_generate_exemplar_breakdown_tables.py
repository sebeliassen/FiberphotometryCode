"""
Generate Individual Strategy Breakdown Table comparing 100% vs 5% vs 1%
File: syllable_generation/analysis/language_models/pipelines/24_generate_exemplar_breakdown_tables.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
MODEL_DIR = os.path.join(OUT_DIR, "production_models")
AUDIT_DIR = os.path.join(OUT_DIR, "exemplar_audit")

# Load Datasets
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

with open(os.path.join(OUT_DIR, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
raw_trans_inputs = wdata["trans_inputs"]
win_meta = wdata["meta"]
win_scalars = wdata["scalars"]

# Kinematics
session_kinematics = {}
for session in raw_sessions:
    sid = session.session_id
    df = session.all_data_df
    vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))
    cx = df["centroid_x"].values if "centroid_x" in df.columns else np.zeros(len(df))
    cy = df["centroid_y"].values if "centroid_y" in df.columns else np.zeros(len(df))
    dist_center = np.sqrt((cx - np.nanmean(cx))**2 + (cy - np.nanmean(cy))**2)
    h = df["heading"].values if "heading" in df.columns else np.zeros(len(df))
    dh = np.diff(h, prepend=h[0])
    dh_wrapped = (dh + np.pi) % (2 * np.pi) - np.pi
    ang_vel = np.clip(np.abs(dh_wrapped) * 80.025, 0.0, 25.0)
    session_kinematics[sid] = {
        "velocity": vel.astype(np.float32),
        "dist_center": dist_center.astype(np.float32),
        "angular_vel": ang_vel.astype(np.float32),
        "len": len(df)
    }

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
    v_val, dc_val, av_val = v_arr[valid], dc_arr[valid], av_arr[valid]
    mean_v, std_v = float(np.mean(v_val)), float(np.std(v_val))
    window_records.append({
        "window_idx": w_i,
        "session_id": sid,
        "mean_speed": mean_v,
        "speed_cv": float(std_v / (mean_v + 1e-3)),
        "angular_vel": float(np.mean(av_val)),
        "dist_center": float(np.mean(dc_val)),
        "cadence": float(s["cadence"]),
        "sequence_entropy": float(s["sequence_entropy"])
    })

df_win_features = pd.DataFrame(window_records)
valid_win_indices = df_win_features["window_idx"].values

# For KMeans K=16 and GMM K=16
for model_name in ["kmeans_k16", "gmm_k16"]:
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "rb") as f:
        m_data = pickle.load(f)
    clf = m_data["model"]
    svd = m_data["svd"]
    cfg = m_data["config"]
    full_labels = m_data["window_labels"][valid_win_indices]
    full_probs = m_data["window_probs"][valid_win_indices]

    mat = raw_trans_inputs[valid_win_indices] + cfg["laplace_smooth"]
    if cfg["norm_type"] == "l1":
        from sklearn.preprocessing import normalize
        mat = normalize(mat, norm="l1")
    latent_z = svd.transform(mat)

    point_scores = np.zeros(len(latent_z), dtype=np.float32)
    if cfg["family"] == "KMeans":
        centroids = clf.cluster_centers_
        for i in range(len(latent_z)):
            point_scores[i] = np.linalg.norm(latent_z[i] - centroids[full_labels[i]])
    else:
        for i in range(len(latent_z)):
            point_scores[i] = -full_probs[i, full_labels[i]]

    breakdown_rows = []
    for s_id in np.sort(np.unique(full_labels)):
        c_indices = np.where(full_labels == s_id)[0]
        n_c = len(c_indices)
        sort_order = np.argsort(point_scores[c_indices])

        # 100%
        sub_100 = df_win_features.iloc[c_indices]
        # 10%
        n_10 = max(int(np.ceil(n_c * 0.10)), 4)
        sub_10 = df_win_features.iloc[c_indices[sort_order[:n_10]]]
        # 1%
        n_1 = max(int(np.ceil(n_c * 0.01)), 4)
        sub_1 = df_win_features.iloc[c_indices[sort_order[:n_1]]]

        breakdown_rows.append({
            "strategy_id": s_id,
            "n_total": n_c,
            "speed_100": float(sub_100["mean_speed"].mean()),
            "speed_10": float(sub_10["mean_speed"].mean()),
            "speed_1": float(sub_1["mean_speed"].mean()),
            "turn_100": float(sub_100["angular_vel"].mean()),
            "turn_10": float(sub_10["angular_vel"].mean()),
            "turn_1": float(sub_1["angular_vel"].mean()),
            "dist_100": float(sub_100["dist_center"].mean()),
            "dist_10": float(sub_10["dist_center"].mean()),
            "dist_1": float(sub_1["dist_center"].mean()),
            "cadence_100": float(sub_100["cadence"].mean()),
            "cadence_10": float(sub_10["cadence"].mean()),
            "cadence_1": float(sub_1["cadence"].mean()),
        })

    df_bd = pd.DataFrame(breakdown_rows)
    out_csv = os.path.join(AUDIT_DIR, f"strategy_exemplar_breakdown_{model_name}.csv")
    df_bd.to_csv(out_csv, index=False)
    print(f"Saved {model_name} archetype breakdown to: {out_csv}")
    print(df_bd.to_string(index=False))

print("Done!")
