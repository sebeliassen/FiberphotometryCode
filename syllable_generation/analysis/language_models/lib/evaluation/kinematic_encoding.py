"""
Kinematic Encoding & Explained Variance Benchmark Module
Located at syllable_generation/analysis/language_models/lib/evaluation/kinematic_encoding.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes ANOVA Eta-squared (eta^2) explained variance: SS_between / SS_total.
    Ignores NaN values and label -1 (outliers/unclustered points).
    """
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    valid_mask = (~np.isnan(values)) & (labels >= 0)
    if not np.any(valid_mask):
        return 0.0
        
    v = values[valid_mask]
    lbls = labels[valid_mask]
    
    total_mean = np.mean(v)
    ss_total = np.sum((v - total_mean) ** 2)
    if ss_total == 0:
        return 0.0
        
    unique_labels = np.unique(lbls)
    ss_between = 0.0
    for l in unique_labels:
        l_mask = (lbls == l)
        n_k = np.sum(l_mask)
        if n_k > 0:
            mean_k = np.mean(v[l_mask])
            ss_between += n_k * ((mean_k - total_mean) ** 2)
        
    eta2 = ss_between / ss_total
    return float(np.clip(eta2, 0.0, 1.0))

def evaluate_session_kinematic_encoding(
    session_dataframes: Dict[str, pd.DataFrame],
    frame_strategy_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluates kinematic encoding across all sessions for a given strategy model.
    
    Metrics evaluated:
    - eta2_velocity: Locomotion Speed
    - eta2_dist_center: Distance to Arena Center
    - eta2_angular_vel: Turning Rate (|angular_velocity|)
    - eta2_cadence: Syllable Transition Cadence
    """
    all_vel = []
    all_dist = []
    all_ang = []
    all_cad = []
    all_strat = []
    
    for s_id, df in session_dataframes.items():
        if s_id not in frame_strategy_dict:
            continue
            
        strats = frame_strategy_dict[s_id]
        n = min(len(df), len(strats))
        sub_df = df.iloc[:n]
        sub_strats = strats[:n]
        
        # 1. Locomotion velocity
        vel = sub_df["velocity_px_s"].values if "velocity_px_s" in sub_df else np.zeros(n)
        
        # 2. Distance to center
        cx = sub_df["centroid_x"].values if "centroid_x" in sub_df else np.zeros(n)
        cy = sub_df["centroid_y"].values if "centroid_y" in sub_df else np.zeros(n)
        c_mean_x = np.nanmean(cx)
        c_mean_y = np.nanmean(cy)
        dist_c = np.sqrt((cx - c_mean_x) ** 2 + (cy - c_mean_y) ** 2)
        
        # 3. Angular velocity
        ang = np.abs(sub_df["angular_velocity"].values) if "angular_velocity" in sub_df else np.zeros(n)
        
        # 4. Syllable Cadence
        sylls = sub_df["syllable"].values if "syllable" in sub_df else np.zeros(n)
        cadence = np.zeros(n, dtype=float)
        if len(sylls) > 1:
            cadence[1:] = (sylls[1:] != sylls[:-1]).astype(float)
            
        all_vel.append(vel)
        all_dist.append(dist_c)
        all_ang.append(ang)
        all_cad.append(cadence)
        all_strat.append(sub_strats)
        
    concat_vel = np.concatenate(all_vel)
    concat_dist = np.concatenate(all_dist)
    concat_ang = np.concatenate(all_ang)
    concat_cad = np.concatenate(all_cad)
    concat_strat = np.concatenate(all_strat)
    
    eta2_vel = compute_eta_squared(concat_vel, concat_strat)
    eta2_dist = compute_eta_squared(concat_dist, concat_strat)
    eta2_ang = compute_eta_squared(concat_ang, concat_strat)
    eta2_cad = compute_eta_squared(concat_cad, concat_strat)
    
    return {
        "eta2_velocity": eta2_vel,
        "eta2_dist_center": eta2_dist,
        "eta2_angular_vel": eta2_ang,
        "eta2_cadence": eta2_cad
    }
