import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def compute_window_scalars(session, start_frame: int, end_frame: int, n_grid: int = 8) -> Dict[str, float]:
    """
    Computes 4 zero-knowledge macroscopic behavioral scalars for a 128s sliding window:
      1. Spatial Exploration Entropy (H_spatial): Shannon entropy over an 8x8 spatial grid.
      2. Switch Cadence: State switches per second.
      3. Turning Chirality Ratio: Mean |angular_velocity| / (velocity + eps).
      4. Sequence Entropy (H_seq): Transition entropy of the syllable sequence.
    """
    df = session.all_data_df
    syll_df = session.syllables
    
    sub_df = df.loc[start_frame:end_frame]
    sub_syll = syll_df.loc[start_frame:end_frame]
    
    # 1. Spatial Exploration Entropy
    cx = sub_df['centroid_x'].values
    cy = sub_df['centroid_y'].values
    valid_coords = ~np.isnan(cx) & ~np.isnan(cy)
    
    if np.sum(valid_coords) > 10:
        x_clean = cx[valid_coords]
        y_clean = cy[valid_coords]
        
        x_min, x_max = np.min(x_clean), np.max(x_clean)
        y_min, y_max = np.min(y_clean), np.max(y_clean)
        
        if x_max > x_min and y_max > y_min:
            x_bins = np.linspace(x_min, x_max, n_grid + 1)
            y_bins = np.linspace(y_min, y_max, n_grid + 1)
            H_hist, _, _ = np.histogram2d(x_clean, y_clean, bins=(x_bins, y_bins))
            p_grid = H_hist.flatten() / np.sum(H_hist)
            p_nonzero = p_grid[p_grid > 0]
            h_spatial = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        else:
            h_spatial = 0.0
    else:
        h_spatial = 0.0
        
    # 2. Switch Cadence (switches per second)
    sylls = sub_syll['syllable'].values
    valid_sylls = sylls[~pd.isna(sylls)]
    if len(valid_sylls) > 1:
        switches = np.sum(valid_sylls[1:] != valid_sylls[:-1])
        duration_sec = len(valid_sylls) / 80.025
        cadence = float(switches / max(duration_sec, 1e-3))
    else:
        cadence = 0.0
        
    # 3. Turning Chirality Ratio (|omega| / (v + eps))
    vel = sub_df['velocity_px_s'].values
    omega = np.abs(sub_df['angular_velocity'].values)
    valid_kin = ~np.isnan(vel) & ~np.isnan(omega)
    if np.sum(valid_kin) > 10:
        chirality = float(np.mean(omega[valid_kin] / (vel[valid_kin] + 10.0)))
    else:
        chirality = 0.0
        
    # 4. Sequence Transition Entropy
    if len(valid_sylls) > 2:
        unique_sylls, syll_indices = np.unique(valid_sylls, return_inverse=True)
        K_s = len(unique_sylls)
        if K_s > 1:
            trans_mat = np.zeros((K_s, K_s), dtype=np.float64)
            for i in range(len(syll_indices) - 1):
                trans_mat[syll_indices[i], syll_indices[i+1]] += 1.0
            row_sums = trans_mat.sum(axis=1, keepdims=True)
            trans_probs = np.divide(trans_mat, row_sums, out=np.zeros_like(trans_mat), where=row_sums > 0)
            p_nonzero = trans_probs[trans_probs > 0]
            h_seq = float(-np.mean(np.sum(trans_probs * np.log2(trans_probs + 1e-12), axis=1)))
        else:
            h_seq = 0.0
    else:
        h_seq = 0.0
        
    return {
        "spatial_entropy": h_spatial,
        "cadence": cadence,
        "chirality": chirality,
        "sequence_entropy": h_seq
    }


def evaluate_linear_probes(
    X_features: np.ndarray, 
    y_targets: np.ndarray, 
    n_splits: int = 5, 
    random_state: int = 42
) -> float:
    """
    Evaluates cross-validated R² score of a Ridge regression probe on held-out test windows.
    """
    if len(X_features) < n_splits * 2:
        return 0.0
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_preds = np.zeros_like(y_targets)
    
    for train_idx, test_idx in kf.split(X_features):
        clf = Ridge(alpha=1.0)
        clf.fit(X_features[train_idx], y_targets[train_idx])
        y_preds[test_idx] = clf.predict(X_features[test_idx])
        
    return float(r2_score(y_targets, y_preds))


def run_scalar_probing_battery(
    X_strategy_props: np.ndarray, 
    X_raw_sylls: np.ndarray, 
    scalar_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Runs the linear probing battery across all 4 macroscopic scalars,
    comparing Strategy Representation against:
      - Raw Syllables (100-D Bag of Words)
      - Random Null Control (Permuted features)
    """
    results = {}
    
    # Random Null Features
    X_null = np.random.randn(*X_strategy_props.shape)
    
    for scalar_name, y in scalar_dict.items():
        r2_null = evaluate_linear_probes(X_null, y)
        r2_raw = evaluate_linear_probes(X_raw_sylls, y)
        r2_strat = evaluate_linear_probes(X_strategy_props, y)
        
        results[scalar_name] = {
            "r2_null": r2_null,
            "r2_raw_syllables": r2_raw,
            "r2_strategy": r2_strat,
            "compression_advantage": r2_strat - r2_raw
        }
        
    return results
