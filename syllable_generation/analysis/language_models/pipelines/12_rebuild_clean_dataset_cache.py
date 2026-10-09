"""
Rebuild Clean Dataset Cache (Filtering Pre-Entry & Empty Arena Frames)
Located at: syllable_generation/analysis/language_models/pipelines/12_rebuild_clean_dataset_cache.py

This script:
1. Loads raw sessions from jonathan_amph_sessions_v2_w_sylls.pkl.
2. Filters out pre-entry / empty-arena NaN frames (where centroid_x is NaN or before first_valid_index).
3. Constructs sliding windows (128s window = 10240 frames @ 80Hz, 10s stride = 800 frames) on clean active mouse frames.
4. Calculates 1-step transition probability matrices P(S_t | S_{t-1}) for each window.
5. Saves clean cache to prep_amph_postural_gmm_k3_transition.pkl.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_transition_matrix(syllable_series: pd.Series, n_syllables: int = 100) -> np.ndarray:
    """
    Computes normalized 1-step transition probability matrix P(S_t | S_{t-1}).
    """
    sylls = pd.to_numeric(syllable_series, errors="coerce").dropna().values.astype(int)
    counts = np.zeros((n_syllables, n_syllables), dtype=float)
    
    if len(sylls) > 1:
        s_prev = sylls[:-1]
        s_curr = sylls[1:]
        valid_mask = (s_prev >= 0) & (s_prev < n_syllables) & (s_curr >= 0) & (s_curr < n_syllables)
        
        for p, c in zip(s_prev[valid_mask], s_curr[valid_mask]):
            counts[p, c] += 1.0
            
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob_matrix = counts / row_sums
    return prob_matrix

def main():
    print("=" * 75)
    print("REBUILDING CLEAN DATASET CACHE (Filtering Pre-Entry & Empty Arena Frames)")
    print("=" * 75)
    
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)
        
    print(f"Loaded {len(sessions)} raw session objects")
    
    # Configuration parameters
    window_size_frames = 10240  # ~128s at 80.025 Hz
    stride_frames = 800         # ~10s stride at 80.025 Hz
    n_syllables = 100
    
    all_win_inputs = []
    session_window_meta = []
    
    clean_sessions_count = 0
    total_trimmed_frames = 0
    
    for s_idx, session in enumerate(sessions):
        df = session.all_data_df
        sid = session.session_id
        
        # Determine cohort group label
        group_label = getattr(session, "group", "unknown")
        if group_label == "unknown" and "group" in df.columns:
            g = df["group"].dropna().iloc[0] if not df["group"].dropna().empty else "unknown"
            group_label = str(g)
            
        # Determine video path
        vid_path = "unknown"
        if hasattr(session, "video_refs_df") and session.video_refs_df is not None and not session.video_refs_df.empty:
            vid_path = str(session.video_refs_df.iloc[0]["path"])
            
        # Trim pre-entry empty arena frames (where centroid_x is NaN)
        first_valid = df["centroid_x"].first_valid_index() if "centroid_x" in df.columns else 0
        last_valid = df["centroid_x"].last_valid_index() if "centroid_x" in df.columns else len(df) - 1
        
        if first_valid is None:
            print(f"  [{s_idx+1}/{len(sessions)}] Session {sid}: ⚠️ SKIPPED (No valid mouse tracking)")
            continue
            
        trimmed_pre = first_valid
        total_trimmed_frames += trimmed_pre
        
        # Subset dataframe to clean active mouse window
        sub_df = df.iloc[first_valid : last_valid + 1].copy()
        n_sub = len(sub_df)
        
        if n_sub < window_size_frames:
            print(f"  [{s_idx+1}/{len(sessions)}] Session {sid}: ⚠️ SKIPPED (Valid duration too short: {n_sub} frames)")
            continue
            
        # Build sliding windows
        win_count = 0
        for win_start in range(0, n_sub - window_size_frames + 1, stride_frames):
            win_end = win_start + window_size_frames
            win_data = sub_df.iloc[win_start:win_end]
            
            # Compute 1-step transition matrix
            trans_mat = compute_transition_matrix(win_data["syllable"], n_syllables=n_syllables)
            all_win_inputs.append(trans_mat.flatten())
            win_count += 1
            
        abs_start_frame = first_valid
        abs_end_frame = first_valid + (win_count - 1) * stride_frames + window_size_frames
        
        session_window_meta.append((
            win_count,
            abs_start_frame,
            abs_end_frame,
            sid,
            vid_path,
            group_label,
            sid
        ))
        clean_sessions_count += 1
        print(f"  [{s_idx+1}/{len(sessions)}] Session {sid[:35]}: {win_count:2d} clean windows | Trimmed pre-entry: {trimmed_pre:5d} frames ({trimmed_pre/80.025:.1f}s)")

    cache_pkl_path = os.path.join(CACHE_DIR, "prep_amph_postural_gmm_k3_transition.pkl")
    with open(cache_pkl_path, "wb") as f:
        pickle.dump((None, all_win_inputs, session_window_meta), f)
        
    print("\n" + "=" * 75)
    print(f"SUCCESS: Rebuilt clean dataset cache with {clean_sessions_count} sessions and {len(all_win_inputs)} total windows")
    print(f"Cache saved to: {cache_pkl_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
