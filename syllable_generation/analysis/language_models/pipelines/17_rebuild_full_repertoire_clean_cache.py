"""
Rebuild Full-Repertoire Clean Dataset Cache (All Active Mouse Frames)
Located at: syllable_generation/analysis/language_models/pipelines/17_rebuild_full_repertoire_clean_cache.py

1. Loads raw sessions from jonathan_amph_sessions_v2_w_sylls.pkl.
2. Filters out empty arena pre-entry frames (trimming to first_valid_index of centroid_x).
3. Preserves ALL active mouse frames (Pre-Injection Baseline Exploration + Post-Injection Response).
4. Tracks post_inj_start_win index for each session to allow strict post-injection evaluation.
5. Saves clean dataset cache to prep_amph_postural_gmm_k3_transition.pkl.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_transition_matrix(syllable_series: pd.Series, n_syllables: int = 100) -> np.ndarray:
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
    return counts / row_sums

def main():
    print("=" * 75)
    print("REBUILDING FULL-REPERTOIRE CLEAN DATASET CACHE (All Active Frames)")
    print("=" * 75)
    
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)
        
    print(f"Loaded {len(sessions)} raw session objects")
    
    window_size_frames = 10240  # ~128s at 80.025 Hz
    stride_frames = 800         # ~10s stride at 80.025 Hz
    n_syllables = 100
    
    all_win_inputs = []
    session_window_meta = []
    
    clean_sessions_count = 0
    total_active_windows = 0
    
    for s_idx, session in enumerate(sessions):
        df = session.all_data_df
        sid = session.session_id
        meta_dict = getattr(session, "metadata", {})
        
        gt = meta_dict.get("genotype", "WT")
        drug = meta_dict.get("drug", "Saline")
        cohort_4way = f"{gt}_{drug}"
        
        inj_frame = meta_dict.get("injection_frame", 0)
        first_valid = df["centroid_x"].first_valid_index() if "centroid_x" in df.columns else 0
        first_valid = first_valid if first_valid is not None else 0
        
        vid_path = "unknown"
        if hasattr(session, "video_refs_df") and session.video_refs_df is not None and not session.video_refs_df.empty:
            vid_path = str(session.video_refs_df.iloc[0]["path"])
            
        # Active mouse frames start at first_valid
        sub_df = df.iloc[first_valid:].copy()
        n_sub = len(sub_df)
        
        if n_sub < window_size_frames:
            print(f"  [{s_idx+1:2d}/{len(sessions)}] Session {sid[:35]}: ⚠️ SKIPPED (Active duration too short)")
            continue
            
        win_count = 0
        post_inj_win_start = 0
        post_inj_found = False
        
        for win_start in range(0, n_sub - window_size_frames + 1, stride_frames):
            win_end = win_start + window_size_frames
            abs_win_start_frame = first_valid + win_start
            
            # Check if this window starts after the injection frame
            if not post_inj_found and abs_win_start_frame >= inj_frame:
                post_inj_win_start = win_count
                post_inj_found = True
                
            win_data = sub_df.iloc[win_start:win_end]
            trans_mat = compute_transition_matrix(win_data["syllable"], n_syllables=n_syllables)
            all_win_inputs.append(trans_mat.flatten())
            win_count += 1
            
        if not post_inj_found:
            post_inj_win_start = max(0, win_count // 2)
            
        session_window_meta.append((
            win_count,
            first_valid,
            first_valid + (win_count - 1) * stride_frames + window_size_frames,
            sid,
            vid_path,
            cohort_4way,
            sid,
            post_inj_win_start,
            inj_frame
        ))
        clean_sessions_count += 1
        total_active_windows += win_count
        n_post_win = win_count - post_inj_win_start
        print(f"  [{s_idx+1:2d}/{len(sessions)}] Session {sid[:35]}: {win_count:2d} total clean wins ({n_post_win:2d} post-inj) | Cohort: {cohort_4way:<14}")

    cache_pkl_path = os.path.join(CACHE_DIR, "prep_amph_postural_gmm_k3_transition.pkl")
    with open(cache_pkl_path, "wb") as f:
        pickle.dump((None, all_win_inputs, session_window_meta), f)
        
    print("\n" + "=" * 75)
    print(f"SUCCESS: Rebuilt full-repertoire clean dataset cache!")
    print(f"Total sessions: {clean_sessions_count} | Total windows: {total_active_windows}")
    print(f"Cache saved to: {cache_pkl_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
