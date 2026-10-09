"""
Rebuild Clean Post-Injection Dataset Cache
Located at: syllable_generation/analysis/language_models/pipelines/15_rebuild_clean_post_injection_cache.py

1. Loads raw sessions from jonathan_amph_sessions_v2_w_sylls.pkl.
2. Embeds true 4-way metadata cohort tags (WT_Saline, WT_Amphetamine, KI_Saline, KI_Amphetamine).
3. Restricts sliding windows strictly to POST-INJECTION active-mouse frames (frame >= max(first_valid_index, injection_frame)).
4. Calculates 1-step transition probability matrices P(S_t | S_{t-1}).
5. Saves clean cache to prep_amph_postural_gmm_k3_transition.pkl.
"""

import os
import pickle
import sys

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
    print("REBUILDING CLEAN POST-INJECTION DATASET CACHE (True 4-Way Cohorts)")
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
    total_post_windows = 0

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

        # Start at max(first_valid, inj_frame) to guarantee active mouse post-injection
        start_frame = max(first_valid, inj_frame)

        vid_path = "unknown"
        if hasattr(session, "video_refs_df") and session.video_refs_df is not None and not session.video_refs_df.empty:
            vid_path = str(session.video_refs_df.iloc[0]["path"])

        sub_df = df.iloc[start_frame:].copy()
        n_sub = len(sub_df)

        if n_sub < window_size_frames:
            print(f"  [{s_idx+1:2d}/{len(sessions)}] Session {sid[:35]}: ⚠️ SKIPPED (Post-inj duration too short)")
            continue

        win_count = 0
        for win_start in range(0, n_sub - window_size_frames + 1, stride_frames):
            win_end = win_start + window_size_frames
            win_data = sub_df.iloc[win_start:win_end]
            trans_mat = compute_transition_matrix(win_data["syllable"], n_syllables=n_syllables)
            all_win_inputs.append(trans_mat.flatten())
            win_count += 1

        abs_start_frame = start_frame
        abs_end_frame = start_frame + (win_count - 1) * stride_frames + window_size_frames

        session_window_meta.append((
            win_count,
            abs_start_frame,
            abs_end_frame,
            sid,
            vid_path,
            cohort_4way,
            sid
        ))
        clean_sessions_count += 1
        total_post_windows += win_count
        print(f"  [{s_idx+1:2d}/{len(sessions)}] Session {sid[:35]}: {win_count:2d} post-inj windows | Cohort: {cohort_4way:<14}")

    cache_pkl_path = os.path.join(CACHE_DIR, "prep_amph_postural_gmm_k3_transition.pkl")
    with open(cache_pkl_path, "wb") as f:
        pickle.dump((None, all_win_inputs, session_window_meta), f)

    print("\n" + "=" * 75)
    print("SUCCESS: Rebuilt clean post-injection dataset cache!")
    print(f"Total sessions: {clean_sessions_count} | Total windows: {total_post_windows}")
    print(f"Cache saved to: {cache_pkl_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
