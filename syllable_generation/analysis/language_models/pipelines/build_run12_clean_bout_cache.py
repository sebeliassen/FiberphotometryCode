#!/usr/bin/env python3
"""
build_run12_clean_bout_cache.py
Extracts clean continuous syllable bouts directly from jonathan_amph_sessions_run12_w_sylls.pkl,
fixing tracking gap merging and capturing genuine arena centroid_x, centroid_y.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
sys.path.insert(0, ROOT_DIR)

from syllable_generation.analysis.language_models.lib.preprocessing.bout_features import Behavior2VecEmbedder

RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation", "data", "jonathan_amph_sessions_run12_w_sylls.pkl")
OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models", "output", "strategies_extended")
CACHE_PATH = os.path.join(OUT_DIR, "run12_continuous_bout_cache.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_session_bouts_clean(session, b2v=None):
    df = session.all_data_df
    syll_df = session.syllables

    first_valid = df['centroid_x'].first_valid_index()
    last_valid = df['centroid_x'].last_valid_index()
    if first_valid is None or last_valid is None:
        return []

    sub_df = df.loc[first_valid:last_valid].copy()
    sub_syll = syll_df.loc[first_valid:last_valid].copy()

    valid_mask = ~pd.isna(sub_syll['syllable'])
    if not np.any(valid_mask):
        return []

    sub_df = sub_df[valid_mask].copy()
    sub_syll = sub_syll[valid_mask].copy()

    # Get active sampling rate
    fps = 80.025
    if hasattr(session, "primary_fps"):
        try:
            fps_val = session.primary_fps() if callable(session.primary_fps) else session.primary_fps
            if fps_val and fps_val > 0:
                fps = float(fps_val)
        except Exception:
            pass

    dt = 1.0 / fps
    vel = sub_df['velocity_px_s'].values
    acc = np.gradient(vel, dt)
    sub_df['acceleration'] = acc

    pose_cols = [f'latent_state_{i}' for i in range(10)]
    pose_data = sub_syll[pose_cols].values
    syll_arr = sub_syll['syllable'].values.astype(int)
    frame_indices = sub_df.index.values
    cx_arr = sub_df['centroid_x'].values
    cy_arr = sub_df['centroid_y'].values

    bouts = []
    curr_syll = syll_arr[0]
    start_idx = 0
    n_samples = len(syll_arr)

    for i in range(1, n_samples):
        # Break on syllable change OR frame gap > 2 (tracking drop)
        is_gap = (frame_indices[i] - frame_indices[i - 1]) > 2
        is_diff_syll = (syll_arr[i] != curr_syll)

        if is_diff_syll or is_gap:
            end_idx = i
            dur_frames = end_idx - start_idx

            mean_pose = np.nanmean(pose_data[start_idx:end_idx], axis=0)
            if np.isnan(mean_pose).any():
                mean_pose = np.nan_to_num(mean_pose, nan=0.0)

            mean_v = np.nanmean(vel[start_idx:end_idx])
            mean_w = np.nanmean(np.abs(sub_df['angular_velocity'].values[start_idx:end_idx]))
            mean_a = np.nanmean(np.abs(acc[start_idx:end_idx]))
            mean_cx = np.nanmean(cx_arr[start_idx:end_idx])
            mean_cy = np.nanmean(cy_arr[start_idx:end_idx])

            syll_int = int(curr_syll)
            emb = b2v.get_embedding(syll_int) if b2v is not None else np.zeros(16, dtype=np.float32)

            feat_vec = np.hstack([
                mean_pose,
                np.log1p(max(0.0, mean_v)),
                np.log1p(max(0.0, mean_w)),
                np.log1p(max(0.0, mean_a)),
                np.log1p(dur_frames),
                emb
            ]).astype(np.float32)

            bouts.append({
                "syllable": syll_int,
                "start_frame": frame_indices[start_idx],
                "end_frame": frame_indices[end_idx - 1],
                "duration_frames": dur_frames,
                "duration_sec": dur_frames / fps,
                "mid_frame": (frame_indices[start_idx] + frame_indices[end_idx - 1]) // 2,
                "centroid_x": float(mean_cx),
                "centroid_y": float(mean_cy),
                "mean_velocity": float(mean_v),
                "vector": feat_vec
            })

            curr_syll = syll_arr[i]
            start_idx = i

    if start_idx < n_samples:
        dur_frames = n_samples - start_idx
        mean_pose = np.nanmean(pose_data[start_idx:], axis=0)
        mean_pose = np.nan_to_num(mean_pose, nan=0.0)
        mean_v = np.nanmean(vel[start_idx:])
        mean_w = np.nanmean(np.abs(sub_df['angular_velocity'].values[start_idx:]))
        mean_a = np.nanmean(np.abs(acc[start_idx:]))
        mean_cx = np.nanmean(cx_arr[start_idx:])
        mean_cy = np.nanmean(cy_arr[start_idx:])
        syll_int = int(curr_syll)
        emb = b2v.get_embedding(syll_int) if b2v is not None else np.zeros(16, dtype=np.float32)

        feat_vec = np.hstack([
            mean_pose,
            np.log1p(max(0.0, mean_v)),
            np.log1p(max(0.0, mean_w)),
            np.log1p(max(0.0, mean_a)),
            np.log1p(dur_frames),
            emb
        ]).astype(np.float32)

        bouts.append({
            "syllable": syll_int,
            "start_frame": frame_indices[start_idx],
            "end_frame": frame_indices[n_samples - 1],
            "duration_frames": dur_frames,
            "duration_sec": dur_frames / fps,
            "mid_frame": (frame_indices[start_idx] + frame_indices[n_samples - 1]) // 2,
            "centroid_x": float(mean_cx),
            "centroid_y": float(mean_cy),
            "mean_velocity": float(mean_v),
            "vector": feat_vec
        })

    return bouts

def main():
    print("=" * 80)
    print("BUILDING RUN12 CLEAN BOUT CACHE FROM JONATHAN AMPH RUN12 PKL")
    print(f"Raw Input:   {RAW_PATH}")
    print(f"Destination: {CACHE_PATH}")
    print("=" * 80, flush=True)

    t0 = time.time()
    with open(RAW_PATH, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} SessionRecord objects in {time.time()-t0:.1f}s.")

    # Train Behavior2Vec across all sessions
    syll_seqs = []
    for s in sessions:
        sub_syll = s.syllables['syllable'].dropna()
        if len(sub_syll) > 0:
            syll_seqs.append(sub_syll.values.astype(int))

    print("Fitting Behavior2Vec embedder (16D) on run12 syllable sequences...")
    b2v = Behavior2VecEmbedder(n_components=16, window_size=2)
    b2v.fit(syll_seqs)

    print("Extracting clean continuous bouts across all 30 sessions...")
    all_bouts = []
    for i, s in enumerate(sessions):
        meta = s.metadata if hasattr(s, "metadata") else {}
        aid = meta.get("animal_id", f"s{i}")
        b = extract_session_bouts_clean(s, b2v)
        all_bouts.append(b)
        print(f"  Session {i:02d} ({aid}): {len(b)} bouts extracted.")

    total_bouts = sum(len(b) for b in all_bouts)
    print(f"✓ Total bouts extracted: {total_bouts:,}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"session_bouts": all_bouts, "b2v": b2v}, f)

    print(f"✓ Saved clean cache to: {CACHE_PATH} ({os.path.getsize(CACHE_PATH)/1024/1024:.1f} MB)")
    print(f"✓ Completed in {(time.time()-t0)/60.0:.2f} min.")

if __name__ == "__main__":
    main()
