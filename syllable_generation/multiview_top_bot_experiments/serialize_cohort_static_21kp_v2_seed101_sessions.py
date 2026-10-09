#!/usr/bin/env python3
"""
serialize_cohort_static_21kp_v2_seed101_sessions.py
===================================================
Weaves the newly discovered syllables and latents from the production model:
  cohort_static_21kp_v2 / model_21kp_static_50ar_300slds_seed101
into the master Karen project SessionRecord objects.

Input:
  - Baseline dataset: syllable_generation/data/sessions_karen_project_w_sylls.pkl (110 sessions)
  - Results H5:       cohort_static_21kp_v2/model_21kp_static_50ar_300slds_seed101/results.h5 (338 segments)
  - Frame Index:      cohort_static_21kp_v2/video_frame_indexes.npz

Output:
  - syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl
"""

import os
import sys
import re
import time
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_PKL = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls.pkl"
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
MODEL_DIR = EXP_DIR / "cohort_static_21kp_v2/model_21kp_static_50ar_300slds_seed101"
RESULTS_H5 = MODEL_DIR / "results.h5"
VFI_PATH = EXP_DIR / "cohort_static_21kp_v2/video_frame_indexes.npz"

OUTPUT_PKL = EXP_DIR / "sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl"


def main():
    t0 = time.time()
    print("=" * 80)
    print("🚀 SERIALIZING COHORT STATIC 21KP V2 SEED 101 SESSION RECORDS")
    print(f"Base Pickle:  {BASE_PKL}")
    print(f"Results H5:   {RESULTS_H5}")
    print(f"VFI File:     {VFI_PATH}")
    print(f"Target PKL:   {OUTPUT_PKL}")
    print("=" * 80)

    # 1. Load VFI and group segments by session ID
    print("\n[1/4] Loading Video Frame Indexes (VFI)...", flush=True)
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k].astype(np.int64) for k in z.files}
    print(f"  Loaded {len(vfi_dict)} segments.")

    # Group segment keys by base session id
    segs_by_sid = defaultdict(list)
    for seg_k in vfi_dict.keys():
        sid = re.sub(r"_seg\d+$", "", seg_k)
        segs_by_sid[sid].append(seg_k)
    print(f"  Indexed segments across {len(segs_by_sid)} base sessions.")

    # 2. Load model results from results.h5
    print("\n[2/4] Reading results.h5 segment data...", flush=True)
    model_data = {}
    with h5py.File(RESULTS_H5, "r") as f_res:
        for seg_k in vfi_dict.keys():
            if seg_k in f_res:
                grp = f_res[seg_k]
                model_data[seg_k] = {
                    "syllable": grp["syllable"][:],
                    "latent_state": grp["latent_state"][:],
                    "centroid": grp["centroid"][:],
                    "heading": grp["heading"][:],
                }
    print(f"  Cached {len(model_data)} segment results.")

    # 3. Load baseline session records
    print("\n[3/4] Loading baseline session records...", flush=True)
    with open(BASE_PKL, "rb") as f:
        sessions = pickle.load(f)
    print(f"  Loaded {len(sessions)} baseline sessions.")

    # 4. Weave new syllables and latents into each session record
    print("\n[4/4] Weaving 21kp V2 syllables into session timelines...", flush=True)
    updated_count = 0
    total_frames_modeled = 0

    for idx, s in enumerate(sessions):
        sid = s.session_id
        if sid not in segs_by_sid:
            print(f"  [{idx+1}/{len(sessions)}] [WARNING] No modeled segments for {sid}!")
            continue

        n_frames = len(s.all_data_df)
        timeline_index = pd.RangeIndex(0, n_frames)

        # Initialize timeline arrays
        syll_arr = np.full(n_frames, -1, dtype=np.int64)
        latents_arr = np.full((n_frames, 10), np.nan, dtype=np.float32)
        c0_arr = np.full(n_frames, np.nan, dtype=np.float32)
        c1_arr = np.full(n_frames, np.nan, dtype=np.float32)
        h_arr = np.full(n_frames, np.nan, dtype=np.float32)

        # Fill from each segment
        seg_keys = sorted(segs_by_sid[sid])
        sess_modeled_frames = 0
        for seg_k in seg_keys:
            if seg_k not in model_data:
                continue
            seg_res = model_data[seg_k]
            vfi = vfi_dict[seg_k]

            # Bounds check
            valid_vfi_mask = (vfi >= 0) & (vfi < n_frames)
            vfi_clean = vfi[valid_vfi_mask]
            
            syll_arr[vfi_clean] = seg_res["syllable"][valid_vfi_mask]
            latents_arr[vfi_clean, :] = seg_res["latent_state"][valid_vfi_mask, :]
            c0_arr[vfi_clean] = seg_res["centroid"][valid_vfi_mask, 0]
            c1_arr[vfi_clean] = seg_res["centroid"][valid_vfi_mask, 1]
            h_arr[vfi_clean] = seg_res["heading"][valid_vfi_mask]
            sess_modeled_frames += len(vfi_clean)

        total_frames_modeled += sess_modeled_frames

        # Syllables series (Int64 with NA for unmodeled frames)
        syll_series = pd.Series(
            np.where(syll_arr >= 0, syll_arr, pd.NA),
            index=timeline_index,
            dtype="Int64"
        )

        # Update s.syllables DataFrame
        syll_df_dict = {
            "syllable": syll_series,
        }
        for d in range(10):
            syll_df_dict[f"latent_state_{d}"] = latents_arr[:, d]
        syll_df_dict["centroid_0"] = c0_arr
        syll_df_dict["centroid_1"] = c1_arr
        syll_df_dict["heading"] = h_arr
        s.syllables = pd.DataFrame(syll_df_dict, index=timeline_index)

        # Update s.all_data_df: preserve old columns but overwrite syllable and onset
        # Keep old syllable as 'k240k_syllable' for backward comparison reference
        if "syllable" in s.all_data_df.columns:
            s.all_data_df["k240k_syllable"] = s.all_data_df["syllable"]
        s.all_data_df["syllable"] = syll_series
        s.all_data_df["static_21kp_v2_syllable"] = syll_series

        # Recalculate onset transitions
        is_valid = ~pd.isna(syll_series.to_numpy())
        vals = syll_series.to_numpy()
        onset_mask = np.zeros(n_frames, dtype=bool)
        if np.any(is_valid):
            first_idx = np.where(is_valid)[0][0]
            onset_mask[first_idx] = True
            for t in range(first_idx + 1, n_frames):
                if is_valid[t] and (not is_valid[t-1] or vals[t] != vals[t-1]):
                    onset_mask[t] = True
        s.all_data_df["onset"] = onset_mask

        # Tag model metadata
        s.kpms_model_name = "model_21kp_static_50ar_300slds_seed101"
        s.kpms_seed = 101
        s.kpms_cohort = "cohort_static_21kp_v2"

        updated_count += 1
        if (idx + 1) % 20 == 0 or (idx + 1) == len(sessions):
            print(f"  [{idx+1}/{len(sessions)}] Processed {sid} ({sess_modeled_frames:,}/{n_frames:,} frames modeled)", flush=True)

    print(f"\nSuccessfully woven {updated_count}/{len(sessions)} sessions.")
    print(f"Total modeled frames: {total_frames_modeled:,}")

    # 5. Save enriched pickle
    print(f"\nSaving enriched pickle to {OUTPUT_PKL}...", flush=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = OUTPUT_PKL.stat().st_size / (1024 ** 3)
    elapsed = time.time() - t0
    print(f"✅ Saved {OUTPUT_PKL.name} ({size_gb:.2f} GB) in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
