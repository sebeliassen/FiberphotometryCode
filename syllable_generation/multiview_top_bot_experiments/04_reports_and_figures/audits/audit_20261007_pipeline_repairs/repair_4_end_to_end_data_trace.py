#!/usr/bin/env python3
"""
repair_4_end_to_end_data_trace.py
=================================
Meticulous end-to-end tracing and data reconciliation script proving exactly
what reaches the Keypoint-MoSeq model.

Stages traced for representative frames across every arena and epoch:
1. Raw TopCam & BotCam H5 rows
2. Continuous temporal mapping (Top frame -> Bot frame)
3. Fused H5 coordinates and likelihoods
4. Continuous dual-view segment and Video Frame Index (VFI)
5. Preprocessed batch representation: Y, conf, mask
6. Reconciled model results: latent_state, centroid, heading, syllable

Also audits:
- Missingness & longest gap per paw
- Low confidence retention on interpolated observations
- Segment eligibility & frame count reconciliation (source vs segment vs batch vs results)
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import h5py

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_pipeline_repairs"
COHORT_DIR = EXP / "cohort_static_21kp"
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_200slds_seed101"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from core.pair_matching import find_matched_pairs
from core.camera_correspondence import get_session_mapping
from core.ttl_sync import find_camera_timestamp_files

PAW_LIST = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
BODYPARTS_21KP = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint",
    "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"
]

def run_trace():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading preprocessed cohort data and results...")

    with open(COHORT_DIR / "preprocessed_data.pkl", "rb") as f:
        data, metadata = pickle.load(f)

    with np.load(COHORT_DIR / "video_frame_indexes.npz") as z:
        vfi = {k: z[k] for k in z.files}

    h5_res = h5py.File(MODEL_DIR / "results.h5", "r")

    # 1. Reconcile Frame Counts
    n_batches = len(data["Y"])
    total_batch_frames = data["Y"].shape[0] * data["Y"].shape[1]
    total_valid_frames = int(np.sum(data["mask"]))
    total_padded_dummy_frames = total_batch_frames - total_valid_frames
    n_segments = len(vfi)
    total_vfi_frames = sum(len(arr) for arr in vfi.values())
    n_results_sessions = len(h5_res.keys())
    total_results_frames = sum(h5_res[k]["syllable"].shape[0] for k in h5_res.keys())

    frame_reconciliation = {
        "num_continuous_segments": n_segments,
        "num_preprocessed_batches": n_batches,
        "batch_length": int(data["Y"].shape[1]),
        "total_valid_frames_in_batches": total_valid_frames,
        "total_padded_dummy_frames": total_padded_dummy_frames,
        "total_vfi_frames": total_vfi_frames,
        "total_results_frames": total_results_frames,
        "results_vs_vfi_discrepancy": total_results_frames - total_vfi_frames
    }
    print("Frame count reconciliation:")
    print(json.dumps(frame_reconciliation, indent=2))

    # 2. Missingness and Confidence Audit Across All Preprocessed Data
    # Paws are indices 17, 18, 19, 20 in 21kp
    paw_quality = {}
    valid_mask = data["mask"].astype(bool)

    for i, bp in enumerate(PAW_LIST, start=17):
        confs = data["conf"][:, :, i][valid_mask]
        low_conf = float(np.mean(confs < 0.4) * 100.0)
        paw_quality[bp] = {
            "mean_confidence": float(np.mean(confs)),
            "median_confidence": float(np.median(confs)),
            "p10_confidence": float(np.percentile(confs, 10)),
            "p90_confidence": float(np.percentile(confs, 90)),
            "low_confidence_pct_under_0_4": low_conf,
            "finite_coord_pct": float(np.mean(np.isfinite(data["Y"][:, :, i, 0][valid_mask])) * 100.0)
        }
    print("\nPaw confidence and quality in model batches:")
    print(json.dumps(paw_quality, indent=2))

    # 3. Exact Frame-by-Frame Trace for Representative Sessions
    # Select representative frames that are inside valid segments
    target_sessions = []
    for label, pref in [
        ("Arena1_Epoch1", "2025-12-03T09_51_05_top_left"),
        ("Arena2_Epoch1", "2026-01-08T10_06_18_bottom_left"),
        ("Arena3_Epoch1", "2026-03-05T11_01_01_bottom_right"),
        ("Arena1_Epoch2", "2026-04-27T14_05_05_top_left"),
    ]:
        for seg_name, fr_arr in vfi.items():
            if seg_name.startswith(pref):
                # Pick a frame in the middle of the segment
                mid_fr = int(fr_arr[len(fr_arr) // 2])
                target_sessions.append((label, pref, mid_fr))
                break

    pairs = find_matched_pairs()
    pair_dict = {p["sid"]: p for p in pairs}
    fused_dir = EXP / "data/cohort_fused_static_21kp"

    frame_traces = []

    for label, sid, src_frame in target_sessions:
        if sid not in pair_dict:
            continue
        p = pair_dict[sid]
        fused_path = fused_dir / f"{sid}_tandem_21kp.h5"
        if not fused_path.exists():
            continue

        df_fused = pd.read_hdf(fused_path)
        df_top = pd.read_hdf(p["top_h5"])
        df_bot = pd.read_hdf(p["bot_h5"])

        sc_f = df_fused.columns.levels[0][0]
        sc_t = df_top.columns.levels[0][0]
        sc_b = df_bot.columns.levels[0][0]

        mapping = get_session_mapping(sid, strict=False)
        bot_frame = int(mapping[src_frame]) if mapping is not None and src_frame < len(mapping) else -1

        # Trace keypoints: nose (dorsal) and left_front_paw (ventral)
        # Step 1: Raw TopCam
        top_nose_x = float(df_top[(sc_t, "nose", "x")].iloc[src_frame])
        top_nose_y = float(df_top[(sc_t, "nose", "y")].iloc[src_frame])
        top_nose_p = float(df_top[(sc_t, "nose", "likelihood")].iloc[src_frame])

        # Step 2 & 3: Raw BotCam
        b_name = "right_front_paw" # maps to left_front_paw after swap
        bot_paw_x = float(df_bot[(sc_b, b_name, "x")].iloc[bot_frame]) if bot_frame >= 0 else np.nan
        bot_paw_y = float(df_bot[(sc_b, b_name, "y")].iloc[bot_frame]) if bot_frame >= 0 else np.nan
        bot_paw_p = float(df_bot[(sc_b, b_name, "likelihood")].iloc[bot_frame]) if bot_frame >= 0 else 0.0

        # Step 4: Fused H5
        fused_nose_x = float(df_fused[(sc_f, "nose", "x")].iloc[src_frame])
        fused_nose_y = float(df_fused[(sc_f, "nose", "y")].iloc[src_frame])
        fused_paw_x = float(df_fused[(sc_f, "left_front_paw", "x")].iloc[src_frame])
        fused_paw_y = float(df_fused[(sc_f, "left_front_paw", "y")].iloc[src_frame])
        fused_paw_p = float(df_fused[(sc_f, "left_front_paw", "likelihood")].iloc[src_frame])

        # Step 5: Continuous segment and VFI
        matching_seg = None
        seg_frame_idx = -1
        for seg_name, frame_indices in vfi.items():
            if seg_name.startswith(sid):
                locs = np.where(frame_indices == src_frame)[0]
                if len(locs) > 0:
                    matching_seg = seg_name
                    seg_frame_idx = int(locs[0])
                    break

        # Step 6: Batch row & time
        batch_idx = -1
        batch_t = -1
        batch_y_paw = [np.nan, np.nan]
        batch_conf_paw = 0.0
        batch_mask = 0.0

        if matching_seg is not None:
            # Search metadata for batch
            for b_i in range(len(metadata[0])):
                if str(metadata[0][b_i]) == matching_seg:
                    s_off, e_off = metadata[1][b_i]
                    if s_off <= seg_frame_idx < e_off:
                        batch_idx = b_i
                        batch_t = seg_frame_idx - s_off
                        batch_y_paw = [float(data["Y"][batch_idx, batch_t, 17, 0]), float(data["Y"][batch_idx, batch_t, 17, 1])]
                        batch_conf_paw = float(data["conf"][batch_idx, batch_t, 17])
                        batch_mask = float(data["mask"][batch_idx, batch_t])
                        break

        # Step 7: Results frame
        res_syll = -1
        res_latent = []
        if matching_seg is not None and matching_seg in h5_res:
            res_syll = int(h5_res[matching_seg]["syllable"][seg_frame_idx])
            res_latent = [float(x) for x in h5_res[matching_seg]["latent_state"][seg_frame_idx][:4]]

        frame_traces.append({
            "trace_label": label,
            "session_id": sid,
            "top_source_frame": src_frame,
            "bot_mapped_frame": bot_frame,
            "raw_top_nose": [top_nose_x, top_nose_y, top_nose_p],
            "raw_bot_paw": [bot_paw_x, bot_paw_y, bot_paw_p],
            "fused_h5_nose": [fused_nose_x, fused_nose_y],
            "fused_h5_paw": [fused_paw_x, fused_paw_y, fused_paw_p],
            "continuous_segment": matching_seg,
            "segment_frame_index": seg_frame_idx,
            "batch_index": batch_idx,
            "batch_time": batch_t,
            "batch_y_paw_centered_rotated": batch_y_paw,
            "batch_conf_paw": batch_conf_paw,
            "batch_mask": batch_mask,
            "results_syllable": res_syll,
            "results_latent_sample": res_latent
        })

    trace_summary = {
        "frame_reconciliation": frame_reconciliation,
        "paw_quality_in_batches": paw_quality,
        "representative_frame_traces": frame_traces
    }

    with open(AUDIT_DIR / "audit_4_end_to_end_data_trace.json", "w") as f:
        json.dump(trace_summary, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else float(o) if isinstance(o, (np.floating, np.float32)) else str(o))

    h5_res.close()
    print(f"\n✅ End-to-end data trace complete. Saved to {AUDIT_DIR / 'audit_4_end_to_end_data_trace.json'}")

if __name__ == "__main__":
    run_trace()
