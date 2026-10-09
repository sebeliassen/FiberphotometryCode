#!/usr/bin/env python3
"""
deep_trace_and_export.py
=============================================================================
Forensic analysis script for the two flagged tiles:
1. Row 2, Col 4: 2026-04-28T13_21_05_bottom_left (Model 123294, Top 151503)
2. Row 3, Col 3: 2025-12-03T16_38_48_bottom_right (Model 22585, Top 49419)

Extracts:
- Exact timestamps (FP3002, Bonsai, Basler) and drift/rate calibration.
- DLC coordinates, likelihoods, and centroid definitions.
- Direct video frame decoding via OpenCV VideoCapture (independent verification).
- High-resolution CSV trace and diagnostic overlays.
=============================================================================
"""

import os
import sys
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import h5py

EXP_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments")
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.pair_matching import find_matched_pairs
from core.ttl_sync import find_camera_timestamp_files, get_session_ttl_offset
from core.camera_correspondence import compute_session_camera_mapping, top_to_bot_frame_continuous

OUTPUT_DIR = EXP_DIR / "diagnostic_two_tiles_export"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed101"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"

TOP_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")
BOT_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files")
CHAMBER_TO_ARENA = {'top_left': 'Arena1', 'bottom_left': 'Arena2', 'bottom_right': 'Arena3', 'top_right': 'Arena4'}

with np.load(VFI_PATH) as z:
    vfi_dict = {k: z[k] for k in z.files}

pairs = find_matched_pairs()
pairs_by_sid = {p['sid']: p for p in pairs}

def calibrate_clocks_rate_aware(top_csv, bot_csv):
    """
    Fits both offset and clock-rate drift between TopCam and BotCam Basler timestamps.
    """
    df_t = pd.read_csv(top_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
    df_b = pd.read_csv(bot_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
    
    t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy()
    b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy()
    t_bas = df_t["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
    b_bas = df_b["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
    
    # Bonsai-based alignment anchors across multiple points (e.g. every 10k frames)
    # where Bonsai timestamp difference is minimal
    anchors_top = []
    anchors_bot = []
    step = 5000
    for i in range(1000, len(t_bon) - 1000, step):
        t_val = t_bon[i]
        b_idx = int(np.searchsorted(b_bon, t_val))
        if 0 < b_idx < len(b_bon):
            # check nearest
            if abs(b_bon[b_idx - 1] - t_val) < abs(b_bon[b_idx] - t_val):
                b_idx -= 1
            dt = abs(b_bon[b_idx] - t_val)
            if dt < 0.005: # within 5ms host time
                anchors_top.append(t_bas[i])
                anchors_bot.append(b_bas[b_idx])
                
    anchors_top = np.array(anchors_top)
    anchors_bot = np.array(anchors_bot)
    
    if len(anchors_top) > 5:
        # Fit linear regression: b_bas = slope * t_bas + intercept
        # Or t_bas = slope * b_bas + intercept
        p = np.polyfit(anchors_top, anchors_bot, deg=1)
        slope, intercept = p[0], p[1]
        ppm = (slope - 1.0) * 1e6
    else:
        slope, intercept, ppm = 1.0, b_bas[0] - t_bas[0], 0.0
        
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ppm_drift": float(ppm),
        "n_anchors": len(anchors_top),
        "df_t": df_t,
        "df_b": df_b
    }

cases = [
    {
        "tile": "Row 2, col 4 (April Exemplar)",
        "sid": "2026-04-28T13_21_05_bottom_left",
        "model_onset": 123294,
        "top_onset": 151503,
        "duration": 53
    },
    {
        "tile": "Row 3, col 3 (December Exemplar)",
        "sid": "2025-12-03T16_38_48_bottom_right",
        "model_onset": 22585,
        "top_onset": 49419,
        "duration": 53
    },
    {
        "tile": "Row 2, col 4 (January Exemplar)",
        "sid": "2026-01-09T12_19_47_top_left",
        "model_onset": 84730,
        "top_onset": 94728,
        "duration": 50
    },
    {
        "tile": "Row 3, col 3 (January Exemplar)",
        "sid": "2026-01-09T09_41_59_bottom_left",
        "model_onset": 102060,
        "top_onset": 121740,
        "duration": 50
    }
]

with h5py.File(RESULTS_PATH, "r") as res_file:
    for case in cases:
        sid = case["sid"]
        tile_name = case["tile"]
        model_onset = case["model_onset"]
        duration = case["duration"]
        
        print("=" * 80)
        print(f"ANALYZING {tile_name}: {sid}")
        print("=" * 80)
        
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
        ts, chamber = m.group(1), m.group(2)
        arena = CHAMBER_TO_ARENA[chamber]
        
        pair = pairs_by_sid[sid]
        top_vids = list(TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
        bot_vids = list(BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))
        top_csv, bot_csv = find_camera_timestamp_files(ts, arena)
        
        top_vid = str(top_vids[0])
        bot_vid = str(bot_vids[0])
        top_h5 = str(pair["top_h5"])
        bot_h5 = str(pair["bot_h5"])
        
        sid_key = sid if sid in vfi_dict else f"{sid}_tandem_23kp"
        vfi = vfi_dict[sid_key]
        pre, post = 10, 40
        clip_len = pre + post
        
        # Model frames and VFI top raw frames
        model_frames = np.arange(model_onset - pre, model_onset + post)
        top_raw_frames = vfi[model_frames]
        
        # Load DLC
        df_top = pd.read_hdf(top_h5)
        df_bot = pd.read_hdf(bot_h5)
        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]
        
        # Clock calibration
        # Continuous mapping and clock calibration from core.camera_correspondence
        mapping_cont, jitter_cont, calib_info = compute_session_camera_mapping(top_csv, bot_csv)
        print(f"  Clock calibration: {calib_info['n_anchors_retained']} anchors, drift = {calib_info['ppm_drift']:+.2f} ppm, mean_res = {calib_info['anchor_mean_res_ms']:.4f} ms")
        
        # Static TTL offset
        ttl_info = get_session_ttl_offset(sid)
        static_offset = int(ttl_info.get("bot0_in_top", 0) - ttl_info.get("top0_in_bot", 0))
        
        # Results.h5 model latents
        grp = res_file[sid_key]
        c_top_all = grp["centroid"][:]
        h_top_all = grp["heading"][:]
        sylls_all = grp["syllable"][:]
        
        # Open video captures directly
        cap_top = cv2.VideoCapture(top_vid)
        cap_bot = cv2.VideoCapture(bot_vid)
        
        records = []
        
        # Prepare video writers for uncropped diagnostic inspection
        uncropped_tandem_path = OUTPUT_DIR / f"{sid}_uncropped_tandem.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_tandem = None
        
        df_t = pd.read_csv(top_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
        b_bas_arr = df_b["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
        slope = calib_info["slope"]
        intercept = calib_info["intercept"]
        
        for t in range(clip_len):
            mf = model_frames[t]
            tf = int(top_raw_frames[t])
            syll = int(sylls_all[mf])
            
            bf_selected = top_to_bot_frame_continuous(sid, tf)
            bf_static = max(0, tf - static_offset)
            
            t_bas = float(df_t["Cam_Basler_Timestamp"].iloc[tf])
            t_bon = float(df_t["Cam_Bonsai_Timestamp"].iloc[tf])
            
            # Predict BotCam Basler index via rate calibration
            b_bas_pred = (t_bas - intercept) / slope
            idx_b = int(np.searchsorted(b_bas_arr, b_bas_pred))
            if idx_b >= len(b_bas_arr):
                idx_b = len(b_bas_arr) - 1
            if idx_b > 0 and abs(b_bas_arr[idx_b - 1] - b_bas_pred) < abs(b_bas_arr[idx_b] - b_bas_pred):
                idx_b -= 1
            bf_rate_aware = idx_b
            
            # Signed residual for displayed frame
            if bf_selected >= 0 and bf_selected < len(b_bas_arr):
                b_sel_bas = b_bas_arr[bf_selected]
                b_sel_cal = slope * b_sel_bas + intercept
                residual_displayed_ms = float((b_sel_cal - t_bas) / 1e6)
                is_valid = bool(abs(residual_displayed_ms) <= 8.35)
                b_bon = float(df_b["Cam_Bonsai_Timestamp"].iloc[bf_selected])
                bf = bf_selected
            else:
                residual_displayed_ms = np.nan
                is_valid = False
                b_bon = np.nan
                bf = -1
                
            b_stat_bas = b_bas_arr[min(len(b_bas_arr)-1, bf_static)]
            b_stat_cal = slope * b_stat_bas + intercept
            residual_static_ms = float((b_stat_cal - t_bas) / 1e6)
            
            b_rate_bas = b_bas_arr[bf_rate_aware]
            b_rate_cal = slope * b_rate_bas + intercept
            residual_rate_aware_ms = float((b_rate_cal - t_bas) / 1e6)
            
            # DLC coordinates on BotCam
            if bf >= 0:
                b_bas = float(df_b["Cam_Basler_Timestamp"].iloc[bf])
                bnose_x = float(df_bot[bot_scorer, "nose_tip", "x"].iloc[bf])
                bnose_y = float(df_bot[bot_scorer, "nose_tip", "y"].iloc[bf])
                bnose_p = float(df_bot[bot_scorer, "nose_tip", "likelihood"].iloc[bf])
                
                btail_x = float(df_bot[bot_scorer, "tail_base", "x"].iloc[bf])
                btail_y = float(df_bot[bot_scorer, "tail_base", "y"].iloc[bf])
                btail_p = float(df_bot[bot_scorer, "tail_base", "likelihood"].iloc[bf])
                
                blpaw_p = float(df_bot[bot_scorer, "left_front_paw", "likelihood"].iloc[bf])
                brpaw_p = float(df_bot[bot_scorer, "right_front_paw", "likelihood"].iloc[bf])
            else:
                b_bas = np.nan
                bnose_x, bnose_y, bnose_p = np.nan, np.nan, 0.0
                btail_x, btail_y, btail_p = np.nan, np.nan, 0.0
                blpaw_p, brpaw_p = 0.0, 0.0
            
            # Model centroid & heading
            mc = c_top_all[mf]
            mh = h_top_all[mf]
            
            # Read decoded frames directly from disk using VideoCapture
            cap_top.set(cv2.CAP_PROP_POS_FRAMES, tf)
            ret_t, frame_t = cap_top.read()
            
            if bf >= 0:
                cap_bot.set(cv2.CAP_PROP_POS_FRAMES, bf)
                ret_b, frame_b = cap_bot.read()
            else:
                ret_b, frame_b = False, None
            
            # Overlay frame numbers and DLC points on uncropped frames
            diag_t = frame_t.copy() if ret_t else np.zeros((404, 404, 3), dtype=np.uint8)
            diag_b = frame_b.copy() if ret_b else np.zeros((972, 972, 3), dtype=np.uint8)
            
            cv2.putText(diag_t, f"TopCam Frame: {tf} (Model {mf})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(diag_t, f"Syllable: {syll}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(diag_b, f"BotCam Frame: {bf} (Resid={residual_displayed_ms:+.3f} ms, Static={bf_static}, StaticResid={residual_static_ms:+.1f} ms)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(diag_b, f"Nose: ({bnose_x:.1f}, {bnose_y:.1f}) p={bnose_p:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if bnose_p > 0.8 else (0, 0, 255), 2)
            cv2.putText(diag_b, f"Tail: ({btail_x:.1f}, {btail_y:.1f}) p={btail_p:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if btail_p > 0.8 else (0, 0, 255), 2)
            
            # Draw DLC points
            if bnose_p > 0.1:
                cv2.circle(diag_b, (int(round(bnose_x)), int(round(bnose_y))), 6, (0, 255, 0) if bnose_p > 0.8 else (0, 0, 255), -1)
                cv2.putText(diag_b, "N", (int(round(bnose_x)) + 8, int(round(bnose_y))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if btail_p > 0.1:
                cv2.circle(diag_b, (int(round(btail_x)), int(round(btail_y))), 6, (255, 0, 0) if btail_p > 0.8 else (0, 0, 255), -1)
                cv2.putText(diag_b, "T", (int(round(btail_x)) + 8, int(round(btail_y))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
            # Resize for side-by-side uncropped tandem display
            target_h = 480
            scale_t = target_h / diag_t.shape[0]
            scale_b = target_h / diag_b.shape[0]
            dt_scaled = cv2.resize(diag_t, (int(round(diag_t.shape[1] * scale_t)), target_h))
            db_scaled = cv2.resize(diag_b, (int(round(diag_b.shape[1] * scale_b)), target_h))
            
            tandem_uncropped = np.concatenate([dt_scaled, db_scaled], axis=1)
            
            if writer_tandem is None:
                writer_tandem = cv2.VideoWriter(str(uncropped_tandem_path), fourcc, 15, (tandem_uncropped.shape[1], tandem_uncropped.shape[0]))
            writer_tandem.write(tandem_uncropped)
            
            records.append({
                "clip_frame_idx": t,
                "model_frame": mf,
                "syllable": syll,
                "top_raw_frame": tf,
                "bot_selected_frame": bf,
                "bot_raw_frame_rate_aware": bf_rate_aware,
                "bot_raw_frame_static": bf_static,
                "residual_displayed_ms": residual_displayed_ms,
                "residual_rate_aware_ms": residual_rate_aware_ms,
                "residual_static_ms": residual_static_ms,
                "selected_calib_identity": "rate_aware_basler_crystal",
                "is_valid": is_valid,
                "top_basler_ns": t_bas,
                "bot_basler_ns": b_bas,
                "top_bonsai_s": t_bon,
                "bot_bonsai_s": b_bon,
                "bot_nose_x": bnose_x,
                "bot_nose_y": bnose_y,
                "bot_nose_p": bnose_p,
                "bot_tail_x": btail_x,
                "bot_tail_y": btail_y,
                "bot_tail_p": btail_p,
                "bot_lpaw_p": blpaw_p,
                "bot_rpaw_p": brpaw_p,
                "model_centroid_x": mc[0],
                "model_centroid_y": mc[1],
                "model_heading": mh
            })
            
        if writer_tandem is not None:
            writer_tandem.release()
        cap_top.release()
        cap_bot.release()
        
        # Save CSV trace
        df_trace = pd.DataFrame(records)
        trace_csv = OUTPUT_DIR / f"{sid}_per_frame_trace.csv"
        df_trace.to_csv(trace_csv, index=False)
        print(f"  Saved trace CSV: {trace_csv} ({len(df_trace)} frames)")
        print(f"  Saved uncropped video: {uncropped_tandem_path}")
        
        # Print summary of onset frame
        onset_rec = df_trace.iloc[pre]
        print(f"  ONSET (frame index {pre}):")
        print(f"    Top frame: {onset_rec['top_raw_frame']}, Bot frame: {onset_rec['bot_selected_frame']}")
        print(f"    Bot Nose: ({onset_rec['bot_nose_x']:.1f}, {onset_rec['bot_nose_y']:.1f}) p={onset_rec['bot_nose_p']:.3f}")
        print(f"    Bot Tail: ({onset_rec['bot_tail_x']:.1f}, {onset_rec['bot_tail_y']:.1f}) p={onset_rec['bot_tail_p']:.3f}")

print("\nDONE forensic deep trace!")
