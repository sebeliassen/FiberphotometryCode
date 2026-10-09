#!/usr/bin/env python3
"""
audit_cohort_basler_calibrated.py
=============================================================================
Calibrated Basler Hardware Crystal Synchronization Audit.

Uses initial FP3002/Bonsai overlap to calibrate cross-camera Basler clock origins,
then computes continuous microsecond-precision hardware exposure correspondence
across all 113 production sessions.
=============================================================================
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
VFI_PATH = BASE_DIR / "kpms_tandem_cohort_production" / "video_frame_indexes.npz"
CACHE_PATH = BASE_DIR / "core" / "ttl_cohort_offsets.json"
OUT_CSV = BASE_DIR / "cohort_basler_calibrated_audit_results.csv"

def compute_calibrated_basler_mapping(top_csv: Path, bot_csv: Path, max_jitter_ms: float = 8.35):
    """
    Computes true exposure-level correspondence using calibrated Cam_Basler_Timestamp.
    """
    df_t = pd.read_csv(top_csv, usecols=["Cam_Basler_Timestamp", "FP3002_System_Timestamp", "Cam_Bonsai_Timestamp"])
    df_b = pd.read_csv(bot_csv, usecols=["Cam_Basler_Timestamp", "FP3002_System_Timestamp", "Cam_Bonsai_Timestamp"])
    
    t_bas = df_t["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
    b_bas = df_b["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
    
    t_fp = df_t["FP3002_System_Timestamp"].to_numpy()
    b_fp = df_b["FP3002_System_Timestamp"].to_numpy()
    
    t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy()
    b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy()
    
    # 1. Cross-camera calibration at frame 0
    # Use FP3002 if running and non-zero at frame 0, else fallback to Bonsai host timestamp
    if abs(t_fp[0] - b_fp[0]) < 100.0 and t_fp[0] > 0 and b_fp[0] > 0:
        # Initial offset from Doric hardware clock
        bot0_in_top = np.searchsorted(t_fp, b_fp[0])
        top0_in_bot = np.searchsorted(b_fp, t_fp[0])
    else:
        bot0_in_top = np.searchsorted(t_bon, b_bon[0])
        top0_in_bot = np.searchsorted(b_bon, t_bon[0])
        
    if top0_in_bot > 0:
        # BotCam started first; TopCam frame 0 corresponds to BotCam frame top0_in_bot
        bas_offset = t_bas[0] - b_bas[top0_in_bot]
    else:
        # TopCam started first; BotCam frame 0 corresponds to TopCam frame bot0_in_top
        bas_offset = t_bas[bot0_in_top] - b_bas[0]
        
    b_bas_cal = b_bas + bas_offset
    
    # 2. Nearest neighbor search on calibrated Basler hardware crystal time
    idx = np.searchsorted(b_bas_cal, t_bas)
    idx = np.clip(idx, 0, len(b_bas_cal) - 1)
    left = np.maximum(0, idx - 1)
    closer_left = np.abs(b_bas_cal[left] - t_bas) < np.abs(b_bas_cal[idx] - t_bas)
    best_bot = np.where(closer_left, left, idx)
    
    dt_ms = np.abs(b_bas_cal[best_bot] - t_bas) / 1e6
    valid = (dt_ms <= max_jitter_ms) & (t_bas >= b_bas_cal[0] - 8e6) & (t_bas <= b_bas_cal[-1] + 8e6)
    mapping = np.where(valid, best_bot, -1)
    
    return mapping, dt_ms, len(df_t), len(df_b)

def main():
    t0 = time.time()
    print("=" * 80)
    print("🔬 AUDITING COHORT VIA CALIBRATED BASLER HARDWARE TIMESTAMPS (113 SESSIONS)")
    print("=" * 80)
    
    with open(CACHE_PATH) as f:
        ttl_cache = json.load(f)
        
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}
        
    session_keys = sorted(vfi_dict.keys())
    results = []
    
    total_retained_cohort = 0
    total_discrepant_cohort = 0
    total_dropped_bot_cohort = 0
    
    error_1f = 0
    error_2_5f = 0
    error_gt5f = 0
    
    for i, sid in enumerate(session_keys):
        clean_sid = sid.replace("_tandem_23kp", "")
        info = ttl_cache.get(sid) or ttl_cache.get(clean_sid)
        if not info:
            continue
            
        top_csv = Path(info["top_csv"])
        bot_csv = Path(info["bot_csv"])
        if not top_csv.exists() or not bot_csv.exists():
            continue
            
        static_offset = int(info.get("bot0_in_top", 0) - info.get("top0_in_bot", 0))
        
        mapping_true, jitter_ms, n_top, n_bot = compute_calibrated_basler_mapping(top_csv, bot_csv)
        
        vfi_frames = vfi_dict[sid]
        n_retained = len(vfi_frames)
        total_retained_cohort += n_retained
        
        valid_vfi = vfi_frames[(vfi_frames >= 0) & (vfi_frames < n_top)]
        bot_static = valid_vfi - static_offset
        bot_true = mapping_true[valid_vfi]
        
        discrepant_mask = (bot_true != bot_static)
        n_discrepant = int(np.sum(discrepant_mask))
        total_discrepant_cohort += n_discrepant
        
        dropped_mask = (bot_true == -1)
        n_dropped = int(np.sum(dropped_mask))
        total_dropped_bot_cohort += n_dropped
        
        shifted_mask = discrepant_mask & (~dropped_mask)
        diff_frames = np.abs(bot_true[shifted_mask] - bot_static[shifted_mask])
        
        s_1f = int(np.sum(diff_frames == 1))
        s_2_5f = int(np.sum((diff_frames >= 2) & (diff_frames <= 5)))
        s_gt5f = int(np.sum(diff_frames > 5))
        
        error_1f += s_1f
        error_2_5f += s_2_5f
        error_gt5f += s_gt5f
        
        pct_discrepant = (n_discrepant / n_retained * 100.0) if n_retained > 0 else 0.0
        
        results.append({
            "session_id": sid,
            "static_offset": static_offset,
            "top_total_frames": n_top,
            "bot_total_frames": n_bot,
            "retained_training_frames": n_retained,
            "discrepant_retained_frames": n_discrepant,
            "pct_discrepant": pct_discrepant,
            "dropped_bot_frames": n_dropped,
            "error_1frame": s_1f,
            "error_2_to_5frames": s_2_5f,
            "error_gt_5frames": s_gt5f,
            "max_abs_frame_error": int(np.max(diff_frames)) if len(diff_frames) > 0 else 0,
            "mean_jitter_ms": float(np.mean(jitter_ms[mapping_true >= 0])),
            "top_csv": str(top_csv),
            "bot_csv": str(bot_csv),
        })
        
        if (i + 1) % 20 == 0 or (i + 1) == len(session_keys):
            print(f"[{i+1:3d}/{len(session_keys)}] Audited {sid[:35]} | Discrepant: {pct_discrepant:5.2f}% ({n_discrepant}/{n_retained})", flush=True)

    df_res = pd.DataFrame(results)
    df_res.sort_values("pct_discrepant", ascending=False, inplace=True)
    df_res.to_csv(OUT_CSV, index=False)
    
    pct_cohort_discrepant = (total_discrepant_cohort / total_retained_cohort * 100.0) if total_retained_cohort > 0 else 0.0
    pct_cohort_dropped = (total_dropped_bot_cohort / total_retained_cohort * 100.0) if total_retained_cohort > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("📊 CALIBRATED BASLER COHORT AUDIT SUMMARY (TRAINING DATA IMPACT)")
    print("=" * 80)
    print(f"Total Sessions Audited:                  {len(df_res)}")
    print(f"Total Retained Training Frames:          {total_retained_cohort:,}")
    print(f"Total Discrepant Retained Frames:        {total_discrepant_cohort:,} ({pct_cohort_discrepant:.3f}%)")
    print(f"  • Completely Dropped in BotCam:        {total_dropped_bot_cohort:,} ({pct_cohort_dropped:.3f}%)")
    print(f"  • Minor 1-Frame Phase Shift:           {error_1f:,} ({error_1f/total_retained_cohort*100:.3f}%)")
    print(f"  • Moderate 2–5 Frame Shift:            {error_2_5f:,} ({error_2_5f/total_retained_cohort*100:.3f}%)")
    print(f"  • Severe >5 Frame Shift:               {error_gt5f:,} ({error_gt5f/total_retained_cohort*100:.3f}%)")
    print(f"\nTop 5 Sessions with Highest Discrepancy Rates:")
    for _, row in df_res.head(5).iterrows():
        print(f"  • {row['session_id'][:35]}: {row['pct_discrepant']:.2f}% discrepant ({row['discrepant_retained_frames']:,}/{row['retained_training_frames']:,}), max err: {row['max_abs_frame_error']}f")
        
    print(f"\nElapsed Audit Time: {time.time() - t0:.2f} seconds")
    print(f"Detailed CSV Report Saved To: {OUT_CSV}")
    print("=" * 80)

if __name__ == "__main__":
    main()
