#!/usr/bin/env python3
"""
audit_cohort_continuous_sync.py
=============================================================================
Comprehensive Cohort Audit: Static Scalar Offset vs. Continuous True Sync.

Evaluates all 113 production sessions in video_frame_indexes.npz to determine:
1. Exact count and percentage of retained training frames affected by mid-session drops.
2. Magnitude distribution of timing errors (1 frame, 2-5 frames, >5 frames).
3. Sessions with the highest discrepancy rates.
=============================================================================
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

HDD_DIR = Path("/mnt/hdd/karen_project")
BASE_DIR = Path(__file__).resolve().parent
VFI_PATH = BASE_DIR / "kpms_tandem_cohort_production" / "video_frame_indexes.npz"
CACHE_PATH = BASE_DIR / "core" / "ttl_cohort_offsets.json"
OUT_CSV = BASE_DIR / "cohort_continuous_sync_audit_results.csv"

def compute_continuous_mapping(top_csv: Path, bot_csv: Path, max_jitter_ms: float = 12.5):
    """
    Computes per-frame nearest-neighbor camera correspondence using Cam_Bonsai_Timestamp.
    Returns:
        mapping: 1D np.ndarray of shape (N_top,) mapping TopCam frame -> BotCam frame (or -1 if dropped/out of bounds)
        jitter_ms: 1D np.ndarray of temporal delta in ms
    """
    df_t = pd.read_csv(top_csv, usecols=["Cam_Bonsai_Timestamp"])
    df_b = pd.read_csv(bot_csv, usecols=["Cam_Bonsai_Timestamp"])
    
    t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy()
    b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy()
    
    # Nearest neighbor search
    idx = np.searchsorted(b_bon, t_bon)
    idx = np.clip(idx, 0, len(b_bon) - 1)
    left = np.maximum(0, idx - 1)
    closer_left = np.abs(b_bon[left] - t_bon) < np.abs(b_bon[idx] - t_bon)
    best_bot = np.where(closer_left, left, idx)
    
    dt_ms = np.abs(b_bon[best_bot] - t_bon) * 1000.0
    valid = (dt_ms <= max_jitter_ms) & (t_bon >= b_bon[0] - 0.008) & (t_bon <= b_bon[-1] + 0.008)
    mapping = np.where(valid, best_bot, -1)
    
    return mapping, dt_ms, len(df_t), len(df_b)

def main():
    t0 = time.time()
    print("=" * 80)
    print("🚀 AUDITING COHORT CONTINUOUS TIMESTAMP CORRESPONDENCE (113 SESSIONS)")
    print("=" * 80)
    
    with open(CACHE_PATH) as f:
        ttl_cache = json.load(f)
        
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}
        
    session_keys = sorted(vfi_dict.keys())
    print(f"Loaded {len(session_keys)} sessions from VFI.")
    
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
            print(f"[WARN] No TTL info for {sid}, skipping.")
            continue
            
        top_csv = Path(info["top_csv"])
        bot_csv = Path(info["bot_csv"])
        if not top_csv.exists() or not bot_csv.exists():
            print(f"[WARN] Missing CSV files for {sid}, skipping.")
            continue
            
        static_offset = int(info.get("bot0_in_top", 0) - info.get("top0_in_bot", 0))
        
        # 1. Compute true continuous mapping
        mapping_true, jitter_ms, n_top, n_bot = compute_continuous_mapping(top_csv, bot_csv)
        
        # 2. Compare against retained training frames (VFI)
        vfi_frames = vfi_dict[sid]
        n_retained = len(vfi_frames)
        total_retained_cohort += n_retained
        
        # Filter valid indices within bounds of TopCam
        valid_vfi = vfi_frames[(vfi_frames >= 0) & (vfi_frames < n_top)]
        
        # Static mapping for these retained frames
        bot_static = valid_vfi - static_offset
        bot_true = mapping_true[valid_vfi]
        
        # Discrepancy definition: true mapping differs from static mapping
        discrepant_mask = (bot_true != bot_static)
        n_discrepant = int(np.sum(discrepant_mask))
        total_discrepant_cohort += n_discrepant
        
        # Dropped in BotCam (bot_true == -1)
        dropped_mask = (bot_true == -1)
        n_dropped = int(np.sum(dropped_mask))
        total_dropped_bot_cohort += n_dropped
        
        # Quantify error magnitude for shifted frames
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
    print("📊 COHORT AUDIT SUMMARY: IMPACT ON RETAINED MODEL TRAINING DATA")
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
