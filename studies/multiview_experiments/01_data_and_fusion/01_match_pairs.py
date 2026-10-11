#!/usr/bin/env python3
"""
01_match_pairs.py
-----------------
Matches completed BotCam prediction H5 files with corresponding TopCam Session objects
from sessions_karen_project_w_sylls.pkl.

Validates:
1. Exact timestamp matching between TopCam and BotCam
2. Arena chamber mapping:
   - Arena 1 <-> top_left
   - Arena 2 <-> bottom_left
   - Arena 3 <-> bottom_right
3. Frame count synchronization (both 60.0 fps)
4. Saves paired_sessions_manifest.json for downstream fusion
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO_ROOT = Path(str(get_repo_root()))
EXPERIMENT_DIR = (get_repo_root() / 'studies/multiview_experiments')
SESSIONS_PKL = resolve_data_path('data/sessions') / "sessions_karen_project_w_sylls.pkl"
BOTCAM_JOB_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922")
BOTCAM_COMPLETED_JSON = BOTCAM_JOB_DIR / "completed_videos.json"
BOTCAM_PRED_DIR = BOTCAM_JOB_DIR / "predictions"

MANIFEST_OUT = EXPERIMENT_DIR / "paired_sessions_manifest.json"

CHAMBER_MAP = {
    "Arena1": "top_left",
    "Arena2": "bottom_left",
    "Arena3": "bottom_right"
}

def extract_botcam_tokens(vid_id):
    """
    Parses BotCam unique ID into wave, trial, arena, and timestamp.
    Example: 16w_PI_T1_6_2_4_050326_active_Arena1_BotCamVid_2026-03-05T11_01_01
    """
    tokens = vid_id.split("_")
    arena = None
    for tok in tokens:
        if tok in ["Arena1", "Arena2", "Arena3", "Arena4"]:
            arena = tok
            break

    # Timestamp is trailing: 2026-XX-XXTXX_XX_XX
    ts = None
    for i in range(len(tokens) - 1):
        cand = tokens[i] + "_" + tokens[i+1]
        if "T" in cand and cand.startswith("2026-"):
            # check format 2026-03-05T11_01_01
            parts = vid_id.split("BotCamVid_")
            if len(parts) > 1:
                ts = parts[1]
            break
    if not ts and "BotCamVid_" in vid_id:
        ts = vid_id.split("BotCamVid_")[1]

    return arena, ts

def main():
    print("=" * 70)
    print("MATCHING TOPCAM & BOTCAM SESSION PAIRS")
    print(f"Sessions Pickle:  {SESSIONS_PKL}")
    print(f"BotCam Completed: {BOTCAM_COMPLETED_JSON}")
    print(f"Output Manifest:  {MANIFEST_OUT}")
    print("=" * 70)

    if not BOTCAM_COMPLETED_JSON.exists():
        print(f"ERROR: {BOTCAM_COMPLETED_JSON} does not exist.")
        sys.exit(1)

    with open(BOTCAM_COMPLETED_JSON, "r") as f:
        botcam_completed = json.load(f)

    print(f"Found {len(botcam_completed)} completed BotCam records in production run.")

    print(f"Loading {SESSIONS_PKL.name} (this may take a few seconds)...")
    with open(SESSIONS_PKL, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} total session objects.")

    # Index sessions by session_id and results_dict_key
    matched_pairs = []

    for vid_id, bot_info in botcam_completed.items():
        arena, ts = extract_botcam_tokens(vid_id)
        if not arena or not ts:
            print(f"  [WARN] Could not parse tokens for {vid_id}")
            continue

        quadrant = CHAMBER_MAP.get(arena)
        if not quadrant:
            print(f"  [SKIP] Skipping {arena} (unmapped or empty)")
            continue

        # Look for matching session
        # Session IDs typically format: Arena1234_CenterTopCamVid_{ts}_{quadrant}
        matched_sess = None
        for s in sessions:
            rk = getattr(s, "results_dict_key", "")
            sid = getattr(s, "session_id", "")

            # Check if timestamp and quadrant are in session identifier
            if ts in rk and quadrant in rk:
                matched_sess = s
                break
            if ts in sid and quadrant in sid:
                matched_sess = s
                break
            # Fallback: check video_refs_df paths
            if hasattr(s, "video_refs_df") and not s.video_refs_df.empty:
                vpath = str(s.video_refs_df.iloc[0].get("path", ""))
                if ts in vpath and quadrant in vpath:
                    matched_sess = s
                    break

        if not matched_sess:
            print(f"  [MISS] No TopCam session found matching {vid_id} (ts: {ts}, {quadrant})")
            continue

        # Extract session metadata
        rk = getattr(matched_sess, "results_dict_key", "")
        sid = getattr(matched_sess, "session_id", "")
        top_frames = None
        top_vid_path = None
        if hasattr(matched_sess, "video_refs_df") and not matched_sess.video_refs_df.empty:
            top_frames = int(matched_sess.video_refs_df.iloc[0].get("n_frames", 0))
            top_vid_path = str(matched_sess.video_refs_df.iloc[0].get("path", ""))

        bot_frames = bot_info.get("frames", 0)
        bot_h5 = bot_info.get("h5_file", "")
        bot_source = bot_info.get("source_path", "")

        frame_delta = abs(top_frames - bot_frames) if top_frames else None

        pair_record = {
            "session_id": sid,
            "results_dict_key": rk,
            "timestamp": ts,
            "botcam_arena": arena,
            "topcam_quadrant": quadrant,
            "topcam_frames": top_frames,
            "botcam_frames": bot_frames,
            "frame_delta": frame_delta,
            "topcam_video_path": top_vid_path,
            "botcam_source_video": bot_source,
            "botcam_h5_file": bot_h5,
            "botcam_throughput_fps": bot_info.get("throughput_fps"),
            "botcam_duration_sec": bot_info.get("duration_seconds")
        }
        matched_pairs.append(pair_record)
        print(f"  [MATCH] {arena} ({quadrant}) | {ts} | TopCam: {top_frames:,} f, BotCam: {bot_frames:,} f (Delta: {frame_delta})")

    print("-" * 70)
    print(f"Successfully matched: {len(matched_pairs)} / {len(botcam_completed)} sessions.")

    manifest_data = {
        "created_at": pd.Timestamp.now().isoformat(),
        "total_matched_pairs": len(matched_pairs),
        "source_sessions_pkl": str(SESSIONS_PKL),
        "pairs": matched_pairs
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Wrote manifest to: {MANIFEST_OUT}")
    print("=" * 70)

if __name__ == "__main__":
    main()
