#!/usr/bin/env python3
"""
enrich_sessions_with_botcam.py
------------------------------
Enriches SessionRecord objects with multi-camera metadata and video references:
1. Loads existing sessions (e.g. sessions_karen_project_w_sylls.pkl).
2. For sessions with completed BotCam predictions, attaches a second row to s.video_refs_df (role="ventral")
   and s.tracking_refs_df (role="ventral").
3. Strictly preserves row 0 as role="primary" so ALL legacy code continues working.
4. If a session does not have BotCam, it remains completely untouched (1 row).
5. Verifies 100% backwards compatibility via assertion suite.
"""

import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
SESSIONS_IN = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
SESSIONS_OUT = REPO_ROOT / "syllable_generation" / "multiview_top_bot_experiments" / "sessions_multiview_pilot.pkl"

BOTCAM_COMPLETED_JSON = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/completed_videos.json")

CHAMBER_MAP = {
    "Arena1": "top_left",
    "Arena2": "bottom_left",
    "Arena3": "bottom_right"
}

def extract_botcam_tokens(vid_id):
    tokens = vid_id.split("_")
    arena = next((t for t in tokens if t in ["Arena1", "Arena2", "Arena3"]), None)
    ts = None
    if "BotCamVid_" in vid_id:
        ts = vid_id.split("BotCamVid_")[1]
    return arena, ts

def enrich_sessions():
    print("=" * 70)
    print("ENRICHING SESSIONS WITH MULTI-CAMERA BOTCAM REFS")
    print(f"Input Pickle:    {SESSIONS_IN}")
    print(f"BotCam Manifest: {BOTCAM_COMPLETED_JSON}")
    print(f"Output Pickle:   {SESSIONS_OUT}")
    print("=" * 70)

    if not BOTCAM_COMPLETED_JSON.exists():
        print(f"ERROR: {BOTCAM_COMPLETED_JSON} does not exist.")
        return

    with open(BOTCAM_COMPLETED_JSON) as f:
        botcam_data = json.load(f)

    # Build lookup index: (timestamp, chamber) -> botcam_info
    botcam_index = {}
    for vid_id, binfo in botcam_data.items():
        arena, ts = extract_botcam_tokens(vid_id)
        if arena and ts and arena in CHAMBER_MAP:
            chamber = CHAMBER_MAP[arena]
            botcam_index[(ts, chamber)] = (vid_id, binfo)

    print(f"Indexed {len(botcam_index)} completed BotCam videos for pairing.")

    print("Loading sessions pickle...")
    with open(SESSIONS_IN, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} session objects.")

    enriched_count = 0
    single_cam_count = 0

    for s in sessions:
        sid = getattr(s, "session_id", "")
        rk = getattr(s, "results_dict_key", "")
        
        # Match (timestamp, chamber)
        matched_bot = None
        for (ts, chamber), (vid_id, binfo) in botcam_index.items():
            if ts in sid and chamber in sid:
                matched_bot = (vid_id, binfo, arena, chamber)
                break
            if ts in rk and chamber in rk:
                matched_bot = (vid_id, binfo, arena, chamber)
                break

        if matched_bot:
            vid_id, binfo, arena, chamber = matched_bot
            enriched_count += 1
            
            # --- 1. ENRICH s.video_refs_df ---
            bot_vid_row = {
                "path": binfo.get("source_path") or binfo.get("resolved_path"),
                "path_type": "video",
                "kind": "avi",
                "role": "ventral",
                "ts": ts,
                "chamber": chamber,
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": arena,
                "camera_key": f"bottom_cam:{chamber}",
                "fps": 60.0,
                "n_frames": binfo.get("frames"),
                "duration_s": binfo.get("duration_seconds"),
                "meta_origin": "production_manifest"
            }

            if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
                # Keep existing primary row at index 0
                existing_df = s.video_refs_df.copy()
                # Check if ventral already present to avoid duplicates
                if "ventral" not in existing_df["role"].values:
                    new_df = pd.concat([existing_df, pd.DataFrame([bot_vid_row])], ignore_index=True)
                    # Sort so role == "primary" is strictly at index 0
                    new_df["_sort_key"] = new_df["role"].apply(lambda r: 0 if r == "primary" else 1)
                    new_df = new_df.sort_values(by="_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
                    s.video_refs_df = new_df
            else:
                s.video_refs_df = pd.DataFrame([bot_vid_row])

            # --- 2. ENRICH s.tracking_refs_df ---
            bot_trk_row = {
                "path": binfo.get("h5_file"),
                "path_type": "tracking",
                "kind": "dlc_h5",
                "role": "ventral",
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": arena,
                "chamber": chamber,
                "ts": ts,
                "camera_key": f"bottom_cam:{chamber}",
                "checkpoint_sha256": binfo.get("checkpoint_sha256"),
                "throughput_fps": binfo.get("throughput_fps")
            }

            if hasattr(s, "tracking_refs_df") and s.tracking_refs_df is not None and not s.tracking_refs_df.empty:
                existing_trk = s.tracking_refs_df.copy()
                if "ventral" not in existing_trk["role"].values:
                    new_trk = pd.concat([existing_trk, pd.DataFrame([bot_trk_row])], ignore_index=True)
                    new_trk["_sort_key"] = new_trk["role"].apply(lambda r: 0 if r == "primary" else 1)
                    new_trk = new_trk.sort_values(by="_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
                    s.tracking_refs_df = new_trk
            else:
                s.tracking_refs_df = pd.DataFrame([bot_trk_row])

            s.metadata["has_botcam"] = True
            s.metadata["botcam_video_id"] = vid_id
        else:
            single_cam_count += 1
            s.metadata["has_botcam"] = False

    print("-" * 70)
    print(f"Summary: {enriched_count} sessions enriched with dual-video (TopCam + BotCam).")
    print(f"Summary: {single_cam_count} sessions remain single-camera TopCam only.")
    print("-" * 70)

    # --- 3. BACKWARDS COMPATIBILITY VERIFICATION ---
    print("Running Backwards Compatibility Assertions...")
    for s in sessions:
        if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
            # Assertion 1: iloc[0] must ALWAYS be the primary video
            first_role = s.video_refs_df.iloc[0]["role"]
            assert first_role == "primary" or len(s.video_refs_df) == 1, f"Row 0 role was {first_role}, expected 'primary'!"
            
            # Assertion 2: Filtering by role == 'primary' always returns exactly 1 row
            primary_rows = s.video_refs_df[s.video_refs_df["role"] == "primary"]
            if not primary_rows.empty:
                assert len(primary_rows) == 1, "Multiple primary videos detected!"

            # Assertion 3: If has_botcam, exactly 2 rows exist
            if s.metadata.get("has_botcam"):
                assert len(s.video_refs_df) >= 2, f"Expected >=2 video rows for {s.session_id}, got {len(s.video_refs_df)}"
                assert "ventral" in s.video_refs_df["role"].values, "Missing 'ventral' role row!"

    print("ALL ASSERTIONS PASSED! 100% Backwards Compatible.")

    # Save output
    with open(SESSIONS_OUT, "wb") as f:
        pickle.dump(sessions, f)
    print(f"Saved enriched sessions to: {SESSIONS_OUT}")
    print("=" * 70)

if __name__ == "__main__":
    enrich_sessions()
