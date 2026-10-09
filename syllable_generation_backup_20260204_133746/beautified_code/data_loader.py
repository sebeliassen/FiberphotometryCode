# data_loader.py
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging # Use logging

# Import from our project structure
import config
from session import Session
# Do NOT import photometry_processor here to avoid circular dependencies if photometry_processor needs Session

# Configure logger for this module
logger = logging.getLogger(__name__)

def find_sessions(frame_ranges_dict: Optional[Dict[str, Tuple[int, int]]]) -> List[Session]:
    """
    Scans the PHOTOMETRY_BASE_DIR, parses folder names, finds associated video/data files,
    and creates a list of Session objects, validating essential files and frame range info.

    Args:
        frame_ranges_dict: Dictionary mapping canonical video keys to (start, end) frame tuples.

    Returns:
        A list of validated Session objects.
    """
    sessions: List[Session] = []
    logger.info(f"--- Finding Sessions ---")
    logger.info(f"Scanning trial directories in: {config.PHOTOMETRY_BASE_DIR}")

    if frame_ranges_dict is None:
        logger.error("Frame ranges data not provided to find_sessions. Cannot validate sessions.")
        return []
    if not frame_ranges_dict:
         logger.warning("Frame ranges dictionary provided is empty. Sessions might lack frame ranges.")

    processed_canonical_keys = set() # Keep track of processed keys to avoid duplicates if structure is odd

    for trial_dir in sorted(config.PHOTOMETRY_BASE_DIR.iterdir()):
        if not trial_dir.is_dir():
            continue
        if not re.match(r'^(?:TRIAL_|T\d+_)', trial_dir.name, flags=re.IGNORECASE):
            logger.debug(f"Skipping non-trial folder: {trial_dir.name}")
            continue


        try:
            # Expecting format like TRIAL_XXX_MouseID1.MouseID2.MouseID3.MouseID4
            parts = trial_dir.name.split('_', 1)[1].split('.')
            arenas = ["arena1", "arena2", "arena3", "arena4"] # Assumed fixed mapping
        except IndexError:
            logger.warning(f"Skipping folder with unexpected name format: {trial_dir.name}")
            continue

        # Find essential CSVs for the whole trial directory
        # Use more specific glob patterns if possible
        photometry_csv = next(trial_dir.glob("photometry_data_combined*.csv"), None)
        center_ts_csv = next(trial_dir.glob("CenterTopCam_AllTimestampsCombined*.csv"), None)

        if not photometry_csv:
            logger.warning(f"Skipping trial {trial_dir.name}: Missing photometry_data_combined CSV.")
            continue
        if not center_ts_csv:
            logger.warning(f"Skipping trial {trial_dir.name}: Missing CenterTopCam_AllTimestampsCombined CSV.")
            continue

        # --- Determine Timestamp from filenames ---
        # Prioritize timestamp CSV, then photometry CSV
        ts = None
        ts_match_ts_csv = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", center_ts_csv.name)
        ts_match_photom = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}(?:[._])\d{2}(?:[._])\d{2})", photometry_csv.name)

        if ts_match_ts_csv:
            ts = ts_match_ts_csv.group(1)
            logger.debug(f"  Timestamp from Timestamp CSV: {ts}")
        elif ts_match_photom:
            ts = ts_match_photom.group(1).replace('.', '_').replace('-', '_') # Normalize format if needed
            logger.debug(f"  Timestamp from Photometry CSV: {ts}")
        else:
            logger.warning(f"Skipping trial {trial_dir.name}: Cannot determine timestamp from CSV filenames.")
            continue

        # --- Create Sessions for each Mouse within the trial ---
        for idx, part in enumerate(parts):
            part = part.strip()
            if part.lower() == "e" or not part.isdigit() or idx >= len(arenas):
                # Skip if empty ('e'), not a number, or more parts than arenas
                continue

            try:
                mouse_id = int(part)
            except ValueError:
                logger.warning(f"Invalid mouse ID format '{part}' in {trial_dir.name}. Skipping.")
                continue

            arena = arenas[idx]
            arena_suffix = config.CENTER_SUFFIX_MAP.get(arena)
            if not arena_suffix:
                logger.warning(f"Arena suffix not found for arena '{arena}' (Mouse {mouse_id}). Skipping.")
                continue

            # Get mouse metadata
            if mouse_id not in config.MOUSE_TO_REGION:
                logger.warning(f"Mouse ID {mouse_id} not found in MOUSE_TO_REGION mapping. Skipping.")
                continue
            # Assuming format [("L-Region", "R-Region", "Genotype")]
            try:
                 left_reg, right_reg, genotype = config.MOUSE_TO_REGION[mouse_id][0]
            except (IndexError, TypeError):
                 logger.warning(f"Invalid region/genotype data format for Mouse ID {mouse_id}. Skipping.")
                 continue


            # --- Define the CANONICAL key used for matching across data types ---
            # Format: Arena1234_CenterTopCamVid_{Timestamp}_{ArenaSuffix}
            canonical_base_key = f"Arena1234_CenterTopCamVid_{ts}_{arena_suffix}"

            if canonical_base_key in processed_canonical_keys:
                 logger.warning(f"Duplicate canonical key '{canonical_base_key}' detected. Skipping second instance.")
                 continue
            processed_canonical_keys.add(canonical_base_key)

            logger.debug(f"  Processing Mouse {mouse_id} ({genotype}) in {arena} -> Key: {canonical_base_key}")

            # --- Construct Expected File Paths ---
            behavior_video_path = config.BEHAVIOR_VIDEO_DIR / f"{canonical_base_key}.avi"
            syllable_video_path = config.SYLLABLE_VIDEO_DIR / f"{canonical_base_key}.mp4" # Expected output from syllable_generator

            # Find 3D Pose Video (more robustly using the key components)
            # Example Pattern: *{ts}*{arena_suffix}*_3dpoints*.mp4
            pose_3d_video_pattern = f"*{ts}*{arena_suffix}*_3dpoints*.mp4"
            pose_3d_matches = sorted(list(config.POSE_3D_VIDEO_DIR.glob(pose_3d_video_pattern))) # Sort for consistency
            pose_3d_video_path = pose_3d_matches[0] if pose_3d_matches else None
            if len(pose_3d_matches) > 1:
                logger.warning(f"    Multiple 3D pose videos found for pattern '{pose_3d_video_pattern}'. "
                               f"Using first match: {pose_3d_video_path.name}")

            # --- Find Frame Range using DIRECT lookup with CANONICAL key ---
            frame_range = frame_ranges_dict.get(canonical_base_key)

            # --- Create Session Object ---
            session = Session(
                trial_folder=trial_dir.name,
                mouse_id=mouse_id,
                timestamp=ts,
                arena=arena,
                left_region=left_reg,
                right_region=right_reg,
                genotype=genotype,
                photometry_csv_path=photometry_csv,
                center_timestamp_csv_path=center_ts_csv,
                behavior_video_path=behavior_video_path,
                pose_3d_video_path=pose_3d_video_path,
                syllable_video_path=syllable_video_path, # Store expected path
                frame_range_original=frame_range # Get from dict (can be None)
            )

            # --- Validate Session ---
            session.is_valid = True
            missing_files = []
            if not behavior_video_path.exists():
                missing_files.append(f"Behavior AVI ({behavior_video_path.name})")
                session.is_valid = False
            if not pose_3d_video_path or not pose_3d_video_path.exists():
                missing_files.append(f"3D Pose MP4 (pattern: {pose_3d_video_pattern})")
                session.is_valid = False
            # Syllable video existence is checked later IF needed (during generation)
            # Frame range is crucial for processing
            if not frame_range:
                missing_files.append("Frame range info (from syllable/kpms processing)")
                session.is_valid = False

            if session.is_valid:
                logger.info(f"    --> VALID Session found: {canonical_base_key}")
                sessions.append(session)
            else:
                logger.warning(f"    --> INVALID Session: {canonical_base_key}. Missing: {'; '.join(missing_files)}")


    logger.info(f"--- Found {len(sessions)} valid sessions meeting criteria. ---")
    return sessions

# --- load_photometry_data function is REMOVED from here ---
# Its logic is now in photometry_processor.load_raw_photometry_and_merge_timestamps