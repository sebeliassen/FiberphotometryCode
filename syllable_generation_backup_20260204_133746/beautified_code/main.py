# main.py
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import traceback
from typing import Optional, Tuple, List, Dict, Any
import logging # Use logging
import sys # For setting up logging stream

import config
import data_loader # Now only contains find_sessions
import video_utils
import plotting # Now only contains generate_trace_plot_frame and helpers
import syllable_generator
import photometry_processor # New module for photometry loading and filtering
from session import Session

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import os



# --- Basic Logging Setup ---
# Configure logging to output to console
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Output to standard out
log_handler.setFormatter(log_formatter)

# Get the root logger and add the handler
root_logger = logging.getLogger()
root_logger.addHandler(log_handler)
# Set the root logger level (e.g., INFO or DEBUG)
# Use DEBUG for detailed output, INFO for less verbose
# root_logger.setLevel(logging.DEBUG if config.PREVIEW_MODE else logging.INFO) # Example: More logs in preview
root_logger.setLevel(logging.DEBUG)

# Optionally, set specific levels for noisy libraries if needed
# logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # Logger for this main module


def create_composite_video(session: Session, global_ylim: Optional[Tuple[float, float]]):
    """
    Generates either the dopamine-only video or the full 2x2 composite video.
    Uses pre-filtered photometry data stored in session.filtered_photometry_segment_df
    and pre-calculated global y-limits.
    """
    # --- Initial Checks ---
    if not session.is_valid:
        logger.warning(f"Skipping video for {session.base_video_key}: Session marked as invalid.")
        return
    if session.filtered_photometry_segment_df is None or session.filtered_photometry_segment_df.empty:
        logger.error(f"Skipping video for {session.base_video_key}: Missing or empty filtered photometry data.")
        return
    if not session.frame_range_original:
        logger.error(f"Skipping video for {session.base_video_key}: Missing original frame range.")
        return
    if global_ylim is None:
        logger.error(f"Skipping video for {session.base_video_key}: Missing global Y limits.")
        return

    from utils_timestamps import load_per_frame_cam_timestamps_seconds

    session._cam_ts_s_per_frame = load_per_frame_cam_timestamps_seconds(
        session.center_timestamp_csv_path,  # your AllTimestampsCombined path
        target_fps=60.0,
    )

    num_frames_to_generate = session.num_output_frames
    if num_frames_to_generate is None or num_frames_to_generate <= 0:
        logger.warning(f"Skipping video for {session.base_video_key}: Calculated 0 frames to generate.")
        return

    start_idx_orig, end_idx_orig = session.frame_range_original

    # Prefer per-frame camera timestamps in SECONDS if you have them;
    # otherwise fall back to the legacy NS series.
    cam_ts_any = getattr(session, "_cam_ts_s_per_frame", None)
    if cam_ts_any is None:
        cam_ts_any = getattr(session, "_cam_ts_ns", None)

    seg_cam_ts = None
    seg_start_ts = None
    _time_scale = 1.0  # seconds by default

    if cam_ts_any is not None and end_idx_orig is not None:
        seg_cam_ts = np.asarray(cam_ts_any[start_idx_orig : end_idx_orig + 1])
        if seg_cam_ts.size > 0:
            seg_start_ts = float(seg_cam_ts[0])
            # Detect units from the median step
            if seg_cam_ts.size > 3:
                med_dt = float(np.median(np.diff(seg_cam_ts)))
                # If the median step is huge (~1e7-1e8), it's likely nanoseconds
                _time_scale = 1e9 if med_dt > 1e6 else 1.0  # 1e9 means seg_cam_ts is in ns

    start_time = time.time()
    mode_text = "Dopamine Only" if config.DOPAMINE_ONLY_MODE else "Composite"
    preview_text = " (PREVIEW)" if config.PREVIEW_MODE else ""
    log_prefix = f"[{session.base_video_key}] "
    logger.info(f"{log_prefix}--- Creating {mode_text} Video{preview_text} ---")

    # --- Prepare Output Path & Writer ---
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"{log_prefix}Failed to create output directory {config.OUTPUT_DIR}: {e}. Aborting.")
        return

    output_path = config.OUTPUT_DIR / session.output_video_filename
    if output_path.exists():
        logger.info(f"{log_prefix}Output file {output_path.name} already exists. Overwriting.")

    output_fps = config.TARGET_FPS
    output_width = config.QUADRANT_WIDTH if config.DOPAMINE_ONLY_MODE else config.COMPOSITE_WIDTH
    output_height = config.QUADRANT_HEIGHT if config.DOPAMINE_ONLY_MODE else config.COMPOSITE_HEIGHT

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps,
        (output_width, output_height)
    )
    if not writer.isOpened():
        logger.error(f"{log_prefix}Failed to open video writer for {output_path}. Aborting video creation.")
        return

    logger.info(f"{log_prefix}Generating {num_frames_to_generate} frames at {output_fps} FPS -> {output_path.name}")

    # --- Precompute segment camera timestamps (NEW: authoritative time → frame mapping) ---
    seg_cam_ts = None
    seg_start_ts = None
    if getattr(session, "_cam_ts_ns", None) is not None:
        seg_cam_ts = session._cam_ts_ns[start_idx_orig : end_idx_orig + 1]
        if seg_cam_ts is not None and len(seg_cam_ts) > 0:
            seg_start_ts = int(seg_cam_ts[0])
        else:
            logger.warning(f"{log_prefix}Empty seg_cam_ts slice; falling back to FPS mapping.")
            seg_cam_ts = None

    # helper: map output time (sec) → absolute behavior frame index
    def beh_idx_from_time(t_sec: float) -> int:
        """
        Map output time (seconds since segment start) -> behavior frame index,
        using camera per-frame timestamps if available (seconds or nanoseconds).
        Falls back to FPS mapping if not.
        """
        if seg_cam_ts is not None and seg_cam_ts.size > 0 and seg_start_ts is not None:
            # Build target timestamp in the SAME UNITS as seg_cam_ts
            if _time_scale == 1e9:  # seg_cam_ts is in ns
                target_ts = seg_start_ts + float(t_sec * 1e9)
            else:                   # seg_cam_ts is in seconds
                target_ts = seg_start_ts + float(t_sec)

            rel_iloc = photometry_processor.find_nearest_iloc(seg_cam_ts, target_ts)
            rel_iloc = max(0, min(int(rel_iloc), seg_cam_ts.size - 1))
            return start_idx_orig + rel_iloc

        # Fallback: linear FPS mapping (only if no timestamps available)
        return start_idx_orig + int(round(t_sec * config.BEHAVIOR_VIDEO_FPS))

    # --- Initialize resources ---
    caps_to_release: List[cv2.VideoCapture] = []

    try:
        filtered_photometry_df = session.filtered_photometry_segment_df

        if config.DOPAMINE_ONLY_MODE:
            logger.debug(f"{log_prefix}Starting Dopamine Only frame generation loop.")
            for i in tqdm(range(num_frames_to_generate), desc=f"{log_prefix}DopaOnly", unit="frame", leave=False):
                t_sec = i / output_fps
                original_behavior_frame_index = beh_idx_from_time(t_sec)
                original_behavior_frame_index = max(start_idx_orig, min(end_idx_orig, original_behavior_frame_index))

                frame_plot = plotting.generate_trace_plot_frame(
                    filtered_photometry_df,
                    global_ylim,
                    original_behavior_frame_index,
                    session.left_region,
                    session.right_region,
                    start_idx_orig,
                    cam_ts_ns=session._cam_ts_ns,               # stays wired for precise time axis
                    log_prefix=f"{log_prefix}[Frame {i}] "
                )

                if frame_plot.shape[:2] != (output_height, output_width):
                    frame_plot = video_utils.resize_and_pad(frame_plot, output_width, output_height)
                writer.write(frame_plot)

        else:
            logger.debug(f"{log_prefix}Starting Composite frame generation loop.")

            # --- Open Input Video Streams ---
            logger.debug(f"{log_prefix}Opening input video captures...")
            cap_behav = video_utils.open_video_capture(session.behavior_video_path)
            cap_pose  = video_utils.open_video_capture(session.pose_3d_video_path)
            if not session.syllable_video_path or not session.syllable_video_path.exists():
                raise IOError(f"Missing required syllable video: {session.syllable_video_path}")
            cap_syll  = video_utils.open_video_capture(session.syllable_video_path)

            opened_caps = [cap for cap in [cap_behav, cap_pose, cap_syll] if cap and cap.isOpened()]
            caps_to_release.extend(opened_caps)
            if len(opened_caps) != 3:
                missing = []
                if not cap_behav or not cap_behav.isOpened(): missing.append("Behavior")
                if not cap_pose  or not cap_pose.isOpened():  missing.append("3D Pose")
                if not cap_syll  or not cap_syll.isOpened():  missing.append("Syllable")
                raise IOError(f"Failed to open: {', '.join(missing)}")

            # NEW: behavior end handling policy
            behavior_gap_policy = getattr(config, "BEHAVIOR_GAP_POLICY", "hold")  # "hold" or "black"
            beh_total = int(cap_behav.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_behav else 0

            # --- Frame Generation Loop ---
            for i in tqdm(range(num_frames_to_generate), desc=f"{log_prefix}Comp", unit="frame", leave=False):
                t_sec = i / output_fps

                # authoritative behavior index from timestamps
                original_behavior_frame_index = beh_idx_from_time(t_sec)
                original_behavior_frame_index = max(start_idx_orig, min(end_idx_orig, original_behavior_frame_index))

                # indices for pose/syllable (both rendered at output fps)
                pose_frame_index = i
                syllable_frame_index = i

                # Behavior: guard when physical file ends early
                beh_read_idx = original_behavior_frame_index
                if beh_total > 0:
                    beh_read_idx = min(beh_read_idx, beh_total - 1)

                frame_behav_raw = video_utils.read_frame_at_index(cap_behav, beh_read_idx)
                if frame_behav_raw is None:
                    if behavior_gap_policy == "hold" and beh_total > 0:
                        # try reading the last available frame
                        frame_behav_raw = video_utils.read_frame_at_index(cap_behav, beh_total - 1)
                    # else leave as None → will become black via resize_and_pad

                frame_pose_raw = video_utils.read_frame_at_index(cap_pose, pose_frame_index)
                frame_syll_raw = video_utils.read_frame_at_index(cap_syll, syllable_frame_index)

                frame_trace_raw = plotting.generate_trace_plot_frame(
                    filtered_photometry_df,
                    global_ylim,
                    original_behavior_frame_index,
                    session.left_region,
                    session.right_region,
                    start_idx_orig,
                    cam_ts_ns=session._cam_ts_ns,               # precise time axis in plot
                    log_prefix=f"{log_prefix}[Frame {i}] "
                )

                # Resize & pad to quadrants
                q_w, q_h = config.QUADRANT_WIDTH, config.QUADRANT_HEIGHT
                frame_tl = video_utils.resize_and_pad(frame_behav_raw, q_w, q_h)  # Top-Left: Behavior
                frame_tr = video_utils.resize_and_pad(frame_pose_raw,  q_w, q_h)  # Top-Right: Pose
                frame_bl = video_utils.resize_and_pad(frame_trace_raw, q_w, q_h)  # Bottom-Left: Trace
                frame_br = video_utils.resize_and_pad(frame_syll_raw,  q_w, q_h)  # Bottom-Right: Syllable

                composite_frame = video_utils.assemble_composite_frame(frame_tl, frame_tr, frame_bl, frame_br)
                if composite_frame.shape[:2] != (output_height, output_width):
                    logger.error(f"{log_prefix}Composite dims wrong for frame {i}. Skipping write.")
                    continue

                # --- Title (black) ---
                import re
                def normalize_hyphens(s: str) -> str:
                    return re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', s)

                safe_left  = normalize_hyphens(session.left_region)
                safe_right = normalize_hyphens(session.right_region)
                title_text = f"Mouse: {session.mouse_id} | Genotype: {session.genotype} | {safe_left}/{safe_right}"

                font=cv2.FONT_HERSHEY_SIMPLEX
                font_scale=1.0
                font_color=(0, 0, 0)   # BLACK
                thickness=2
                line_type=cv2.LINE_AA
                margin = 20

                (tw, th), base = cv2.getTextSize(title_text, font, font_scale, thickness)
                text_x = (output_width - tw) // 2
                text_y = margin + th
                cv2.putText(composite_frame, title_text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

                writer.write(composite_frame)

        end_time = time.time()
        logger.info(f"{log_prefix}✅ Finished {mode_text}{preview_text}: {output_path.name} (Duration: {end_time - start_time:.2f}s)")

    except IOError as e:
        logger.error(f"{log_prefix}IOError during video generation: {e}")
        if output_path.exists() and writer and writer.isOpened():
            writer.release()
            try: output_path.unlink()
            except OSError as ose: logger.error(f"{log_prefix}Error removing file: {ose}")

    except Exception as e:
        logger.error(f"{log_prefix}ERROR during video generation loop: {e}", exc_info=True)
        if output_path.exists() and writer and writer.isOpened():
            writer.release()
            try: output_path.unlink(missing_ok=True)
            except OSError as ose: logger.error(f"{log_prefix}Error removing file: {ose}")

    finally:
        logger.debug(f"{log_prefix}Cleaning up resources...")
        released = 0
        for cap in caps_to_release:
            if cap and cap.isOpened():
                cap.release()
                released += 1
        if released:
            logger.debug(f"{log_prefix}Released {released} video captures.")
        if writer and writer.isOpened():
            writer.release()
            logger.debug(f"{log_prefix}Released video writer.")


def _render_one(session, global_ylim):
    """Small wrapper so each unit of work is self-contained for the executor."""
    # Optional: quiet matplotlib debug in workers
    try:
        import matplotlib as mpl, logging
        mpl.set_loglevel("warning")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    except Exception:
        pass

    # Do the work
    create_composite_video(session, global_ylim)
    return session.base_video_key

def render_sessions_parallel(sessions, global_ylim, workers=None, mode=None):
    if not sessions:
        return

    # Resolve settings
    workers = workers or getattr(config, "MAX_WORKERS", 1)
    mode = (mode or getattr(config, "PARALLEL_MODE", "process")).lower()
    workers = max(1, int(workers))
    workers = min(workers, len(sessions))

    if workers == 1 or len(sessions) == 1:
        for s in sessions:
            _render_one(s, global_ylim)
        return

    Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

    # Use "spawn" start method for safety with Matplotlib/OpenCV
    ctx = multiprocessing.get_context("spawn") if mode == "process" else None
    init_kwargs = {"mp_context": ctx} if ctx is not None else {}

    logger.info(f"Rendering {len(sessions)} videos with {workers} {mode}(s)...")

    # Submit jobs
    with Executor(max_workers=workers, **init_kwargs) as ex:
        fut_map = {ex.submit(_render_one, s, global_ylim): s.base_video_key for s in sessions}
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            try:
                fut.result()
                logger.info(f"[{key}] ✅ finished")
            except Exception as e:
                logger.error(f"[{key}] ❌ failed: {e}", exc_info=True)


def main():
    """Main pipeline execution function."""
    overall_start_time = time.time()
    logger.info("="*60)
    logger.info(" Starting Neuroscience Video Generation Pipeline")
    logger.info(f" Preview Mode: {config.PREVIEW_MODE}")
    logger.info(f" Dopamine Only Mode: {config.DOPAMINE_ONLY_MODE}")
    logger.info("="*60)


    # --- 1. Generate Syllable Videos and Get Frame Ranges ---
    # This step now also handles caching internally
    logger.info("\n--- Step 1: Generating Syllable Data & Videos ---")
    frame_ranges_data = syllable_generator.generate_all_syllable_videos()

    if frame_ranges_data is None:
        logger.critical("Failed to generate or load frame ranges from syllable processing. Aborting.")
        return
    if not frame_ranges_data:
        logger.warning("Syllable processing returned no frame ranges. No sessions can be processed fully. Aborting.")
        return
    logger.info(f"Successfully obtained frame ranges for {len(frame_ranges_data)} potential sessions.")


    # --- 2. Find Valid Experiment Sessions ---
    logger.info("\n--- Step 2: Finding Valid Experiment Sessions ---")
    sessions = data_loader.find_sessions(frame_ranges_data)
    if not sessions:
        logger.critical("No valid sessions found matching criteria. Aborting.")
        return
    logger.info(f"Found {len(sessions)} valid sessions for processing.")


    # --- 3. First Pass: Load, Merge Timestamps, Filter Photometry, Collect Data for Y-Limits ---
    logger.info("\n--- Step 3: Preprocessing Photometry Data (Load, Merge, Filter) ---")
    all_filtered_zscore_values: List[float] = []
    sessions_to_process: List[Session] = [] # Keep track of sessions that pass preprocessing

    for session in tqdm(sessions, desc="Preprocessing Photometry", unit="session", ascii=True):
        log_prefix = f"[{session.base_video_key}] "
        logger.info(f"\n{log_prefix}Processing session...")

        # --- Load Raw Data and Merge Timestamps ---
        raw_photometry_df = photometry_processor.load_raw_photometry_and_merge_timestamps(
            session_photometry_csv=session.photometry_csv_path,
            session_timestamp_csv=session.center_timestamp_csv_path,
            arena=session.arena,
            left_region_name=session.left_region,
            right_region_name=session.right_region,
            log_prefix=log_prefix
        )

        if raw_photometry_df is None or raw_photometry_df.empty:
            logger.error(f"{log_prefix}Failed to load/merge photometry and timestamps. Skipping session.")
            session.is_valid = False # Mark as invalid if loading fails
            continue
        # Store raw merged data if needed? Not currently used downstream.
        # session.photometry_df = raw_photometry_df

        cam_ts_ns = photometry_processor.load_camera_timestamps_ns(session.center_timestamp_csv_path)
        if cam_ts_ns is None:
            logger.error(f"{log_prefix}Failed to load camera timestamps. Skipping session.")
            session.is_valid = False
            continue

        session._cam_ts_ns = cam_ts_ns
        # --- Filter the Relevant Segment ---
        if not session.frame_range_original:
            logger.error(f"{log_prefix}Missing frame range. Cannot filter photometry. Skipping session.")
            session.is_valid = False
            continue

        start_idx, end_idx = session.frame_range_original
        filtered_segment_df = photometry_processor.filter_photometry_segment(
            raw_photometry_df,
            [session.left_region, session.right_region],
            start_idx,
            end_idx,
            cam_ts_ns=cam_ts_ns,                 # <— ADD THIS
            log_prefix=log_prefix
)

        if filtered_segment_df is None or filtered_segment_df.empty:
            logger.error(f"{log_prefix}Filtering photometry segment failed. Skipping session.")
            session.is_valid = False
            continue

        # Store the filtered data in the session object
        session.filtered_photometry_segment_df = filtered_segment_df
        sessions_to_process.append(session) # Add to list of successfully preprocessed sessions
        logger.info(f"{log_prefix}Successfully loaded and filtered photometry segment.")


        # --- Collect Data for Global Y-Limits ---
        for region in [session.left_region, session.right_region]:
            col_name = f"{region}_filtered"
            if col_name in filtered_segment_df.columns:
                # Get valid (non-NaN) filtered Z-score values
                valid_data = filtered_segment_df[col_name].dropna().values
                if len(valid_data) > 0:
                    all_filtered_zscore_values.extend(valid_data)
                    logger.debug(f"{log_prefix}Collected {len(valid_data)} data points from '{col_name}' for global limits.")
                else:
                     logger.debug(f"{log_prefix}No valid data points in '{col_name}' after filtering.")
            else:
                logger.warning(f"{log_prefix}Filtered column '{col_name}' not found after filtering. Cannot use for global limits.")


        # --- Optional: Log details for the first successfully processed session ---
        if sessions_to_process and session == sessions_to_process[0]: # Check if this is the first one added
             logger.debug(f"\n{log_prefix}--- First Session Filtered Data Details ---")
             logger.debug(f"{log_prefix}Filtered Segment Info:")
             # Use string capture for info() output if needed, or just log describe()
             # buf = io.StringIO()
             # filtered_segment_df.info(buf=buf, memory_usage=False, verbose=True)
             # logger.debug(buf.getvalue())
             logger.debug(f"\n{filtered_segment_df.describe()}")
             for region in [session.left_region, session.right_region]:
                 col_name = f"{region}_filtered"
                 if col_name in filtered_segment_df.columns:
                      logger.debug(f"\n{log_prefix}Stats for '{col_name}':")
                      stats = filtered_segment_df[col_name].agg(['count', 'min', 'max', 'mean', 'median'])
                      logger.debug(f"{stats.to_string()}")
                      nan_count = filtered_segment_df[col_name].isna().sum()
                      logger.debug(f"  NaN Count: {nan_count}")
             logger.debug(f"{log_prefix}-----------------------------------------\n")


    # --- Check if any sessions are left after preprocessing ---
    if not sessions_to_process:
        logger.critical("\nNo sessions successfully preprocessed (loading/filtering failed). Aborting video creation.")
        return

    logger.info(f"\n--- Finished Photometry Preprocessing for {len(sessions_to_process)} sessions ---")


    # --- 4. Calculate Global Y-Limits ---
    logger.info("\n--- Step 4: Calculating Global Y-Limits for Plots ---")
    global_ylim: Tuple[float, float] = (-3.0, 3.0) # Default fallback

    if all_filtered_zscore_values:
        num_points = len(all_filtered_zscore_values)
        logger.info(f"Calculating global limits from {num_points} collected filtered Z-score data points.")
        try:
            # Calculate percentiles
            q_low = np.percentile(all_filtered_zscore_values, config.GLOBAL_YLIM_PERCENTILE_LOW)
            q_high = np.percentile(all_filtered_zscore_values, config.GLOBAL_YLIM_PERCENTILE_HIGH)

            # Add padding (e.g., 5% of the range)
            padding = abs(q_high - q_low) * 0.05
            temp_ylim = (q_low - padding, q_high + padding)

            # Validate calculated limits
            if np.isfinite(temp_ylim[0]) and np.isfinite(temp_ylim[1]) and temp_ylim[0] < temp_ylim[1]:
                global_ylim = temp_ylim
                logger.info(f"Global Y-limits set to: ({global_ylim[0]:.2f}, {global_ylim[1]:.2f}) "
                            f"based on {config.GLOBAL_YLIM_PERCENTILE_LOW}-{config.GLOBAL_YLIM_PERCENTILE_HIGH} percentiles.")
            else:
                logger.warning(f"Calculated global quantile Y limits are invalid ({temp_ylim}). Using default {global_ylim}.")
        except Exception as e:
            logger.error(f"Error calculating global Y limits from percentiles: {e}. Using default {global_ylim}.", exc_info=True)
    else:
        logger.warning("No filtered data points collected. Cannot calculate global Y limits. Using default {global_ylim}.")


    # --- 5. Second Pass: Generate Videos ---
    mode_info = "(Dopamine Only)" if config.DOPAMINE_ONLY_MODE else "(Composite)"
    logger.info(f"\n--- Step 5: Starting Video Creation {mode_info} for {len(sessions_to_process)} preprocessed sessions ---")

    # Iterate through the sessions that were successfully preprocessed
    render_sessions_parallel(
        sessions=sessions_to_process,
        global_ylim=global_ylim,
        workers=getattr(config, "MAX_WORKERS", 1),
        mode=getattr(config, "PARALLEL_MODE", "process"),
    )

    overall_end_time = time.time()
    logger.info("\n" + "="*60)
    logger.info(f" Pipeline processing complete. Total time: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info("="*60)


if __name__ == "__main__":
    # --- Dependency Checks ---
    missing_libs = []
    try: import scipy.signal
    except ImportError: missing_libs.append("scipy")
    # Check kpms/pdf2image based on the objects in syllable_generator
    if syllable_generator.kpms is None: missing_libs.append("keypoint_moseq")
    if syllable_generator.convert_from_path is None: missing_libs.append("pdf2image (and poppler utilities)")
    try: import matplotlib.pyplot
    except ImportError: missing_libs.append("matplotlib")
    try: import cv2
    except ImportError: missing_libs.append("opencv-python")
    # Add pandas, numpy checks if desired, though they are usually core dependencies

    if missing_libs:
        logger.critical("Missing critical libraries. Please install them:")
        for lib in missing_libs:
            logger.critical(f"  - {lib}")
        logger.critical("Aborting execution.")
    else:
        logger.info("All critical libraries seem to be available.")
        main() # Run the main pipeline