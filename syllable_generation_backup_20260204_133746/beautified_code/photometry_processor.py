# photometry_processor.py
import pandas as pd
import numpy as np
from scipy import signal
from typing import List, Optional, Tuple
import logging # Use logging instead of print

import config
from session import Session # Import Session for type hinting if needed elsewhere

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Helper function moved from plotting.py ---
# (These could also live in a separate timestamp_utils.py if they grow)
def find_nearest_iloc(sorted_array: np.ndarray, value: float) -> int:
    """Finds the index (iloc) of the element nearest to 'value' in a sorted numpy array."""
    if sorted_array is None or len(sorted_array) == 0:
        logger.warning("find_nearest_iloc received empty or None array.")
        return -1
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx == 0: return 0
    if idx == len(sorted_array): return len(sorted_array) - 1
    left_val = sorted_array[idx - 1]
    right_val = sorted_array[idx]
    if abs(value - left_val) < abs(value - right_val): return idx - 1
    else: return idx

def _estimate_target_behavior_timestamp(
    behavior_timestamps_np: np.ndarray,
    target_original_frame: int,
    frame_range_original_start: int,
    behavior_video_fps: float = config.BEHAVIOR_VIDEO_FPS, # Use config directly
    timestamp_scale_factor: float = 1e9 # Assume ns -> s
    ) -> Optional[float]:
    """Estimates the BehaviorTimestamp corresponding to a target frame index."""
    if len(behavior_timestamps_np) == 0:
        logger.warning("_estimate_target_behavior_timestamp received empty timestamps.")
        return None
    try:
        # Assume the first timestamp corresponds roughly to the start of the recording
        # For segment estimation, finding the timestamp near the start_iloc is better
        # Let's assume the array passed might be a segment, so using the first element is safer
        start_ts = behavior_timestamps_np[0]

        # Calculate time offset relative to the reference start frame
        time_offset_seconds = (target_original_frame - frame_range_original_start) / behavior_video_fps

        # Estimate target timestamp
        target_timestamp = start_ts + time_offset_seconds * timestamp_scale_factor
        return target_timestamp
    except (IndexError, ValueError) as e:
        logger.error(f"Error estimating target timestamp: {e}")
        return None

# --- Renamed and Modified from data_loader.load_photometry_data ---
def load_raw_photometry_and_merge_timestamps(
    session_photometry_csv: str,
    session_timestamp_csv: str,
    arena: str,
    left_region_name: str,
    right_region_name: str,
    log_prefix: str = ""
) -> Optional[pd.DataFrame]:
    """
    Loads raw photometry, calculates dF/F Z-score, loads timestamps,
    and merges using merge_asof (nearest). Uses the camera clock for
    BehaviorTimestamp (ns) to avoid drift.
    """
    logger.info(f"{log_prefix}Loading raw photometry and timestamps...")
    try:
        # --- Load Raw Photometry ---
        left_col, right_col = config.ARENA_TO_PHOTOMETRY_COLS.get(arena, (None, None))
        if not left_col or not right_col:
            logger.error(f"{log_prefix}Arena '{arena}' not found in ARENA_TO_PHOTOMETRY_COLS.")
            return None

        usecols = ["FrameCounter", "SystemTimestamp", "ComputerTimestamp", "LedState", left_col, right_col]
        df = pd.read_csv(session_photometry_csv, usecols=usecols)

        # --- Split Iso/Real and Align ---
        iso_df  = df[df.LedState == 1].copy()
        real_df = df[df.LedState == 2].copy()
        min_len = min(len(iso_df), len(real_df))
        if abs(len(iso_df) - len(real_df)) > 5:
            logger.warning(f"{log_prefix}Large length diff ({abs(len(iso_df) - len(real_df))}) "
                           f"between iso ({len(iso_df)}) and real ({len(real_df)}). Trimming.")
        iso_df  = iso_df.iloc[:min_len]
        real_df = real_df.iloc[:min_len]
        if min_len == 0:
            logger.error(f"{log_prefix}Zero length data after iso/real split.")
            return None

        out = pd.DataFrame(index=real_df.index)
        # Keep the device SystemTimestamp (seconds) for merging
        out["SystemTimestamp"] = real_df["SystemTimestamp"]

        # --- dF/F → Z-score per side (unchanged logic) ---
        logger.debug(f"{log_prefix}Calculating dF/F Z-score...")
        for side, col in [(left_region_name, left_col), (right_region_name, right_col)]:
            phot = real_df[col].values
            iso  = iso_df[col].values

            valid_mask = np.isfinite(phot) & np.isfinite(iso)
            if not np.any(valid_mask):
                logger.warning(f"{log_prefix}No valid data points for {side} dF/F calculation.")
                out[side] = 0.0
                continue

            phot_valid, iso_valid = phot[valid_mask], iso[valid_mask]
            try:
                # Linear fit correction between iso and phot
                coeffs = np.polyfit(iso_valid, phot_valid, 1)
                iso_pred = np.polyval(coeffs, iso_valid)
                dff_valid = (phot_valid - iso_pred)  # absolute deviation from predicted iso line
            except np.linalg.LinAlgError:
                logger.warning(f"{log_prefix}Linear fit failed for {side}. Using mean difference.")
                mean_diff = float(np.nanmean(phot_valid) - np.nanmean(iso_valid))
                dff_valid = (phot_valid - iso_valid) + mean_diff

            dff_full = np.full_like(phot, np.nan)
            dff_full[valid_mask] = dff_valid

            if len(dff_valid) > 1:
                mu, sd = float(np.nanmean(dff_valid)), float(np.nanstd(dff_valid))
                if sd > 1e-9:
                    z = (dff_valid - mu) / sd
                    out[side] = np.full_like(phot, np.nan)
                    out.loc[valid_mask, side] = z
                else:
                    logger.warning(f"{log_prefix}Near-zero std for {side}; Z=0.")
                    out[side] = 0.0
            elif len(dff_valid) == 1:
                logger.warning(f"{log_prefix}Only one point for {side}; Z=0.")
                out[side] = 0.0
            else:
                out[side] = np.nan

        logger.debug(f"{log_prefix}Finished dF/F Z-score calculation.")

        # --- Load All Timestamps (camera + FP device) ---
        logger.debug(f"{log_prefix}Loading center timestamps...")
        df_ts = pd.read_csv(session_timestamp_csv)

        # We merge on FP device seconds (SystemTimestamp ↔ FP3002_System_Timestamp).
        if "FP3002_System_Timestamp" not in df_ts.columns:
            logger.error(f"{log_prefix}Missing 'FP3002_System_Timestamp' in {session_timestamp_csv}")
            return None

        # Prefer camera Basler clock (ns) for the canonical behavior timebase
        have_cam_ns = "Cam_Basler_Timestamp" in df_ts.columns

        # Prepare numeric/clean columns for merge
        out["SystemTimestamp"] = pd.to_numeric(out["SystemTimestamp"], errors="coerce")
        df_ts["FP3002_System_Timestamp"] = pd.to_numeric(df_ts["FP3002_System_Timestamp"], errors="coerce")

        if have_cam_ns:
            df_ts["Cam_Basler_Timestamp"] = pd.to_numeric(df_ts["Cam_Basler_Timestamp"], errors="coerce")

        # Drop NaNs and sort on merge keys
        out = out.dropna(subset=["SystemTimestamp"]).sort_values("SystemTimestamp")
        df_ts = df_ts.dropna(subset=["FP3002_System_Timestamp"]).sort_values("FP3002_System_Timestamp")

        if out.empty or df_ts.empty:
            logger.error(f"{log_prefix}Empty data after cleaning merge keys.")
            return None

        # Create a frame index tied to camera rows (0-based, row order == frame order)
        df_ts = df_ts.reset_index(drop=True)
        df_ts["BehaviorFrameIndex"] = np.arange(len(df_ts), dtype=np.int64)

        keep_ts_cols = ["FP3002_System_Timestamp", "BehaviorFrameIndex"]
        if have_cam_ns:
            keep_ts_cols.append("Cam_Basler_Timestamp")

        # --- Nearest-time merge: SystemTimestamp (s) ↔ FP3002_System_Timestamp (s) ---
        logger.debug(f"{log_prefix}Performing merge_asof (direction='nearest')...")
        merged = pd.merge_asof(
            out,
            df_ts[keep_ts_cols],
            left_on="SystemTimestamp",
            right_on="FP3002_System_Timestamp",
            direction="nearest"
        )
        logger.debug(f"{log_prefix}Merge complete.")

        # --- Define canonical BehaviorTimestamp (ns) ---
        if have_cam_ns and "Cam_Basler_Timestamp" in merged.columns and merged["Cam_Basler_Timestamp"].notna().any():
            merged["BehaviorTimestamp"] = merged["Cam_Basler_Timestamp"]
            source = "Cam_Basler_Timestamp (ns)"
        else:
            # Fallback: use FP device seconds and convert to ns
            merged["BehaviorTimestamp"] = merged["FP3002_System_Timestamp"] * 1e9
            source = "FP3002_System_Timestamp * 1e9 (fallback)"
        logger.info(f"{log_prefix}BehaviorTimestamp source: {source}")

        # --- Finalize DataFrame ---
        keep_cols = [left_region_name, right_region_name, "BehaviorTimestamp", "SystemTimestamp", "BehaviorFrameIndex"]
        keep_cols = [c for c in keep_cols if c in merged.columns]
        final_df = merged[keep_cols].sort_values("BehaviorTimestamp").reset_index(drop=True)

        logger.info(f"{log_prefix}Successfully loaded and merged data. Shape: {final_df.shape}")
        return final_df

    except FileNotFoundError as e:
        logger.error(f"{log_prefix}File not found: {e}")
        return None
    except KeyError as e:
        logger.error(f"{log_prefix}Missing column during processing: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error during photometry loading/merging: {e}", exc_info=True)
        return None


def load_camera_timestamps_ns(all_timestamps_csv: str) -> Optional[np.ndarray]:
    df = pd.read_csv(all_timestamps_csv, usecols=["Cam_Basler_Timestamp"])
    ts = pd.to_numeric(df["Cam_Basler_Timestamp"], errors="coerce").to_numpy()
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        logger.error(f"No valid Cam_Basler_Timestamp values in {all_timestamps_csv}")
        return None
    return ts  # nanoseconds, one per behavior frame, index == original frame index

# --- Moved from plotting.py ---
def filter_photometry_segment(
    photometry_df: pd.DataFrame,
    regions: List[str],
    frame_range_original_start: int,
    frame_range_original_end: int,
    cam_ts_ns: Optional[np.ndarray] = None,   # <— ADD THIS
    log_prefix: str = ""
) -> Optional[pd.DataFrame]:
    """
    Slices the relevant segment from the raw photometry_df (containing Z-scores
    and BehaviorTimestamp) and applies a low-pass Butterworth filter.

    Args:
        photometry_df: DataFrame from load_raw_photometry_and_merge_timestamps.
        regions: List of region names (column names for Z-score data).
        frame_range_original_start: Start frame index (original 60fps video).
        frame_range_original_end: End frame index (original 60fps video).
        log_prefix: String prefix for log messages.

    Returns:
        DataFrame containing the filtered segment with original Z-score columns and
        new '_filtered' columns, or None if an error occurs. The DataFrame includes
        the 'BehaviorTimestamp' column.
    """
    logger.info(f"{log_prefix}Filtering photometry segment...")
    if photometry_df is None or photometry_df.empty or "BehaviorTimestamp" not in photometry_df.columns:
        logger.error(f"{log_prefix}Invalid or empty DataFrame passed to filter_photometry_segment.")
        return None

    behavior_timestamps = photometry_df["BehaviorTimestamp"].values
    if len(behavior_timestamps) == 0:
        logger.error(f"{log_prefix}BehaviorTimestamp column is empty.")
        return None

    # --- Estimate Segment Boundaries using Timestamps ---
    # --- Use real per-frame camera timestamps if provided ---
    if cam_ts_ns is not None and len(cam_ts_ns) > frame_range_original_end:
        segment_start_ts = cam_ts_ns[frame_range_original_start]
        segment_end_ts   = cam_ts_ns[frame_range_original_end]
    else:
        # fallback = old fps-based estimate (unchanged)
        segment_start_ts = _estimate_target_behavior_timestamp(
            behavior_timestamps, frame_range_original_start, frame_range_original_start
        )
        segment_end_ts = _estimate_target_behavior_timestamp(
            behavior_timestamps, frame_range_original_end, frame_range_original_start
        )

    if segment_start_ts is None or segment_end_ts is None:
        logger.error(f"{log_prefix}Failed to estimate segment start/end timestamps.")
        return None

    # Find corresponding indices (iloc) in the full dataframe
    start_iloc_segment = find_nearest_iloc(behavior_timestamps, segment_start_ts)
    end_iloc_segment = find_nearest_iloc(behavior_timestamps, segment_end_ts)

    if start_iloc_segment == -1 or end_iloc_segment == -1 or start_iloc_segment > end_iloc_segment:
        logger.error(f"{log_prefix}Invalid segment indices found: start={start_iloc_segment}, end={end_iloc_segment}")
        return None

    # --- Slice Segment ---
    # Ensure slicing includes the end point
    segment_df = photometry_df.iloc[start_iloc_segment : end_iloc_segment + 1].copy()
    if segment_df.empty:
        logger.error(f"{log_prefix}Segment DataFrame is empty after slicing "
                     f"iloc[{start_iloc_segment}:{end_iloc_segment+1}].")
        return None
    logger.info(f"{log_prefix}Sliced segment iloc[{start_iloc_segment}:{end_iloc_segment+1}], shape: {segment_df.shape}")

    # --- Estimate Sampling Rate (fs) within the segment ---
    segment_timestamps = segment_df["BehaviorTimestamp"].values
    if len(segment_timestamps) < 2:
        logger.warning(f"{log_prefix}Segment too short ({len(segment_timestamps)} points) to estimate sampling rate. Skipping filtering.")
        # Add '_filtered' columns with original data if skipping filter
        for region in regions:
            if region in segment_df.columns:
                segment_df[f"{region}_filtered"] = segment_df[region]
            else:
                segment_df[f"{region}_filtered"] = np.nan # Or handle as needed
        return segment_df # Return unfiltered segment

    # Use median time difference for robustness against outliers
    TIMESTAMP_SCALE_FACTOR = 1e9 # ns -> s
    #time_diffs = np.diff(segment_timestamps) / TIMESTAMP_SCALE_FACTOR
    median_dt_sec = np.median(np.diff(segment_timestamps)) / 1e9

    if median_dt_sec <= 0 or not np.isfinite(median_dt_sec):
        logger.warning(f"{log_prefix}Invalid median time difference ({median_dt_sec:.4f}s). Cannot estimate fs. Skipping filtering.")
        # Add '_filtered' columns with original data
        for region in regions:
             if region in segment_df.columns: segment_df[f"{region}_filtered"] = segment_df[region]
             else: segment_df[f"{region}_filtered"] = np.nan
        return segment_df

    fs = 1.0 / median_dt_sec
    nyq = 0.5 * fs
    norm = (config.FILTER_CUTOFF_HZ or 0) / nyq
    logger.info(f"{log_prefix} fs={fs:.2f} Hz, nyq={nyq:.2f} Hz, cutoff={config.FILTER_CUTOFF_HZ} Hz (norm={norm:.3f}), order={config.FILTER_ORDER}")
    logger.info(f"{log_prefix}Estimated sampling rate (fs): {fs:.2f} Hz (Median dt: {median_dt_sec*1000:.2f} ms)")

    # Check if filtering is feasible/meaningful
    if config.FILTER_CUTOFF_HZ >= nyq:
        logger.warning(f"{log_prefix}Filter cutoff frequency ({config.FILTER_CUTOFF_HZ} Hz) >= Nyquist frequency ({nyq:.2f} Hz). Skipping filtering.")
        # Add '_filtered' columns with original data
        for region in regions:
             if region in segment_df.columns: segment_df[f"{region}_filtered"] = segment_df[region]
             else: segment_df[f"{region}_filtered"] = np.nan
        return segment_df

    # --- Design Low-pass Butterworth Filter ---
    try:
        # Using second-order sections (sos) is numerically more stable
        sos = signal.butter(config.FILTER_ORDER, config.FILTER_CUTOFF_HZ,
                            btype='low', analog=False, output='sos', fs=fs)
        logger.info(f"{log_prefix}Applying {config.FILTER_ORDER}-order low-pass Butterworth filter (Cutoff: {config.FILTER_CUTOFF_HZ} Hz)...")
    except Exception as e:
        logger.error(f"{log_prefix}Failed to design Butterworth filter: {e}", exc_info=True)
        # Decide how to handle - return None? Or return unfiltered? Let's return unfiltered with a warning.
        for region in regions:
            if region in segment_df.columns: segment_df[f"{region}_filtered"] = segment_df[region]
            else: segment_df[f"{region}_filtered"] = np.nan
        return segment_df

    # --- Apply Filter (using sosfiltfilt for zero phase distortion) ---
    for region in regions:
        filtered_col_name = f"{region}_filtered"
        if region not in segment_df.columns:
            logger.warning(f"{log_prefix}Region column '{region}' not found in segment. Skipping filtering for it.")
            segment_df[filtered_col_name] = np.nan # Add column with NaNs
            continue

        original_zscore = segment_df[region].copy()

        # Interpolate NaNs before filtering for continuity. Limit direction avoids filling edges only.
        interpolated_zscore = original_zscore.interpolate(method='linear', limit_direction='both')

        # Check if interpolation left NaNs (e.g., all NaNs input)
        if interpolated_zscore.isna().any():
            # Fill remaining NaNs (e.g., if the whole series was NaN)
            fill_value = 0.0
            logger.warning(f"{log_prefix}Interpolated Z-score for {region} still contains NaNs. Filling with {fill_value}.")
            interpolated_zscore = interpolated_zscore.fillna(fill_value)

        # Check signal length relative to filter order for sosfiltfilt
        # Default padlen is 3 * (max(sos.shape[-1], sos.shape[-2]) - 1) which is complex
        # A simpler rule of thumb: length should be > 3 * filter_order
        required_len = 3 * config.FILTER_ORDER
        if len(interpolated_zscore) <= required_len:
             logger.warning(f"{log_prefix}Signal length ({len(interpolated_zscore)}) too short for filter padding "
                            f"(requires > ~{required_len}) for region {region}. Skipping filter application.")
             # Store the uninterpolated original Z-score in the filtered column
             segment_df[filtered_col_name] = original_zscore
             continue

        try:
            # Apply the filter (forward-backward)
            filtered_signal = signal.sosfiltfilt(sos, interpolated_zscore.values)

            # Store the filtered signal, preserving NaNs from the *original* data
            segment_df[filtered_col_name] = np.where(original_zscore.isna(), np.nan, filtered_signal)
            logger.debug(f"{log_prefix}Filtering successful for {region}.")

        except Exception as e:
            logger.error(f"{log_prefix}Failed to apply sosfiltfilt for region {region}: {e}", exc_info=True)
            # Store the uninterpolated original Z-score as fallback
            segment_df[filtered_col_name] = original_zscore

    logger.info(f"{log_prefix}Finished filtering photometry segment.")
    return segment_df