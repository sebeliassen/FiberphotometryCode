# syllable_generator.py
import pickle
import os
import re
import traceback
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any # Added Any
from tqdm import tqdm
import logging # Use logging

try:
    import keypoint_moseq as kpms
except ImportError:
    logging.error("keypoint_moseq library not found. Cannot generate syllable data.")
    kpms = None

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
except ImportError:
    logging.warning("pdf2image library not found. Cannot generate syllable videos from PDFs.")
    convert_from_path = None
    pdfinfo_from_path = None # Need this too

# Import config from the main project structure
import config as main_config

# Configure logger for this module
logger = logging.getLogger(__name__)


def load_kpms_data() -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Loads KpMS results, frame indices, and used_frames data."""
    if not kpms:
        logger.error("kpms library not loaded. Cannot load data.")
        return None, None, None

    logger.info("--- Loading Keypoint-MoSeq Data ---")
    results = None
    video_frame_indexes = None
    p3d_plot_used_frames = None

    # Load Model, Metadata, and Extract Results
    try:
        kpms_model_path = main_config.KPMS_PROJECT_DIR / main_config.KPMS_MODEL_NAME
        logger.info(f"Loading kpms checkpoint: {kpms_model_path}")
        # Assuming load_checkpoint loads necessary components
        model, _, metadata, _ = kpms.load_checkpoint(main_config.KPMS_PROJECT_DIR, main_config.KPMS_MODEL_NAME)

        if model is None or metadata is None:
            logger.error(f"load_checkpoint failed to return model or metadata from {kpms_model_path}")
            return None, None, None
        logger.debug(f"Kpms model and metadata loaded successfully.")

        logger.debug("Extracting results using loaded model and metadata...")
        # This might be memory intensive if results are large
        results = kpms.extract_results(model, metadata, main_config.KPMS_PROJECT_DIR, main_config.KPMS_MODEL_NAME)
        if results is None:
             logger.error("kpms.extract_results returned None.")
             return None, None, None
        logger.info(f"Extracted kpms results for {len(results)} items.")

    except FileNotFoundError as e:
        logger.error(f"Kpms checkpoint file or directory not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load Keypoint-MoSeq model/results: {e}", exc_info=True)
        return None, None, None

    # Load Video Frame Indexes
    try:
        logger.info(f"Loading video frame indexes from: {main_config.KPMS_VIDEO_FRAME_INDEXES_PKL}")
        with open(main_config.KPMS_VIDEO_FRAME_INDEXES_PKL, "rb") as f:
            video_frame_indexes = pickle.load(f)
        logger.info(f"Loaded frame indexes for {len(video_frame_indexes)} items.")
    except FileNotFoundError:
        logger.error(f"video_frame_indexes.pkl not found at {main_config.KPMS_VIDEO_FRAME_INDEXES_PKL}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load video frame indexes: {e}", exc_info=True)
        return None, None, None

    # Load Used Frames from 3D Pose output directory
    p3d_plot_used_frames = {}
    try:
        logger.info(f"Loading used_frames pkl files from: {main_config.POSE_3D_VIDEO_DIR}")
        count = 0
        for fname in os.listdir(main_config.POSE_3D_VIDEO_DIR):
            # Make matching more specific if possible (e.g., *_used_frames.pkl)
            if fname.lower().endswith("_used_frames.pkl"):
                try:
                    fpath = main_config.POSE_3D_VIDEO_DIR / fname
                    with open(fpath, "rb") as f:
                        # Store with the filename as key
                        p3d_plot_used_frames[fname] = pickle.load(f)
                    count += 1
                    logger.debug(f"  Loaded used_frames: {fname}")
                except Exception as e:
                    logger.warning(f"  Failed to load or read {fname}: {e}")
        logger.info(f"Loaded {count} used_frames files.")
        if count == 0:
             logger.error("No '_used_frames.pkl' files found in "
                          f"{main_config.POSE_3D_VIDEO_DIR}. Cannot filter syllable data.")
             return None, None, None

    except FileNotFoundError:
        logger.error(f"Directory for used_frames not found: {main_config.POSE_3D_VIDEO_DIR}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load used_frames data: {e}", exc_info=True)
        return None, None, None

    # Final check
    if results is None or video_frame_indexes is None or not p3d_plot_used_frames:
         logger.error("One or more essential KpMS data components failed to load.")
         return None, None, None

    return results, video_frame_indexes, p3d_plot_used_frames


def build_syllable_series(results: Dict, video_frame_indexes: Dict) -> Dict[str, pd.Series]:
    """Groups KpMS results by base filename and creates concatenated syllable Series."""
    logger.info("Building concatenated syllable series...")
    grouped: Dict[str, List[Tuple[int, str, Dict]]] = {}
    # Regex to capture base name and optional _partX suffix
    pattern = re.compile(r'(.*?)(_part(\d+))?$')

    for fname, f_dict in results.items():
        if 'syllable' not in f_dict:
            logger.debug(f"Skipping '{fname}': missing 'syllable' key in results.")
            continue
        if fname not in video_frame_indexes:
            logger.warning(f"Skipping '{fname}': missing corresponding entry in video_frame_indexes.")
            continue

        match = pattern.fullmatch(fname)
        if not match:
            logger.warning(f"Could not parse base filename from '{fname}'. Using full name as base.")
            base_fname = fname
            part_num = 1 # Assume part 1 if no suffix matched
        else:
            base_fname = match.group(1)
            part_num = int(match.group(3)) if match.group(3) else 1 # Default to part 1 if no number

        # Group by the derived base_fname
        grouped.setdefault(base_fname, []).append((part_num, fname, f_dict))

    # Concatenate parts for each base filename
    final_dict: Dict[str, pd.Series] = {}
    for base_fname, parts in grouped.items():
        # Sort parts by part number to ensure correct order
        parts_sorted = sorted(parts, key=lambda x: x[0])
        logger.debug(f"Concatenating {len(parts_sorted)} parts for base '{base_fname}'")

        syllable_list = []
        index_list = []
        try:
            for part_num, fname, f_dict in parts_sorted:
                syllables = f_dict['syllable']
                indices = video_frame_indexes[fname]

                if len(syllables) != len(indices):
                    logger.warning(f"  Length mismatch for '{fname}' (part {part_num}): "
                                   f"Syllables ({len(syllables)}) != Indices ({len(indices)}). Skipping part.")
                    continue # Skip this part

                syllable_list.append(syllables)
                index_list.append(indices)

            if not syllable_list: # Check if all parts were skipped
                 logger.warning(f"No valid parts to concatenate for base '{base_fname}'. Skipping.")
                 continue

            concatenated_syllable = np.concatenate(syllable_list)
            concatenated_index = np.concatenate(index_list)

            # Create the pandas Series
            temp_series = pd.Series(data=concatenated_syllable, index=concatenated_index)

            # --- Ensure the index is sorted and unique ---
            if not temp_series.index.is_monotonic_increasing:
                logger.debug(f"  Sorting index for '{base_fname}'.")
                temp_series = temp_series.sort_index()

            # Remove duplicate indices, keeping the first occurrence
            duplicates = temp_series.index.duplicated(keep='first')
            if duplicates.any():
                logger.warning(f"  Found {duplicates.sum()} duplicate indices for '{base_fname}'. Keeping first.")
                temp_series = temp_series[~duplicates]

            final_dict[base_fname] = temp_series

        except ValueError as e:
            logger.error(f"  Error concatenating data for {base_fname}: {e}. Skipping.", exc_info=True)
            continue
        except KeyError as e:
            logger.error(f"  KeyError during concatenation for {base_fname}: {e}. Skipping.", exc_info=True)
            continue

    logger.info(f"Built syllable series for {len(final_dict)} base items.")
    return final_dict


def filter_syllable_data(
    syllable_series_dict: Dict[str, pd.Series],
    p3d_plot_used_frames: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, pd.Series], Dict[str, Tuple[int, int]]]:
    """
    Filters syllable Series based on used_frames flags and extracts frame ranges.
    Uses the CANONICAL key format (Arena1234_...) for the output dictionaries.
    """
    logger.info("Filtering syllable data and extracting frame ranges...")
    final_dict_filtered: Dict[str, pd.Series] = {}
    frame_ranges_output: Dict[str, Tuple[int, int]] = {} # Use CANONICAL keys

    if not syllable_series_dict or not p3d_plot_used_frames:
         logger.error("Input dictionaries for filtering are empty. Cannot filter.")
         return {}, {}

    # Regex to parse timestamp and arena suffix from the kpms result base key
    # Example key: Arena1234_CenterTopCamVid_2024-12-13T14_50_39_bottom_leftDLC_resnet50...
    # We need the timestamp (group 1) and the suffix (group 3)
    key_pattern = re.compile(
        r'.*?(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})' # Timestamp (YYYY-MM-DDTHH_MM_SS)
        r'.*?(_(top_left|top_right|bottom_left|bottom_right))' # Arena Suffix (_suffix)
        r'.*' # Match the rest of the string
    )

    for kpms_base_key, series in syllable_series_dict.items():
        match = key_pattern.fullmatch(kpms_base_key)
        if not match:
             logger.warning(f"Cannot parse timestamp/suffix from kpms key '{kpms_base_key}'. Skipping.")
             continue

        ts = match.group(1) # e.g., 2024-12-13T14_50_39
        orient_with_underscore = match.group(2) # e.g., _top_left
        orient_only = match.group(3) # e.g., top_left

        # Find matching used_frames file. Keys are filenames like '..._used_frames.pkl'
        # We need to match based on timestamp and the orientation SUFFIX (e.g., _top_left)
        used_frames_key_pattern = f"*{ts}*{orient_with_underscore}*_used_frames.pkl"
        # Use Path.match for glob-style matching on keys (filenames)
        matches = [k for k in p3d_plot_used_frames if Path(k).match(used_frames_key_pattern)]

        if not matches:
            logger.warning(f"No matching used_frames file found for pattern '{used_frames_key_pattern}' "
                           f"(derived from kpms key '{kpms_base_key}'). Skipping filtering for this item.")
            continue
        if len(matches) > 1:
             # Sort matches for consistent selection if multiple occur
             matches.sort()
             logger.warning(f"Multiple ({len(matches)}) used_frames files matched pattern '{used_frames_key_pattern}'. "
                            f"Using first sorted match: '{matches[0]}'.")

        used_frames_filename = matches[0]
        used_frames_flags = p3d_plot_used_frames[used_frames_filename]

        # Find the indices where used_frames_flags is 1 (or True)
        ones_indices = np.where(used_frames_flags)[0] # np.where returns a tuple

        if ones_indices.size == 0:
            logger.warning(f"No 'used' frames (value 1) found in '{used_frames_filename}' "
                           f"for kpms key '{kpms_base_key}'. Skipping filtering.")
            continue

        # Determine the range based on the first and last 'used' frame index
        first_used_idx = ones_indices[0]
        last_used_idx = ones_indices[-1]

        # Filter the syllable series to include only frames within this range
        # Using .loc for label-based indexing (frame indices are the labels)
        # Ensure indices are within the bounds of the series index
        min_series_idx = series.index.min()
        max_series_idx = series.index.max()
        actual_start_idx = max(first_used_idx, min_series_idx)
        actual_end_idx = min(last_used_idx, max_series_idx)

        if actual_start_idx > actual_end_idx:
             logger.warning(f"Filtering range [{first_used_idx}, {last_used_idx}] is outside "
                            f"series index range [{min_series_idx}, {max_series_idx}] "
                            f"for '{kpms_base_key}'. Skipping.")
             continue

        filtered_series = series.loc[actual_start_idx:actual_end_idx]

        if filtered_series.empty:
            logger.warning(f"Series for '{kpms_base_key}' became empty after filtering by "
                           f"range [{actual_start_idx}, {actual_end_idx}]. Skipping.")
            continue

        # --- Use the CANONICAL key format for output ---
        # Format: Arena1234_CenterTopCamVid_{Timestamp}_{ArenaSuffix(no underscore)}
        canonical_key = f"Arena1234_CenterTopCamVid_{ts}_{orient_only}"

        final_dict_filtered[canonical_key] = filtered_series
        # Store the original range derived from used_frames
        frame_ranges_output[canonical_key] = (first_used_idx, last_used_idx)

        logger.debug(f"  Filtered '{kpms_base_key}' -> '{canonical_key}': "
                     f"Range [{first_used_idx}, {last_used_idx}], "
                     f"Series length {len(filtered_series)}")

    logger.info(f"Filtered data and generated frame ranges for {len(final_dict_filtered)} items.")
    return final_dict_filtered, frame_ranges_output


# --- Helper function for DRY PDF processing ---
def _load_and_process_pdf_page(
    pdf_path: Path,
    target_width: int,
    target_height: int
    ) -> Optional[np.ndarray]:
    """Loads the first page of a PDF, resizes, and pads to target dimensions."""
    if not convert_from_path:
        logger.warning("_load_and_process_pdf_page called but pdf2image not available.")
        return None
    if not pdf_path.exists():
        # logger.debug(f"PDF not found: {pdf_path}") # This can be very verbose
        return None

    try:
        # Optimization: Check page count first? Maybe not worth the overhead.
        # info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
        # if info.get('Pages', 0) < 1: return None # Skip if no pages

        # Load only the first page with appropriate DPI for target size
        # DPI choice affects loading time and initial resolution. 150 is a guess.
        pil_images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=150)
        if not pil_images: return None
        pil_img = pil_images[0]

        # Convert PIL Image to OpenCV format (RGB -> BGR)
        img_rgb = np.array(pil_img)
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
             logger.warning(f"Unexpected image format from PDF {pdf_path.name}. Shape: {img_rgb.shape}")
             return None # Return None or a blank canvas? Let's return None.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Resize and pad (similar to video_utils.resize_and_pad but maybe simpler here)
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0: return None
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0: return None

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a white canvas and center the resized image
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8) # White background
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    except (PDFPageCountError, PDFSyntaxError) as pdf_err:
        # Handle specific pdf2image errors if needed
        logger.warning(f"PDF processing error for {pdf_path.name}: {pdf_err}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load/process PDF {pdf_path.name}: {e}", exc_info=False) # exc_info=False to reduce noise
        return None


def generate_single_syllable_video(
    video_name: str,
    series: pd.Series,
    frame_range: Tuple[int, int],
    output_dir: Path,
    pdf_dir: Path
    ):
    """Generates a video visualizing syllable changes based on PDF plots."""
    if not convert_from_path:
        logger.error("pdf2image not available. Cannot generate syllable video.")
        return

    start_idx_orig, end_idx_orig = frame_range
    # Calculate output frames based on original frame rate (60fps) and target FPS (30fps) -> downsample by 2
    # N_output = (N_original / FPS_original) * FPS_target
    # N_output = ( (end - start + 1) / 60 ) * 30 = (end - start + 1) / 2
    num_output_frames = (end_idx_orig - start_idx_orig + 1) // 2 # Integer division
    if num_output_frames <= 0:
        logger.warning(f"Calculated 0 output frames for {video_name} (Range: {frame_range}). Skipping video.")
        return

    out_path = output_dir / f"{video_name}.mp4"
    logger.info(f"Generating syllable video: {out_path.name} ({num_output_frames} frames)")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        main_config.TARGET_FPS,
        (main_config.SYLLABLE_VIDEO_WIDTH, main_config.SYLLABLE_VIDEO_HEIGHT)
    )
    if not writer.isOpened():
        logger.error(f"Failed to open video writer for {out_path}")
        return

    # --- State Variables ---
    last_syllable_label: Optional[int] = None
    # Initialize with a blank white frame
    last_frame_img: np.ndarray = np.full((main_config.SYLLABLE_VIDEO_HEIGHT, main_config.SYLLABLE_VIDEO_WIDTH, 3), 255, dtype=np.uint8)
    # Keep track of PDFs known to be missing or failed to load to avoid retrying
    missing_or_failed_pdfs: set[int] = set()
    generated_frame_count: int = 0

    # --- Pre-check Series ---
    if series.empty:
        logger.warning(f"Syllable series for {video_name} is empty. Video will be blank.")
    elif not series.index.is_monotonic_increasing:
        logger.error(f"Series index for {video_name} is not sorted! Skipping video generation.")
        writer.release()
        return
    min_series_idx = series.index.min() if not series.empty else -1

    # --- Attempt to initialize with the first valid frame's image ---
    initial_syllable_found = False
    if not series.empty:
        # Check a few initial frames to find the first displayable syllable
        initial_indices_to_check = [start_idx_orig + 2 * k for k in range(min(50, num_output_frames))] # Check up to 50 frames
        for initial_idx in initial_indices_to_check:
            if initial_idx >= min_series_idx:
                try:
                    initial_label_lookup = series.asof(initial_idx) # Find label at or before this index
                    if not pd.isna(initial_label_lookup):
                        initial_syllable = int(initial_label_lookup)
                        if initial_syllable not in missing_or_failed_pdfs:
                            pdf_path = pdf_dir / f"Syllable{initial_syllable}.pdf"
                            initial_img = _load_and_process_pdf_page(
                                pdf_path, main_config.SYLLABLE_VIDEO_WIDTH, main_config.SYLLABLE_VIDEO_HEIGHT
                            )
                            if initial_img is not None:
                                last_frame_img = initial_img
                                last_syllable_label = initial_syllable
                                initial_syllable_found = True
                                logger.debug(f"  Initialized syllable video '{video_name}' with Syllable {initial_syllable}.")
                                break # Stop searching once found
                            else:
                                # PDF missing or failed to load
                                missing_or_failed_pdfs.add(initial_syllable)
                                # logger.debug(f"  Initial PDF load failed/missing for Syllable {initial_syllable}")

                except Exception as e:
                    logger.warning(f"  Error during initial syllable lookup for {video_name} at index {initial_idx}: {e}")
                    break # Stop trying if lookup fails badly

        if not initial_syllable_found:
            logger.warning(f"  Could not initialize '{video_name}' with a valid syllable image "
                           f"(checked first {len(initial_indices_to_check)} frames). Video might start blank.")
            missing_attempted = {s for s in missing_or_failed_pdfs if s in series.unique()} # Log only relevant missing ones
            if missing_attempted:
                 logger.warning(f"    Initial syllables attempted but PDF missing/failed: {sorted(list(missing_attempted))}")


    # --- Main Frame Generation Loop ---
    try:
        # Loop through the desired *output* frame indices
        for i in tqdm(range(num_output_frames), desc=f"  SyllVid: {video_name[:20]:<20}", unit="fr", leave=False):
            # Calculate the corresponding original frame index (60fps) for this output frame (30fps)
            current_original_60fps_idx = start_idx_orig + 2 * i

            current_syllable_label: Optional[int] = None

            # Lookup the syllable label for the current original frame index
            if series.empty or current_original_60fps_idx < min_series_idx:
                # If before the start of the series, reuse the last known label
                current_syllable_label = last_syllable_label
            else:
                try:
                    # Use asof to get the label active at this frame index or the last one before it
                    label_lookup = series.asof(current_original_60fps_idx)
                    if not pd.isna(label_lookup):
                        current_syllable_label = int(label_lookup)
                    else:
                        # If asof returns NaN (e.g., gap in series before this index), reuse last label
                        current_syllable_label = last_syllable_label
                except Exception as e:
                     logger.warning(f"\n    Error using asof() for {video_name} frame {current_original_60fps_idx}: {e}. Reusing last label.")
                     current_syllable_label = last_syllable_label # Reuse last label on error

            # If no label could be determined (even initially), write the last image (blank)
            if current_syllable_label is None:
                 writer.write(last_frame_img)
                 generated_frame_count += 1
                 continue

            # Check if the syllable label has changed OR if it's the first frame being processed
            if current_syllable_label != last_syllable_label:
                # Only attempt to load if we haven't marked it as missing/failed
                if current_syllable_label not in missing_or_failed_pdfs:
                    pdf_path = pdf_dir / f"Syllable{current_syllable_label}.pdf"
                    new_frame_img = _load_and_process_pdf_page(
                        pdf_path, main_config.SYLLABLE_VIDEO_WIDTH, main_config.SYLLABLE_VIDEO_HEIGHT
                    )

                    if new_frame_img is not None:
                        # Successfully loaded the new image
                        last_frame_img = new_frame_img
                        last_syllable_label = current_syllable_label
                        logger.debug(f"    {video_name}: Switched to Syllable {current_syllable_label} at output frame {i}")
                    else:
                        # PDF was missing or failed to load, mark it and continue using the old image
                        missing_or_failed_pdfs.add(current_syllable_label)
                        logger.warning(f"\n    {video_name}: PDF missing or load failed for new Syllable {current_syllable_label}. "
                                       f"Continuing with Syllable {last_syllable_label}.")
                        # last_syllable_label remains the same
                        # last_frame_img remains the same

            # Write the current frame image (either the newly loaded one or the previous one)
            writer.write(last_frame_img)
            generated_frame_count += 1

    except Exception as e:
        logger.error(f"\n  Error during syllable video generation loop for {video_name}: {e}", exc_info=True)
    finally:
        writer.release()
        logger.info(f"  Finished writing {generated_frame_count}/{num_output_frames} frames for {out_path.name}")
        if generated_frame_count != num_output_frames and num_output_frames > 0:
             logger.warning(f"    Frame count mismatch for {out_path.name}!")
        # Optionally remove the file if frame count is zero? Depends on desired behavior.


def generate_all_syllable_videos() -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Main function to load KpMS data, filter it, generate syllable videos (with caching),
    and return the calculated frame ranges dictionary.
    """
    # --- Load Data ---
    results, video_frame_indexes, p3d_plot_used_frames = load_kpms_data()
    if results is None or video_frame_indexes is None or p3d_plot_used_frames is None:
        logger.critical("Failed to load necessary KpMS data. Aborting syllable processing.")
        return None

    # --- Build Series ---
    syllable_series_dict = build_syllable_series(results, video_frame_indexes)
    if not syllable_series_dict:
        logger.critical("Failed to build syllable series dictionary. Aborting.")
        return None

    # --- Filter Data & Get Frame Ranges ---
    filtered_data, frame_ranges = filter_syllable_data(syllable_series_dict, p3d_plot_used_frames)
    if not filtered_data:
        # If filtering resulted in no data, it's not necessarily critical, but worth noting.
        logger.warning("No valid data remaining after filtering syllable series based on used_frames.")
        # Return the (empty) frame_ranges dict. The main script should handle this.
        return frame_ranges # Return empty dict

    # --- Setup Output and Source Dirs ---
    output_dir = main_config.SYLLABLE_VIDEO_DIR
    pdf_dir = main_config.SYLLABLE_PDF_DIR

    logger.info(f"--- Checking/Generating Syllable Videos ---")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PDF source directory: {pdf_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Failed to create output directory {output_dir}: {e}. Aborting video generation.")
        return frame_ranges # Return ranges found so far, but can't generate videos

    if not convert_from_path:
         logger.error(f"pdf2image library not available. Cannot generate videos.")
         return frame_ranges # Return ranges, skip video gen

    if not pdf_dir.exists():
         logger.error(f"Syllable PDF directory not found: {pdf_dir}. Cannot generate videos.")
         return frame_ranges # Return ranges, skip video gen

    # --- Generate Videos (with Caching) ---
    skipped_count = 0
    generated_count = 0
    failed_count = 0
    videos_to_generate = list(filtered_data.items()) # Create list for stable iteration order if needed

    # Use tqdm directly on the list
    for video_name, series in tqdm(videos_to_generate, desc="Syllable Video Generation", unit="video"):
        out_path = output_dir / f"{video_name}.mp4"

        # --- Cache Check ---
        # Check if file exists and is non-empty (basic check)
        if out_path.exists() and out_path.stat().st_size > 100: # Check for > 100 bytes, arbitrary small size
            # logger.debug(f"Skipping existing video: {out_path.name}") # Can be verbose
            skipped_count += 1
            continue
        elif out_path.exists():
             logger.warning(f"Existing syllable video '{out_path.name}' is very small. Regenerating.")
             try: out_path.unlink() # Attempt to remove potentially corrupt file
             except OSError as e: logger.error(f"Failed to remove small file {out_path.name}: {e}")


        # Get the corresponding frame range using the canonical video_name key
        current_frame_range = frame_ranges.get(video_name)
        if current_frame_range is None:
            logger.error(f"Frame range not found for '{video_name}' during video generation phase. Skipping.")
            failed_count += 1
            continue

        try:
            generate_single_syllable_video(
                video_name,
                series,
                current_frame_range,
                output_dir,
                pdf_dir
            )
            generated_count += 1
        except Exception as e:
             logger.error(f"Unexpected error generating syllable video for '{video_name}': {e}", exc_info=True)
             failed_count += 1
             # Attempt to clean up potentially broken output file
             if out_path.exists():
                 try: out_path.unlink()
                 except OSError as ose: logger.error(f"Failed to remove broken file {out_path.name}: {ose}")


    # --- Final Summary ---
    logger.info(f"--- Finished Syllable Video Generation ---")
    logger.info(f"  Generated: {generated_count}")
    logger.info(f"  Skipped (Cached): {skipped_count}")
    logger.info(f"  Failed: {failed_count}")

    # --- Save Frame Ranges ---
    if frame_ranges:
        try:
            frame_range_path = main_config.FRAME_RANGES_PKL
            logger.info(f"Saving final frame ranges ({len(frame_ranges)} items) to: {frame_range_path}")
            with open(frame_range_path, "wb") as f:
                pickle.dump(frame_ranges, f)
            logger.info("Frame ranges saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save frame_ranges pickle to {frame_range_path}: {e}", exc_info=True)
    else:
        logger.warning("No frame ranges were generated or found during filtering. PKL file not saved.")


    return frame_ranges