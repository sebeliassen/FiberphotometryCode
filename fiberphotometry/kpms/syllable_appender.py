# fiberphotometry/kpms/syllable_appender.py
import os
import logging
from pathlib import Path
from typing import List, Any, Dict
import pandas as pd
import numpy as np

import keypoint_moseq as kpms

# Standardized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_and_append_results(
    sessions: List[Any],
    project_dir: str,
    model_name: str,
    video_frame_indexes: Dict,
    original_lengths: Dict,
    data_key_name: str = 'all_results'
):
    """
    Loads the latest checkpoint of a trained model, finds all result "parts" for
    each session, and attaches them as a dictionary to a new attribute on the
    session object.
    """
    logging.info(f"--- Starting syllable extraction and alignment for model: {model_name} ---")

    # 1. Load the latest checkpoint
    try:
        model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
        logging.info(f"Successfully loaded model '{model_name}' from checkpoint {current_iter}.")
    except (FileNotFoundError, IndexError):
        logging.error(f"Could not find any checkpoints for model '{model_name}'. Aborting.")
        return

    # 2. Extract results directly from the model
    results = kpms.extract_results(model, metadata, project_dir, model_name, save_results=False)
    logging.info(f"Successfully extracted results for {len(results)} video parts.")

    # 3. Create the final NaN-padded data structure for each session
    sessions_updated = 0
    for session in sessions:
        if not hasattr(session, 'kpms_h5_path') or not session.kpms_h5_path:
            continue

        h5_basename = os.path.basename(session.kpms_h5_path)
        h5_key_prefix = h5_basename.removesuffix('.h5')

        # --- THIS IS THE FIX ---
        # The key for the `original_lengths` dictionary does not have the .h5 extension.
        lookup_key = h5_basename.removesuffix('.h5')
        original_len = original_lengths.get(lookup_key)
        # -----------------------
        
        if original_len is None:
            logging.warning(f"Could not find original length for {h5_basename}. Skipping alignment.")
            continue

        session_part_keys = [k for k in results if k.startswith(h5_key_prefix)]
        if not session_part_keys:
            logging.warning(f"No result parts found for H5 file '{h5_basename}'.")
            continue

        # Initialize the final, full-length NaN arrays based on the structure of the first part
        final_aligned_results = {}
        first_part_data = results[session_part_keys[0]]
        for data_key, sample_array in first_part_data.items():
            nan_shape = (original_len,) if sample_array.ndim == 1 else (original_len, sample_array.shape[1])
            final_aligned_results[data_key] = np.full(nan_shape, np.nan, dtype=np.float32)

        # Loop through all parts and place their data onto the final arrays
        for part_key in session_part_keys:
            original_indices = video_frame_indexes[part_key]
            trimmed_data_dict = results[part_key]
            for data_key, trimmed_array in trimmed_data_dict.items():
                final_aligned_results[data_key][original_indices] = trimmed_array
        
        # Attach the single dictionary of full-length arrays to the session
        setattr(session, data_key_name, final_aligned_results)
        logging.info(f"Created final aligned data structure for session {session.trial_id}.")
        sessions_updated += 1

    logging.info(f"--- Finished alignment. Updated {sessions_updated} sessions. ---")