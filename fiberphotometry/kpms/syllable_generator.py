# fiberphotometry/kpms/syllable_generator.py

import os
import logging
from pathlib import Path
from typing import List, Any, Dict, Tuple
import pandas as pd
import numpy as np
import keypoint_moseq as kpms
import jax
from jax_moseq.utils import set_mixed_map_iters
from .trimming import create_trimmed_coords_and_confs
from .batching import create_batches
import random

# --- SETUP LOGGING TO A FILE ---
log_file_path = Path(os.getcwd()) / 'model_fitting.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)

# ==============================================================================
# 1. NEW, SMARTER SUBSETTING HELPER FUNCTION
# (This replaces your old subset_data_random_window)
# ==============================================================================
def subset_data_and_indexes(
    coords_to_subset: Dict[str, np.ndarray],
    confs_to_subset: Dict[str, np.ndarray],
    indexes_to_subset: Dict[str, np.ndarray],
    subset_fraction: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Takes a random, continuous window from each data part and also slices
    the corresponding video frame indexes to create a new, consistent map.
    """
    subset_coords, subset_confs, subset_indexes = {}, {}, {}
    logging.info(f"Subsetting data and indexes to a {subset_fraction:.0%} random window.")

    for key, arr in coords_to_subset.items():
        total_frames = arr.shape[0]
        window_size = int(total_frames * subset_fraction)
        if window_size == 0: continue

        start_index = random.randint(0, total_frames - window_size)
        end_index = start_index + window_size

        subset_coords[key] = coords_to_subset[key][start_index:end_index]
        subset_confs[key] = confs_to_subset[key][start_index:end_index]
        # Create the new index map for the subset by slicing the original map
        subset_indexes[key] = indexes_to_subset[key][start_index:end_index]

    return subset_coords, subset_confs, subset_indexes


# ==============================================================================
# 2. UPDATED MAIN GENERATOR FUNCTION
# (This replaces your old generate_syllable_models)
# ==============================================================================
def generate_syllable_models(
    sessions: List[Any],
    project_dir: str,
    kappa: float,
    subset_fraction: float = 1.0,
    latent_dim: int = 4,
    num_models: int = 3,
    ar_iters: int = 1,
    n_jax_batches: int = 1,
    fitting_iters: int = 1,
    test_mode: bool = False
) -> Tuple[List[str], Dict, Dict, tuple]: # Cleaner return signature
    """
    Fits and saves keypoint-moseq models using the consolidated subsetting logic.
    """
    # --- Step 1 & 2: Setup and H5 Path finding (no changes needed) ---
    project_path = Path(project_dir)
    if not project_path.exists():
        kpms.setup_project(project_dir)
    config = kpms.load_config(project_dir)

    all_h5_paths = []
    for session in sessions:
        session.kpms_h5_path = None
        for df_name, df in session.dfs.data.items():
            if (df_name.startswith('tracking_') and 'h5_path' in df.columns and
                not df['h5_path'].empty and pd.notna(df['h5_path'].iloc[0]) and
                'Center' in df['h5_path'].iloc[0]):
                session.kpms_h5_path = df['h5_path'].iloc[0]
                all_h5_paths.append(session.kpms_h5_path)
                break

    # --- Step 3: Load and preprocess data (Consolidated Logic) ---
    logging.info("STEP 3: Loading and preprocessing data.")
    coordinates, confidences, _ = kpms.load_keypoints(all_h5_paths, 'deeplabcut')
    
    # Get ground-truth info from the FULL data
    original_lengths = {key: len(arr) for key, arr in coordinates.items()}
    trimmed_coords, trimmed_confs, _, full_video_frame_indexes = create_trimmed_coords_and_confs(
        coordinates, confidences, errors=None
    )

    # Prepare the data for fitting, calling the new helper function
    coords_for_fitting = trimmed_coords
    confs_for_fitting = trimmed_confs
    video_indexes_for_fitting = full_video_frame_indexes
    
    if subset_fraction < 1.0:
        coords_for_fitting, confs_for_fitting, video_indexes_for_fitting = subset_data_and_indexes(
            trimmed_coords, trimmed_confs, full_video_frame_indexes, subset_fraction
        )

    # Format the data for the model
    data, metadata = kpms.format_data(coords_for_fitting, confs_for_fitting, **config)

    # --- Step 4 & 5: PCA Fitting and Model Fitting Loop (no changes needed) ---
    # ... (This entire section of code remains exactly the same as yours) ...
    # It will correctly use the `data` and `metadata` from the subset.
    
    # --- 4. PCA Fitting ---
    pca_path = project_path / 'pca.p'
    if pca_path.exists() and test_mode: # Only use existing in test mode
        pca = kpms.load_pca(project_dir)
    else:
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, project_dir)
    kpms.update_config(project_dir, latent_dim=latent_dim)
    config = kpms.load_config(project_dir)

    # --- 5. Model Fitting Loop ---
    set_mixed_map_iters(n_jax_batches)

    final_model_names = []
    for model_idx in range(num_models):
        logging.info(f"--- Processing Model {model_idx + 1}/{num_models} ---")
        model = kpms.init_model(data, pca=pca, **config, seed=jax.random.PRNGKey(model_idx))
        model = kpms.update_hypparams(model, kappa=kappa)
        model, ar_model_name = kpms.fit_model(
            model, data, metadata, project_dir,
            ar_only=True, num_iters=ar_iters,
            generate_progress_plots=False
        )
        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, ar_model_name, iteration=ar_iters)
        model = kpms.update_hypparams(model, kappa=kappa/10)
        model_name = f'final_model_{kappa:.2e}_idx{model_idx}'
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=False, start_iter=current_iter, num_iters=current_iter + fitting_iters,
            generate_progress_plots=False
        )
        final_model_names.append(model_name)

    # --- Return the consolidated variables ---
    logging.info("Model fitting complete.")
    return final_model_names, original_lengths, video_indexes_for_fitting, metadata