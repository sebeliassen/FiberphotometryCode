# fiberphotometry/kpms/syllable_generator.py
"""
This module provides functions to fit and save keypoint-moseq models based on
session data from the fiberphotometry project structure.
"""
import os
import logging
from pathlib import Path
from typing import List, Any, Dict, Tuple
import pandas as pd
import numpy as np

import keypoint_moseq as kpms
import jax

# Local modules from your project
from .trimming import create_trimmed_coords_and_confs
from .batching import create_batches

# Standardized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_syllable_models(
    sessions: List[Any],
    project_dir: str,
    kappa: float,
    latent_dim: int = 8,
    num_models: int = 3,
    ar_iters: int = 1,
    fitting_iters: int = 1,
    test_mode: bool = False
) -> Tuple[List[str], Dict, Dict]:
    """
    Fits and saves keypoint-moseq models. This function finds all relevant H5
    files from the session objects, preprocesses the data, fits a PCA model
    (or loads an existing one), and then fits one or more AR-HMM models.

    Args:
        sessions (List[Any]): List of session objects to be analyzed.
        project_dir (str): Path to the keypoint-moseq project directory.
        kappa (float): The concentration parameter for the transition matrix.
        latent_dim (int): Number of principal components to use.
        num_models (int): Number of models to train (from different random seeds).
        ar_iters (int): Number of iterations for AR-only fitting.
        fitting_iters (int): Number of fitting iterations for full model fitting.
        test_mode (bool): If True, runs the analysis on only the first detected
                          H5 file for a quick technical test.

    Returns:
        A tuple containing:
        - final_model_names (List[str]): Names of the trained models.
        - video_frame_indexes (dict): Maps result part keys to their original frame indexes.
        - original_lengths (dict): Maps H5 basenames to their original, untrimmed length.
    """
    # 1. Setup Project & Config
    project_path = Path(project_dir)
    if not project_path.exists():
        logging.info(f"Project directory '{project_dir}' not found. Creating it.")
        kpms.setup_project(project_dir)
    else:
        logging.info(f"Using existing project directory: '{project_dir}'")
    config = kpms.load_config(project_dir)

    # 2. Find all H5 paths and assign them to session objects in a single loop
    all_h5_paths = []
    logging.info("Finding H5 paths and assigning them to session objects...")
    for session in sessions:
        session.kpms_h5_path = None
        for df_name, df in session.dfs.data.items():
            if (df_name.startswith('tracking_') and
                'h5_path' in df.columns and
                not df['h5_path'].empty and
                pd.notna(df['h5_path'].iloc[0]) and
                'Center' in df['h5_path'].iloc[0]):
                h5_path = df['h5_path'].iloc[0]
                session.kpms_h5_path = h5_path
                all_h5_paths.append(h5_path)
                break
        if session.kpms_h5_path is None:
            logging.warning(f"Could not find a 'Center' H5 path for session with trial_id {session.trial_id}")

    if test_mode:
        logging.warning("--- TEST MODE ENABLED ---")
        if not all_h5_paths:
            raise ValueError("Cannot run in test mode, no H5 files were found.")
        all_h5_paths = [all_h5_paths[0]]
        logging.info(f"Test mode will process only: {all_h5_paths[0]}")
    
    if not all_h5_paths:
        raise ValueError("Could not find any H5 files to process.")
    logging.info(f"Found {len(all_h5_paths)} total H5 files to process.")

    # 3. Load and preprocess data
    coordinates, confidences, _ = kpms.load_keypoints(all_h5_paths, 'deeplabcut')
    original_lengths = {key: len(arr) for key, arr in coordinates.items()}

    trimmed_coords, trimmed_confs, _, video_frame_indexes = create_trimmed_coords_and_confs(
        coordinates, confidences, errors=None
    )
    data, metadata = kpms.format_data(trimmed_coords, trimmed_confs, **config)

    # 4. PCA Fitting
    pca_path = project_path / 'pca.p'
    if pca_path.exists():
        logging.info(f"Found existing PCA model at '{pca_path}'. Loading it.")
        pca = kpms.load_pca(project_dir)
    else:
        logging.info("No existing PCA model found. Fitting and saving a new one.")
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, project_dir)
    kpms.update_config(project_dir, latent_dim=latent_dim)
    config = kpms.load_config(project_dir)

    # 5. Model Fitting Loop
    final_model_names = []
    for model_idx in range(num_models):
        logging.info(f"--- Processing model index: {model_idx} ---")
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
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        logging.info(f"Completed model fitting. Final model name: '{model_name}'.")
        final_model_names.append(model_name)

    logging.info("Model fitting complete.")
    return final_model_names, video_frame_indexes, original_lengths