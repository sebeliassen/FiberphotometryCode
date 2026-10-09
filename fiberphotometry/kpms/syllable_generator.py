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


def ensure_kpms_project(project_dir):
    """
    Ensure a valid keypoint-moseq project at `project_dir`.
    - Creates the parent dir if needed.
    - If config.yml is missing, force a (re)setup with overwrite=True.
    - Always return a loaded config.
    """
    project_path = Path(project_dir)
    project_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = project_path / "config.yml"
    if not cfg_path.exists():
        # Directory may exist but be empty — force creation of config.yml
        kpms.setup_project(str(project_path), overwrite=True)

    return kpms.load_config(str(project_path))

def collect_kpms_paths(objs):
    """
    Accepts a list of either 'sessions' (your current API) OR 'trials' (your new objects)
    and returns (all_h5_paths, objs) where:
      - all_h5_paths is an ordered list of DLC file paths (HDF5 or CSV) for kpms.load_keypoints
      - each obj gets a .kpms_h5_path attribute set (for bookkeeping)
    We DO NOT load DLC ourselves; we just pass paths to kpms.load_keypoints in the given order.
    """
    paths = []
    for obj in objs:
        dlc_path = None

        # 1) If the object already has kpms_h5_path, use it
        if hasattr(obj, "kpms_h5_path") and obj.kpms_h5_path:
            dlc_path = obj.kpms_h5_path

        # 2) Trial-style: prefer the 'tracking' attribute you’ve set up earlier
        if dlc_path is None and hasattr(obj, "tracking") and obj.tracking:
            dlc_path = str(obj.tracking)

        # 3) Session-style: look for a df named like your current API and pull the first 'h5_path'
        if dlc_path is None and hasattr(obj, "dfs") and hasattr(obj.dfs, "data"):
            for df_name, df in obj.dfs.data.items():
                if df_name.startswith("tracking_") and "h5_path" in df.columns and not df["h5_path"].empty:
                    candidate = df["h5_path"].iloc[0]
                    if pd.notna(candidate):
                        dlc_path = str(candidate)
                        break

        # 4) Attach for traceability and append if found
        setattr(obj, "kpms_h5_path", dlc_path)
        if dlc_path:
            paths.append(dlc_path)
        else:
            logging.warning(f"Skipping {getattr(obj, 'trial_id', getattr(obj, 'name', 'unknown'))}: no DLC path found")

    if not paths:
        raise ValueError("No DLC paths found in provided objects.")

    return paths, objs

def apply_manual_windows_override(
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    manual_windows: Dict[str, tuple],       # {'<video_root or path>': (start_frame, end_frame)}
    include_only_listed: bool = True,
    end_inclusive: bool = True,
    verbose: bool = True,
):
    """
    MANUAL WINDOWS **OVERRIDE** with verbose diagnostics.
    Slices the *raw* coordinates/confidences by provided (start,end) frame ranges.
    Returns sliced coords/confs AND a 'video_frame_indexes' mapping (original frame numbers).
    """
    # Normalize keys once
    mw_norm = _normalize_manual_windows_keys(manual_windows)

    if verbose:
        print("\n[Manual Windows Override] Diagnostics")
        print("  • Provided manual_windows keys:", list(manual_windows.keys())[:3], "..." if len(manual_windows) > 3 else "")
        print("  • Normalized roots:", list(mw_norm.keys())[:6], "..." if len(mw_norm) > 6 else "")

    coords_sel, confs_sel, frame_idx_sel = {}, {}, {}
    dropped_not_listed = []
    dropped_empty = []
    kept_parts = 0
    kept_frames_total = 0

    # Show a few available part keys and derived roots
    if verbose:
        sample_parts = list(coordinates.keys())[:5]
        print("  • Sample KPMS part keys:", sample_parts)
        sample_roots = [_video_root_from_part_key(k) for k in sample_parts]
        print("  • Their derived roots   :", sample_roots)

    for part_key, coords_arr in coordinates.items():
        n = coords_arr.shape[0]
        root = _video_root_from_part_key(part_key)

        has_window = root in mw_norm
        if verbose:
            print(f"\n  [{part_key}] n={n}")
            print(f"    → root: {root!r} | has manual window? {has_window}")

        if not has_window:
            if include_only_listed:
                dropped_not_listed.append(root)
                if verbose:
                    print("    → dropped (not listed and include_only_listed=True)")
                continue
            # keep full part
            coords_sel[part_key] = coords_arr
            confs_sel[part_key]  = confidences[part_key]
            frame_idx_sel[part_key] = np.arange(n, dtype=int)
            kept_parts += 1
            kept_frames_total += n
            if verbose:
                print("    → kept full part (no manual window provided; include_only_listed=False)")
            continue

        start_f, end_f = mw_norm[root]
        # guard against weird inputs
        if start_f is None or end_f is None:
            dropped_empty.append(root)
            if verbose:
                print(f"    → dropped (None window): {start_f}, {end_f}")
            continue

        # clip to bounds
        lo_raw, hi_raw = int(start_f), int(end_f)
        lo = max(0, lo_raw)
        hi = min(n - 1, hi_raw)
        if end_inclusive:
            slice_idx = np.arange(lo, hi + 1, dtype=int)
        else:
            slice_idx = np.arange(lo, hi, dtype=int)

        if verbose:
            print(f"    window requested: [{start_f}, {end_f}] | clipped to data: [{lo}, {hi}] | end_inclusive={end_inclusive}")
            print(f"    resulting slice length: {slice_idx.size}")

        if slice_idx.size == 0:
            dropped_empty.append(root)
            if verbose:
                print("    → dropped (empty after clipping)")
            continue

        coords_sel[part_key] = coords_arr[slice_idx]
        confs_sel[part_key]  = confidences[part_key][slice_idx]
        frame_idx_sel[part_key] = slice_idx
        kept_parts += 1
        kept_frames_total += slice_idx.size
        if verbose:
            print("    → kept (sliced)")

    if not coords_sel:
        # Strong hinting to fix key mismatches
        avail_roots = sorted({_video_root_from_part_key(k) for k in coordinates.keys()})
        if verbose:
            print("\n[Manual Windows Override] RESULT: no parts kept.")
            print("  • Normalized manual roots:", sorted(mw_norm.keys()))
            print("  • Available data roots    :", avail_roots[:10], "..." if len(avail_roots) > 10 else "")
            if dropped_not_listed:
                print(f"  • Dropped (not listed) count: {len(dropped_not_listed)} (showing up to 10):",
                      sorted(set(dropped_not_listed))[:10])
            if dropped_empty:
                print(f"  • Dropped (empty windows) count: {len(dropped_empty)} (showing up to 10):",
                      sorted(set(dropped_empty))[:10])
        raise ValueError("After applying manual windows override, no frames remain. "
                         "Check your (start,end) ranges and that your manual window keys match the derived roots.")

    if verbose:
        print(f"\n[Manual Windows Override] kept_parts={kept_parts}, kept_frames_total={kept_frames_total}")
        if dropped_not_listed:
            print(f"  • dropped_not_listed (unique roots): {len(set(dropped_not_listed))}")
        if dropped_empty:
            print(f"  • dropped_empty (unique roots): {len(set(dropped_empty))}")

    return coords_sel, confs_sel, frame_idx_sel

def intersect_trimmed_with_windows(
    trimmed_coords: Dict[str, np.ndarray],
    trimmed_confs: Dict[str, np.ndarray],
    full_video_frame_indexes: Dict[str, np.ndarray],
    manual_windows: Dict[str, tuple],
    include_only_listed: bool = True,
    end_inclusive: bool = True,
):
    """
    AND-combine KPMS trimming with your manual frame windows.
    Works on the *trimmed* dicts using their full_video_frame_indexes (orig frame #s).
    """
    out_coords, out_confs, out_idx = {}, {}, {}

    for part_key, coords_arr in trimmed_coords.items():
        idx_orig = full_video_frame_indexes[part_key]  # len == len(trimmed sample)
        root = _video_root_from_part_key(part_key)

        if root not in manual_windows:
            if include_only_listed:
                continue   # drop parts not listed
            out_coords[part_key] = coords_arr
            out_confs[part_key]  = trimmed_confs[part_key]
            out_idx[part_key]    = idx_orig
            continue

        start_f, end_f = manual_windows[root]
        mask = (idx_orig >= start_f) & (idx_orig <= end_f) if end_inclusive else \
               (idx_orig >= start_f) & (idx_orig <  end_f)

        if not np.any(mask):
            continue

        out_coords[part_key] = coords_arr[mask]
        out_confs[part_key]  = trimmed_confs[part_key][mask]
        out_idx[part_key]    = idx_orig[mask]

    return out_coords, out_confs, out_idx

import re

def _canon_crop_suffix(s: str) -> str:
    """
    Canonicalize '..._Crop' vs '..._Crop_' to exactly '..._Crop'.
    Also trims whitespace just in case.
    """
    s = s.strip()
    # collapse both '..._Crop' and '..._Crop_' to '..._Crop'
    return re.sub(r'(_Crop)_*$', r'\1', s)

def _video_root_from_part_key(part_key: str) -> str:
    """
    Map KPMS part keys like
      'T10_Tracking_video..._Crop_DLC_resnet50_..._760000' (or ..._part1)
    to the canonical root 'T10_Tracking_video..._Crop'.
    """
    base = part_key.split('_part')[0]
    if '_DLC_' in base:
        base = base.split('_DLC_', 1)[0]
    else:
        base = base.split('DLC_', 1)[0]
    return _canon_crop_suffix(base)

def _normalize_manual_windows_keys(manual_windows: dict) -> dict:
    """
    Normalize user-provided manual window keys so they match our canonical root.
    Accepts paths (with dirs/exts) or bare roots; returns keys as '..._Crop'.
    """
    exts = ('.mp4', '.avi', '.mov', '.mkv', '.csv', '.h5')
    out = {}
    for k, v in manual_windows.items():
        base = k.split('/')[-1].split('\\')[-1]  # strip dirs
        for ext in exts:
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        # strip DLC suffix if present
        if '_DLC_' in base:
            base = base.split('_DLC_', 1)[0]
        elif 'DLC_' in base:
            base = base.split('DLC_', 1)[0]
        out[_canon_crop_suffix(base)] = v
    return out


# ==============================================================================
# 2. UPDATED MAIN GENERATOR FUNCTION
# (This replaces your old generate_syllable_models)
# ==============================================================================
def generate_syllable_models(
    sessions,  # can be sessions OR trials
    project_dir: str,
    kappa: float,
    subset_fraction: float = 1.0,
    latent_dim: int = 4,
    num_models: int = 3,
    ar_iters: int = 25,
    n_jax_batches: int = 1,
    fitting_iters: int = 100,
    test_mode: bool = False,
    manual_windows: Dict[str, tuple] = None,   # <-- NEW
):
    project_path = Path(project_dir)
    config = ensure_kpms_project(project_path)

    all_h5_paths, sessions = collect_kpms_paths(sessions)

    # --- Step 3: Load and preprocess data (Consolidated Logic) ---
    logging.info("STEP 3: Loading and preprocessing data.")
    # Load DLC → full arrays
    coordinates, confidences, _ = kpms.load_keypoints(all_h5_paths, 'deeplabcut')

    # For bookkeeping, keep full lengths (unchanged)
    original_lengths = {key: len(arr) for key, arr in coordinates.items()}

    # ====== CHOOSE PATH ======
    if manual_windows:
        # OVERRIDE: ignore KPMS automatic trimming, use only your (start,end) windows
        coords_for_fitting, confs_for_fitting, video_indexes_for_fitting = apply_manual_windows_override(
            coordinates, confidences, manual_windows,
            include_only_listed=True, end_inclusive=True
        )
    else:
        # Original auto-trim path
        trimmed_coords, trimmed_confs, _, full_video_frame_indexes = create_trimmed_coords_and_confs(
            coordinates, confidences, errors=None
        )
        coords_for_fitting = trimmed_coords
        confs_for_fitting  = trimmed_confs
        video_indexes_for_fitting = full_video_frame_indexes

    # Optional: random contiguous subset *within the already chosen frames*
    if subset_fraction < 1.0:
        coords_for_fitting, confs_for_fitting, video_indexes_for_fitting = subset_data_and_indexes(
            coords_for_fitting, confs_for_fitting, video_indexes_for_fitting, subset_fraction
        )

    # Format for the model
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

        kpms.print_dims_to_explain_variance(pca, 0.9)
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
            ar_only=True, num_iters=ar_iters
        )
        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, ar_model_name, iteration=ar_iters)
        model = kpms.update_hypparams(model, kappa=kappa/10)
        model_name = f'final_model_{kappa:.2e}_idx{model_idx}'
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, model_name,
            ar_only=False, start_iter=current_iter, num_iters=current_iter + fitting_iters
        )
        final_model_names.append(model_name)

    # --- Return the consolidated variables ---
    logging.info("Model fitting complete.")
    return final_model_names, original_lengths, video_indexes_for_fitting, metadata