# fiberphotometry/kpms/syllable_appender.py
import os
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional
import pandas as pd
import numpy as np

import keypoint_moseq as kpms

# Standardized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def _basename_and_stem(path: str):
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    return base, stem


def _ensure_kpms_h5_paths(objs):
    """
    Best effort to populate obj.kpms_h5_path when missing.
    Mirrors the generator logic lightly:
      - if obj.kpms_h5_path exists, keep it
      - else if obj has .tracking (single DLC path), use that
      - else if obj has .dfs.data with a 'tracking_*' df and 'h5_path' col, use first row
    """
    for obj in objs:
        if getattr(obj, "kpms_h5_path", None):
            continue
        dlc_path = None
        if getattr(obj, "tracking", None):
            dlc_path = str(obj.tracking)
        elif hasattr(obj, "dfs") and hasattr(obj.dfs, "data"):
            for df_name, df in obj.dfs.data.items():
                if df_name.startswith("tracking_") and "h5_path" in df.columns and not df["h5_path"].empty:
                    candidate = df["h5_path"].iloc[0]
                    if pd.notna(candidate):
                        dlc_path = str(candidate)
                        break
        if dlc_path:
            setattr(obj, "kpms_h5_path", dlc_path)

def _normalize_root_from_result_key(k: str) -> str:
    base = k.split("_part")[0]
    base = base.split("_DLC_")[0]
    return base

def _normalize_root_from_path(p: str) -> str:
    name = os.path.splitext(os.path.basename(str(p)))[0]
    if "_DLC_" in name:
        name = name.split("_DLC_")[0]
    if name.endswith("_Crop_"):
        name = name[:-1]   # '..._Crop_' -> '..._Crop'
    return name

def get_aligned_results(session, model_name):
    store = getattr(session, 'kpms_results_by_model', None)
    if not store or model_name not in store:
        raise KeyError(f"No kpms results for model '{model_name}' on session {getattr(session, 'trial_id', '?')}")
    return store[model_name]

def get_aligned_syllables(session, model_name, key='syllables'):
    return get_aligned_results(session, model_name)[key]

def list_models_for_session(session):
    store = getattr(session, 'kpms_results_by_model', None)
    return list(store.keys()) if store else []

def stack_syllables_by_model(session, model_names, key='syllables'):
    return np.column_stack([get_aligned_results(session, m)[key] for m in model_names])

def extract_and_append_results(
    objs,
    project_dir,
    model_name,
    video_frame_indexes,
    original_lengths,
    metadata=None,
    data_key_name='all_results',
    verbose: bool = False,
):
    log = logging.getLogger(__name__)
    log.info("=== Starting syllable extraction/alignment for model: %s ===", model_name)

    # Load checkpoint + extract results
    try:
        try:
            kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        except Exception:
            if verbose:
                log.debug("reindex_syllables_in_checkpoint skipped/failed (non-fatal).")
        model, _, metadata_ckpt, ckpt_iter = kpms.load_checkpoint(project_dir, model_name)
        if verbose:
            log.info("Loaded checkpoint: %s (iter=%s)", model_name, ckpt_iter)
    except (FileNotFoundError, IndexError) as e:
        log.error("No checkpoints for model '%s': %s", model_name, e)
        return

    results = kpms.extract_results(model, metadata_ckpt, project_dir, model_name, save_results=False)
    log.info("Extracted results for %d parts.", len(results))

    # Group result part keys by normalized root
    results_by_root = {}
    for k in results.keys():
        r = _normalize_root_from_result_key(k)
        results_by_root.setdefault(r, []).append(k)

    # 🔧 Normalize original_lengths keys to the same root convention
    orig_len_by_root = {}
    for k, L in original_lengths.items():
        r = _normalize_root_from_result_key(k)  # safe for DLC keys or already-root keys
        L = int(L)
        # if multiple entries map to same root, keep the max (or first; max is safest)
        orig_len_by_root[r] = max(orig_len_by_root.get(r, 0), L)

    if verbose:
        log.info("[DEBUG] sample roots from results: %s",
                 sorted(results_by_root.keys())[:8])
        log.info("[DEBUG] sample roots from orig_len_by_root: %s",
                 sorted(orig_len_by_root.keys())[:8])
        log.info("[DEBUG] sample keys from video_frame_indexes: %s",
                 sorted(video_frame_indexes.keys())[:5])

    updated = 0
    for i, obj in enumerate(objs):
        obj_id = getattr(obj, 'trial_id', getattr(obj, 'name', f'obj{i+1}'))
        kp = getattr(obj, "kpms_h5_path", None) or getattr(obj, "tracking", None)
        if not kp:
            if verbose:
                log.warning("[%s] no kpms_h5_path/tracking path on object; skipping", obj_id)
            continue

        root = _normalize_root_from_path(kp)
        part_keys = results_by_root.get(root, [])
        if not part_keys:
            log.warning("[%s] no results parts found for root=%s (from path=%s)", obj_id, root, kp)
            continue

        orig_len = orig_len_by_root.get(root)
        if orig_len is None:
            log.warning("[%s] missing original length for root=%s; keys available: %s",
                        obj_id, root, list(orig_len_by_root.keys())[:3])
            continue

        if verbose:
            log.info("[%s] root=%s | parts=%d | orig_len=%d", obj_id, root, len(part_keys), orig_len)

        # Initialize full-length arrays from first part's shapes
        template = results[part_keys[0]]
        final_aligned = {}
        for key, arr in template.items():
            shape = (orig_len,) if arr.ndim == 1 else (orig_len, arr.shape[1])
            final_aligned[key] = np.full(shape, np.nan, dtype=np.float32)

        # Stitch each part using its exact part key in video_frame_indexes
        missing_idx_parts = []
        for pk in part_keys:
            idx = video_frame_indexes.get(pk)
            if idx is None:
                missing_idx_parts.append(pk)
                continue

            idx = np.asarray(idx, dtype=int)
            trimmed = results[pk]

            # sanity: keep only in-range indices
            good = (idx >= 0) & (idx < orig_len)
            if not np.all(good) and verbose:
                log.debug("[%s] clipping %d out-of-range indices for part=%s",
                          obj_id, (~good).sum(), pk)
            idx = idx[good]

            for key, arr in trimmed.items():
                arr_use = arr[:len(good)] if len(arr) != len(good) else arr
                arr_use = arr_use[good]
                final_aligned[key][idx] = arr_use

        if missing_idx_parts:
            log.warning("[%s] root=%s: missing frame indexes for parts: %s",
                        obj_id, root, missing_idx_parts)

        # Attach to object
        if not hasattr(obj, 'kpms_results_by_model') or getattr(obj, 'kpms_results_by_model') is None:
            setattr(obj, 'kpms_results_by_model', {})
        obj.kpms_results_by_model[model_name] = final_aligned

        # remove legacy projection to avoid confusion
        if hasattr(obj, data_key_name):
            try:
                delattr(obj, data_key_name)
            except Exception:
                pass

        updated += 1
        log.info("[%s] aligned results attached for model '%s'.", obj_id, model_name)

    log.info("=== Finished alignment. Updated %d objects. ===", updated)