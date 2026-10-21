#!/usr/bin/env python3
"""
patch_checkpoint.py
=============================================================================
Transparent monkeypatch for Keypoint-MoSeq checkpoint loading.

Provides drop-in backward and forward compatibility:
1. Uncompacted (Stock) Checkpoints:
   - Passes through 100% identically to stock kpms.io.load_checkpoint.
   - Zero side-effects, zero performance overhead.
2. Compacted & Rolling Checkpoints:
   - Auto-detects 'is_packed' metadata attribute and available iterations.
   - Resolves default iteration (None) to the latest retained convergence iteration.
   - If 'data' group was omitted from checkpoint to bound disk/RAM, transparently
     loads and validates observation data from project's 'preprocessed_data.pkl'.
   - In 'slim' mode (if noise scales were stripped), safely virtualizes None
     fallbacks so code expecting dictionary keys does not raise a KeyError.
=============================================================================
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import keypoint_moseq as kpms
import numpy as np
from keypoint_moseq.io import _get_path, load_hdf5

_ORIGINAL_LOAD_CHECKPOINT = kpms.io.load_checkpoint


def is_checkpoint_packed(path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a checkpoint.h5 file has been compacted/packed.

    Returns
    -------
    is_packed : bool
    attrs_dict : dict
    """
    path = Path(path)
    if not path.exists():
        return False, {}

    try:
        with h5py.File(path, "r") as f:
            is_packed = bool(f.attrs.get("is_packed", False))
            packing_mode = str(f.attrs.get("packing_mode", "standard"))
            available = f.attrs.get("available_iterations", None)
            orig = f.attrs.get("original_iterations", None)
            avail_list = available.tolist() if available is not None else []
            orig_list = orig.tolist() if orig is not None else []
            return is_packed, {
                "is_packed": is_packed,
                "packing_mode": packing_mode,
                "available_iterations": avail_list,
                "original_iterations": orig_list,
            }
    except Exception:
        return False, {}


def patched_load_checkpoint(
    project_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    path: Optional[str] = None,
    iteration: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int]:
    """
    Drop-in replacement for keypoint_moseq.io.load_checkpoint that transparently
    supports legacy, rolling, and compacted checkpoints.
    """
    path = _get_path(project_dir, model_name, path, "checkpoint.h5")

    if not os.path.exists(path):
        if os.path.exists(f"{path}.packed"):
            path = f"{path}.packed"
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{path}'")

    with h5py.File(path, "r") as f:
        is_packed = bool(f.attrs.get("is_packed", False))
        packing_mode = str(f.attrs.get("packing_mode", "standard"))
        available_iters = f.attrs.get("available_iterations", None)
        orig_iters = f.attrs.get("original_iterations", None)
        has_data = "data" in f
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    if iteration is None:
        iteration = int(saved_iterations[-1])
    else:
        if iteration not in saved_iterations:
            if is_packed:
                avail_str = available_iters.tolist() if available_iters is not None else saved_iterations.tolist()
                orig_str = orig_iters.tolist() if orig_iters is not None else "unknown"
                raise KeyError(
                    f"Requested snapshot iteration {iteration}, but this checkpoint has been compacted "
                    f"({packing_mode} mode). Intermediate burn-in iterations were pruned. "
                    f"Available iterations: {avail_str} (Original run history was: {orig_str})."
                )
            else:
                raise AssertionError(
                    f"No snapshot found for iteration {iteration}. "
                    f"Available iterations are {saved_iterations}"
                )

    model = load_hdf5(path, f"model_snapshots/{iteration}")
    metadata = load_hdf5(path, "metadata")

    if has_data:
        data = load_hdf5(path, "data")
    else:
        # Resolve observation data from preprocessed_data.pkl in project directory
        resolved_proj = Path(project_dir) if project_dir else Path(path).parent.parent
        pkl_path = resolved_proj / "preprocessed_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as pf:
                cached_obj = pickle.load(pf)
            data = cached_obj["data"] if (isinstance(cached_obj, dict) and "data" in cached_obj) else cached_obj
            # Validate shape/identity consistency between metadata and data
            if isinstance(metadata, (tuple, list)) and len(metadata) >= 2:
                rec_keys = metadata[0]
                if "Y" in data:
                    assert len(rec_keys) == data["Y"].shape[0], (
                        f"Metadata keys ({len(rec_keys)}) mismatch data sequences ({data['Y'].shape[0]})"
                    )
        else:
            raise KeyError(
                f"'data' group was not found in checkpoint '{path}', and source cache '{pkl_path}' does not exist."
            )

    # Virtualize missing noise fields if in slim mode
    if is_packed and packing_mode == "slim":
        if "states" in model and "s" not in model["states"]:
            model["states"]["s"] = None
        if "noise_prior" not in model:
            model["noise_prior"] = None

    return model, data, metadata, iteration


def apply_kpms_checkpoint_patch() -> None:
    """Apply the checkpoint monkeypatch to keypoint_moseq globally."""
    kpms.io.load_checkpoint = patched_load_checkpoint
    kpms.load_checkpoint = patched_load_checkpoint
    print("[kpms_patch] Checkpoint compaction monkeypatch active.")


def remove_kpms_checkpoint_patch() -> None:
    """Restore stock keypoint_moseq checkpoint loader."""
    kpms.io.load_checkpoint = _ORIGINAL_LOAD_CHECKPOINT
    kpms.load_checkpoint = _ORIGINAL_LOAD_CHECKPOINT
    print("[kpms_patch] Restored stock keypoint_moseq loader.")
