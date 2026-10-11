"""
fpcode.kpms.checkpoints
=======================
Checkpoints compaction, packing detection, and transparent loading monkeypatches
for Keypoint-MoSeq models.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import keypoint_moseq as kpms
import numpy as np
from keypoint_moseq.io import _get_path, load_hdf5

_ORIGINAL_LOAD_CHECKPOINT = kpms.io.load_checkpoint


def compact_checkpoint(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    mode: str = "full",
    keep_iterations: Optional[List[int]] = None,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
    in_place: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compact a Keypoint-MoSeq checkpoint.h5 file.

    Parameters
    ----------
    source_path : str or Path
        Path to the input checkpoint.h5 file.
    dest_path : str or Path, optional
        Path where the compacted checkpoint will be written. If None and in_place=True,
        atomically replaces source_path.
    mode : {"full", "slim"}, default="full"
        - "full": Preserves all tensors at keep_iterations (including states/s and noise_prior).
        - "slim": Omits states/s and noise_prior to maximize storage savings.
    keep_iterations : list of int, optional
        Specific iteration numbers to keep. Defaults to [max_iter].
    compression : str or None, default="gzip"
        HDF5 dataset compression filter (e.g., "gzip", "lzf", or None).
    compression_opts : int, default=4
        Compression level for gzip (1-9).
    in_place : bool, default=False
        If True, replaces source_path atomically upon completion.
    verbose : bool, default=True
        Whether to print progress details.

    Returns
    -------
    dict
        Compaction summary with sizes, compression ratio, and timings.
    """
    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found at {source_path}")

    if dest_path is None:
        if not in_place:
            raise ValueError("Must provide dest_path or set in_place=True.")
        dest_path = source_path
    else:
        dest_path = Path(dest_path).resolve()

    orig_size_bytes = source_path.stat().st_size
    orig_size_gb = orig_size_bytes / 1e9

    t0 = time.time()
    if verbose:
        print(f"[compact] Starting compaction of: {source_path}")
        print(f"  Mode:            {mode}")
        print(f"  Original Size:   {orig_size_gb:.2f} GB ({orig_size_bytes:,} bytes)")

    with h5py.File(source_path, "r") as f_src:
        if "model_snapshots" not in f_src:
            raise ValueError(f"Malformed checkpoint: missing 'model_snapshots' group in {source_path}")

        saved_iterations = sorted([int(k) for k in f_src["model_snapshots"].keys()])
        if len(saved_iterations) == 0:
            raise ValueError(f"No snapshot iterations found in {source_path}")

        max_iter = max(saved_iterations)
        if keep_iterations is None:
            keep_iterations = [max_iter]
        else:
            keep_iterations = sorted(list(set(keep_iterations)))

        for it in keep_iterations:
            if it not in saved_iterations:
                raise ValueError(f"Requested iteration {it} not found in {source_path}. Available: {saved_iterations}")

        parent_dir = dest_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=parent_dir, prefix="compact_", suffix=".h5", delete=False) as tmp_f:
            tmp_path = Path(tmp_f.name)

        try:
            with h5py.File(tmp_path, "w") as f_dst:
                for root_key in ["data", "metadata"]:
                    if root_key in f_src:
                        f_src.copy(f_src[root_key], f_dst, name=root_key)

                snaps_dst = f_dst.create_group("model_snapshots")
                if "model_snapshots" in f_src:
                    for ak, av in f_src["model_snapshots"].attrs.items():
                        snaps_dst.attrs[ak] = av

                for it in keep_iterations:
                    src_snap_key = f"model_snapshots/{it}"
                    if mode == "full":
                        f_src.copy(f_src[src_snap_key], snaps_dst, name=str(it))
                    elif mode == "slim":
                        dst_snap = snaps_dst.create_group(str(it))
                        src_snap = f_src[src_snap_key]
                        for ak, av in src_snap.attrs.items():
                            dst_snap.attrs[ak] = av

                        for k in src_snap.keys():
                            if k == "noise_prior":
                                continue
                            elif k == "states":
                                dst_states = dst_snap.create_group("states")
                                for sak, sav in src_snap["states"].attrs.items():
                                    dst_states.attrs[sak] = sav
                                for st_k in src_snap["states"].keys():
                                    if st_k == "s":
                                        continue
                                    src_snap["states"].copy(src_snap["states"][st_k], dst_states, name=st_k)
                            else:
                                src_snap.copy(src_snap[k], dst_snap, name=k)

                f_dst.attrs["is_packed"] = True
                f_dst.attrs["packing_mode"] = mode
                f_dst.attrs["original_iterations"] = np.array(saved_iterations, dtype=np.int64)
                f_dst.attrs["available_iterations"] = np.array(keep_iterations, dtype=np.int64)


            tmp_path.replace(dest_path)

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    new_size_bytes = dest_path.stat().st_size
    new_size_gb = new_size_bytes / 1e9
    reclaimed_bytes = orig_size_bytes - new_size_bytes
    reclaimed_gb = reclaimed_bytes / 1e9
    ratio = (orig_size_bytes / new_size_bytes) if new_size_bytes > 0 else 0.0
    elapsed = time.time() - t0

    if verbose:
        print(f"[compact] Completed in {elapsed:.2f}s")
        print(f"  New Size:        {new_size_gb:.2f} GB ({new_size_bytes:,} bytes)")
        print(f"  Reclaimed:       {reclaimed_gb:.2f} GB ({ratio:.2f}x smaller)")

    saved_gb = orig_size_gb - new_size_gb
    reduction_percent = (1.0 - (new_size_bytes / orig_size_bytes)) * 100.0 if orig_size_bytes > 0 else 0.0

    return {
        "source_path": str(source_path),
        "dest_path": str(dest_path),
        "mode": mode,
        "keep_iterations": keep_iterations,
        "original_size_bytes": orig_size_bytes,
        "compact_size_bytes": new_size_bytes,
        "original_size_gb": orig_size_gb,
        "compact_size_gb": new_size_gb,
        "saved_gb": saved_gb,
        "reduction_percent": reduction_percent,
        "elapsed_sec": elapsed,
        # Modern aliases
        "source": str(source_path),
        "destination": str(dest_path),
        "compacted_bytes": new_size_bytes,
        "reclaimed_bytes": reclaimed_bytes,
        "compression_ratio": ratio,
        "elapsed_seconds": elapsed,
        "retained_iterations": keep_iterations,
    }



def is_checkpoint_packed(path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
    """Check if a checkpoint.h5 file has been compacted/packed."""
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
    iteration: Optional[int] = None,
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
        resolved_proj = Path(project_dir) if project_dir else Path(path).parent.parent
        pkl_path = resolved_proj / "preprocessed_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as pf:
                cached_obj = pickle.load(pf)
            data = cached_obj["data"] if (isinstance(cached_obj, dict) and "data" in cached_obj) else cached_obj
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
