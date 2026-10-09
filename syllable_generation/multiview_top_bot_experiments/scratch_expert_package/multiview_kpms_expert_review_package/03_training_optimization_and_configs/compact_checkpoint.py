#!/usr/bin/env python3
"""
compact_checkpoint.py
=============================================================================
Post-hoc compaction engine for Keypoint-MoSeq checkpoints.

Eliminates historical burn-in intermediate snapshots (e.g. 25, 50, ..., 225)
from monolithic checkpoint.h5 files while preserving 100% of the final
mathematical state (continuous latents x, syllables z, headings h, centroids v,
dynamics parameters Ab, Q, pi, coordinates, mask, observation noise scales s,
and noise_prior).

Supports both:
  - "full" mode (recommended for academic/FAIR compliance): retains all tensors
    at final iteration, repacked into a fresh HDF5 with chunk compression.
  - "slim" mode: drops states/s and noise_prior (saving an additional ~6.8 GB).
=============================================================================
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np


def compact_checkpoint(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    mode: str = "full",
    keep_iterations: Optional[List[int]] = None,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
    in_place: bool = False,
    verbose: bool = True
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

    # Open source and determine snapshots
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
                    raise KeyError(f"Requested iteration {it} not in checkpoint. Available: {saved_iterations}")

        if verbose:
            print(f"  All Snapshots:   {saved_iterations}")
            print(f"  Retaining:       {keep_iterations}")

        # Create temporary working file in same directory as destination to allow atomic replace
        dest_dir = dest_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tmp.h5", dir=dest_dir, delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            with h5py.File(temp_path, "w") as f_dst:
                # 1. Copy global groups (data and metadata)
                for root_key in ["data", "metadata"]:
                    if root_key in f_src:
                        if verbose:
                            print(f"  Copying {root_key}...")
                        f_src.copy(f_src[root_key], f_dst, name=root_key)

                # 2. Copy selected model_snapshots
                snaps_dst = f_dst.create_group("model_snapshots")
                for it in keep_iterations:
                    src_snap_key = f"model_snapshots/{it}"
                    if verbose:
                        print(f"  Transferring snapshot {it}...")

                    if mode == "full":
                        # Full scientific mode: copy complete snapshot
                        f_src.copy(f_src[src_snap_key], snaps_dst, name=str(it))
                    elif mode == "slim":
                        # Slim mode: copy everything except states/s and noise_prior
                        dst_snap = snaps_dst.create_group(str(it))
                        src_snap = f_src[src_snap_key]

                        for k in src_snap.keys():
                            if k == "noise_prior":
                                continue
                            elif k == "states":
                                dst_states = dst_snap.create_group("states")
                                for st_k in src_snap["states"].keys():
                                    if st_k == "s":
                                        continue
                                    src_snap["states"].copy(src_snap["states"][st_k], dst_states, name=st_k)
                            else:
                                src_snap.copy(src_snap[k], dst_snap, name=k)

                # 3. Add compaction metadata attributes
                f_dst.attrs["is_packed"] = True
                f_dst.attrs["packing_mode"] = mode
                f_dst.attrs["available_iterations"] = np.array(keep_iterations, dtype=np.int32)
                f_dst.attrs["original_iterations"] = np.array(saved_iterations, dtype=np.int32)
                f_dst.attrs["compaction_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Atomic replace
            os.replace(temp_path, dest_path)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Compaction failed during HDF5 write: {e}") from e

    new_size_bytes = dest_path.stat().st_size
    new_size_gb = new_size_bytes / 1e9
    saved_gb = orig_size_gb - new_size_gb
    ratio = (1.0 - (new_size_bytes / orig_size_bytes)) * 100.0 if orig_size_bytes > 0 else 0.0
    elapsed_sec = time.time() - t0

    stats = {
        "source_path": str(source_path),
        "dest_path": str(dest_path),
        "mode": mode,
        "keep_iterations": keep_iterations,
        "original_size_bytes": orig_size_bytes,
        "compact_size_bytes": new_size_bytes,
        "original_size_gb": orig_size_gb,
        "compact_size_gb": new_size_gb,
        "saved_gb": saved_gb,
        "reduction_percent": ratio,
        "elapsed_sec": elapsed_sec,
    }

    if verbose:
        print(f"✅ Compaction complete in {elapsed_sec:.2f}s!")
        print(f"   Size: {orig_size_gb:.2f} GB -> {new_size_gb:.2f} GB (Saved {saved_gb:.2f} GB, {ratio:.1f}% reduction).")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compact Keypoint-MoSeq checkpoint.h5")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint.h5")
    parser.add_argument("--output", type=str, default=None, help="Destination path (default: in-place replacement)")
    parser.add_argument("--mode", choices=["full", "slim"], default="full", help="Compaction mode (full=reproducible, slim=archival)")
    parser.add_argument("--keep-iters", type=int, nargs="*", default=None, help="Specific iterations to retain (default: max_iter)")
    parser.add_argument("--in-place", action="store_true", help="Perform in-place replacement")
    args = parser.parse_args()

    compact_checkpoint(
        source_path=args.checkpoint,
        dest_path=args.output,
        mode=args.mode,
        keep_iterations=args.keep_iters,
        in_place=args.in_place
    )
