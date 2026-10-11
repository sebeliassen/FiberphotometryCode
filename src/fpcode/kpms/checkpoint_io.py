"""Bounded checkpoint I/O without importing Keypoint-MoSeq, JAX or CUDA."""

import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from fpcode.common.file_io import copy_group_bounded
from fpcode.common.logging import get_logger

logger = get_logger("fpcode.kpms.checkpoint_io")


def inspect_checkpoint_iterations(ckpt_path: Path) -> List[int]:
    """List numeric model snapshot iterations; retain nonnumeric groups separately."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with h5py.File(path, "r") as checkpoint:
        iterations = []
        for key in checkpoint.get("model_snapshots", {}):
            try:
                iterations.append(int(key))
            except ValueError:
                pass
    return sorted(iterations)


def compact_kpms_checkpoint(
    checkpoint_path: Path,
    keep_iterations: Optional[List[int]] = None,
    chunk_mb: int = 64,
    *,
    dest_path: Optional[Path] = None,
) -> Path:
    """Stream selected snapshots and metadata; publish by atomic replacement.

    Omitting ``dest_path`` preserves the historical in-place API. A distinct
    destination leaves the source untouched. Exact snapshot names, nonnumeric
    snapshot groups and root/group/dataset attributes are retained.
    """
    source = Path(checkpoint_path).resolve()
    destination = source if dest_path is None else Path(dest_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    if isinstance(chunk_mb, bool) or not isinstance(chunk_mb, int) or chunk_mb < 1:
        raise ValueError("chunk_mb must be a positive integer")
    if keep_iterations is not None and any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in keep_iterations
    ):
        raise ValueError("keep_iterations must contain integers")

    temporary = None
    with h5py.File(source, "r") as original:
        snapshots = original.get("model_snapshots", {})
        numeric = {}
        nonnumeric = []
        for key in snapshots:
            try:
                iteration = int(key)
            except ValueError:
                nonnumeric.append(key)
                continue
            if iteration in numeric:
                raise ValueError(f"Ambiguous snapshot names for iteration {iteration}")
            numeric[iteration] = key
        saved = sorted(numeric)
        retained = ([saved[-1]] if saved else []) if keep_iterations is None else sorted(set(keep_iterations))
        missing = set(retained) - set(saved)
        if missing:
            raise ValueError(f"Requested iterations not found: {sorted(missing)}; available: {saved}")
        if destination == source and original.attrs.get("is_packed", False) and saved == retained and not nonnumeric:
            return source
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="checkpoint.compact.", suffix=".h5",
                                             delete=False) as handle:
                temporary = Path(handle.name)
            with h5py.File(temporary, "w") as output:
                try:
                    descriptor = output.id.get_vfd_handle()
                except Exception:
                    descriptor = None
                for name in ("metadata", "data"):
                    if name in original:
                        copy_group_bounded(original[name], output.create_group(name), descriptor,
                                           chunk_bytes=chunk_mb * 1024 * 1024)
                selected = output.create_group("model_snapshots")
                if "model_snapshots" in original:
                    for name, value in original["model_snapshots"].attrs.items():
                        selected.attrs[name] = value
                for key in [numeric[value] for value in retained] + nonnumeric:
                    copy_group_bounded(snapshots[key], selected.create_group(key), descriptor,
                                       chunk_bytes=chunk_mb * 1024 * 1024)
                for name, value in original.attrs.items():
                    output.attrs[name] = value
                output.attrs["is_packed"] = True
                output.attrs["packing_mode"] = "bounded_stream"
                output.attrs["available_iterations"] = np.array(retained, dtype=np.int32)
                output.attrs["original_iterations"] = np.array(saved, dtype=np.int32)
                output.attrs["packed_iterations"] = np.array(retained, dtype=np.int32)
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                output.attrs["compacted_at"] = stamp
                output.attrs["compaction_timestamp"] = stamp
                output.flush()
                if descriptor is not None:
                    try:
                        os.fdatasync(descriptor)
                        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError, TypeError):
                        pass
            # Close the original before replacing it, including for in-place use.
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("Compacted %s, retaining iterations %s", destination, retained)
    return destination
