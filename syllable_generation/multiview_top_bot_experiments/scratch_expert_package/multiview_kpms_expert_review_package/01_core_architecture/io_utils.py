"""
core.io_utils
=============
Safe I/O utilities for HDF5 datasets, Keypoint-MoSeq checkpoints,
and pickle session databases.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import h5py
import numpy as np
import pandas as pd
import pickle

from .logging_utils import get_logger

logger = get_logger("core.io_utils")

def safe_load_hdf5_dataset(h5_path: Path, dataset_name: str = "df_with_missing") -> pd.DataFrame:
    """Safely loads a pandas DataFrame from HDF5."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return pd.read_hdf(path, key=dataset_name)

def inspect_checkpoint_iterations(ckpt_path: Path) -> List[int]:
    """Returns sorted list of saved model snapshot iterations in a checkpoint."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with h5py.File(path, "r") as f:
        if "model_snapshots" not in f:
            return []
        iters = [int(k) for k in f["model_snapshots"].keys() if k.isdigit()]
        return sorted(iters)

def safe_load_pickle(pkl_path: Path) -> Any:
    """Loads a pickle file with clear error logging."""
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    logger.info(f"Loading pickle: {path.name} ({path.stat().st_size / (1024**2):.1f} MB)...")
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_save_pickle(data: Any, pkl_path: Path) -> None:
    """Saves data to a pickle file using an atomic temporary swap."""
    path = Path(pkl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    logger.info(f"Saved pickle: {path.name} ({path.stat().st_size / (1024**2):.1f} MB)")


def copy_dataset_bounded(src_ds, dst_group, name, dst_fd, chunk_bytes: int = 64 * 1024 * 1024):
    """
    Copies an HDF5 dataset in bounded chunks with proactive posix_fadvise page cache eviction.
    """
    import os
    shape = src_ds.shape
    dtype = src_ds.dtype
    total_bytes = src_ds.size * dtype.itemsize

    if total_bytes < 16 * 1024 * 1024:
        dst_group.create_dataset(name, data=src_ds[()])
        return

    chunks = src_ds.chunks if src_ds.chunks else True
    dst_ds = dst_group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

    bytes_per_row = (total_bytes // shape[0]) if shape[0] > 0 else total_bytes
    step = max(1, chunk_bytes // bytes_per_row) if bytes_per_row > 0 else 1

    for start_idx in range(0, shape[0], step):
        end_idx = min(start_idx + step, shape[0])
        dst_ds[start_idx:end_idx] = src_ds[start_idx:end_idx]
        if dst_fd is not None:
            try:
                os.fdatasync(dst_fd)
                os.posix_fadvise(dst_fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass


def copy_group_bounded(src_grp, dst_grp, dst_fd, chunk_bytes: int = 64 * 1024 * 1024):
    """
    Recursively copies an HDF5 group in bounded chunks.
    """
    for k in src_grp.attrs:
        dst_grp.attrs[k] = src_grp.attrs[k]

    for key, item in src_grp.items():
        if isinstance(item, h5py.Dataset):
            copy_dataset_bounded(item, dst_grp, key, dst_fd, chunk_bytes=chunk_bytes)
        elif isinstance(item, h5py.Group):
            sub_dst = dst_grp.create_group(key)
            copy_group_bounded(item, sub_dst, dst_fd, chunk_bytes=chunk_bytes)


def compact_kpms_checkpoint(
    checkpoint_path: Path,
    keep_iterations: Optional[List[int]] = None,
    chunk_mb: int = 64,
) -> Path:
    """
    Zero-OOM standalone-compatible compaction engine with bounded chunk streaming
    and POSIX_FADV_DONTNEED page cache eviction.

    Preserves:
    1. /metadata and /data root groups
    2. Selected model snapshot iterations (default: highest available iteration)
    3. All root and group attributes
    """
    import os
    import time
    import psutil

    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    rss0 = psutil.Process().memory_info().rss / 1e6
    t0 = time.time()
    orig_sz = ckpt_path.stat().st_size / 1e9
    logger.info(f"[compact_kpms_checkpoint] Starting compaction on: {ckpt_path}")
    logger.info(f"  Initial Process RSS: {rss0:.1f} MB | Original Size: {orig_sz:.2f} GB")

    with h5py.File(ckpt_path, "r") as f_src:
        saved_iters = sorted([int(k) for k in f_src["model_snapshots"].keys()])
        if keep_iterations is None:
            keep_iterations = [max(saved_iters)]
        else:
            keep_iterations = sorted(list(set(keep_iterations)))

        logger.info(f"  Available Snapshots: {saved_iters}")
        logger.info(f"  Retaining Snapshots: {keep_iterations}")

        if f_src.attrs.get("is_packed", False) and set(saved_iters) == set(keep_iterations):
            logger.info("  [Optimization] Checkpoint is already packed and contains only requested iterations.")
            return ckpt_path

        tmp_path = ckpt_path.with_name(f"checkpoint.compact.tmp_{os.getpid()}.h5")

        with h5py.File(tmp_path, "w") as f_dst:
            try:
                dst_fd = f_dst.id.get_vfd_handle()
            except Exception:
                dst_fd = None

            # 1. Global groups (metadata, data)
            for rk in ["metadata", "data"]:
                if rk in f_src:
                    sub_dst = f_dst.create_group(rk)
                    copy_group_bounded(f_src[rk], sub_dst, dst_fd, chunk_bytes=chunk_mb * 1024 * 1024)

            # 2. Selected snapshots
            snaps_dst = f_dst.create_group("model_snapshots")
            for it in keep_iterations:
                logger.info(f"  Transferring snapshot {it} via bounded chunks ({chunk_mb} MiB)...")
                it_dst = snaps_dst.create_group(str(it))
                copy_group_bounded(f_src[f"model_snapshots/{it}"], it_dst, dst_fd, chunk_bytes=chunk_mb * 1024 * 1024)

            # 3. Global attributes
            for k in f_src.attrs:
                f_dst.attrs[k] = f_src.attrs[k]
            f_dst.attrs["is_packed"] = True
            f_dst.attrs["packing_mode"] = "bounded_stream"
            f_dst.attrs["available_iterations"] = np.array(keep_iterations, dtype=np.int32)
            f_dst.attrs["original_iterations"] = np.array(saved_iters, dtype=np.int32)
            f_dst.attrs["packed_iterations"] = np.array(keep_iterations, dtype=np.int32)
            f_dst.attrs["compacted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f_dst.attrs["compaction_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if dst_fd is not None:
                try:
                    os.fdatasync(dst_fd)
                    os.posix_fadvise(dst_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass

    # Atomic rename replace
    tmp_path.replace(ckpt_path)
    dur = time.time() - t0
    final_sz = ckpt_path.stat().st_size / 1e9
    rss_end = psutil.Process().memory_info().rss / 1e6
    logger.info(
        f"✅ Compaction complete in {dur:.2f}s! Size: {orig_sz:.2f} GB -> {final_sz:.2f} GB | "
        f"Final RSS: {rss_end:.1f} MB (Delta: {rss_end - rss0:+.1f} MB)"
    )
    return ckpt_path

