"""
fpcode.common.file_io
=============
Generic bounded HDF5, NumPy and pickle file I/O.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger("fpcode.common.io")


def safe_load_hdf5_dataset(h5_path: Path, dataset_name: str = "df_with_missing") -> pd.DataFrame:
    """Safely loads a pandas DataFrame from HDF5."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return pd.read_hdf(path, key=dataset_name)




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


load_pickle_safe = safe_load_pickle
save_pickle_atomic = safe_save_pickle


def load_npz_safe(npz_path: Path) -> Dict[str, np.ndarray]:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_npz_atomic(data: Dict[str, Any], npz_path: Path) -> None:
    path = Path(npz_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(path)


def copy_dataset_bounded(src_ds, dst_group, name, dst_fd, chunk_bytes: int = 64 * 1024 * 1024):
    """
    Copies an HDF5 dataset in bounded chunks with proactive posix_fadvise page cache eviction,
    preserving dtype, shape, scalar data, and all dataset attributes.
    """
    import os

    shape = src_ds.shape
    dtype = src_ds.dtype
    total_bytes = src_ds.size * dtype.itemsize

    # Scalar dataset (shape == ()) or small dataset (< 16 MB)
    if shape == () or total_bytes < 16 * 1024 * 1024:
        dst_ds = dst_group.create_dataset(name, data=src_ds[()])
        for k in src_ds.attrs:
            dst_ds.attrs[k] = src_ds.attrs[k]
        return dst_ds

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

    for k in src_ds.attrs:
        dst_ds.attrs[k] = src_ds.attrs[k]
    return dst_ds


copy_dataset = copy_dataset_bounded


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



