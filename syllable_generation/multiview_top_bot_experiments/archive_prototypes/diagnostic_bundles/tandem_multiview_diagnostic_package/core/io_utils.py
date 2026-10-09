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
