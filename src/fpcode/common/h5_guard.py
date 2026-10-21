"""
fpcode.common.h5_guard
======================
Immutability guard for production checkpoints and analytical datasets.
Blocks in-place write access (r+, w, a) on Tier-A artifacts.
"""

from pathlib import Path
from typing import Any, Union

import h5py

IMMUTABLE_PATTERNS = [
    "*checkpoint*.h5",
    "*results*.h5",
    "*results_reindexed*.h5",
    "*pca*.p",
    "*pca*.pkl",
    "*init_model*.pkl",
]

WRITE_MODES = {"r+", "w", "w-", "x", "a"}


def is_immutable_artifact(path: Union[str, Path]) -> bool:
    """Returns True if the file matches Tier-A protected naming patterns."""
    p = Path(path)
    name_lower = p.name.lower()
    for pat in IMMUTABLE_PATTERNS:
        clean_pat = pat.replace("*", "").lower()
        if clean_pat in name_lower:
            return True
    return False


def check_h5_immutability(path: Union[str, Path], mode: str = "r") -> None:
    """Raises PermissionError if a write mode is requested on an immutable file."""
    clean_mode = mode.lower().strip()
    if clean_mode in WRITE_MODES and is_immutable_artifact(path):
        raise PermissionError(
            f"[h5_guard] BLOCKED in-place write access (mode='{mode}') on immutable Tier-A artifact: '{path}'.\n"
            f"Per architectural rules, existing checkpoints must NEVER be modified in place.\n"
            f"Write new outputs to a separate path atomically (temp + fsync + rename)."
        )


def safe_open_h5(path: Union[str, Path], mode: str = "r", **kwargs: Any) -> h5py.File:
    """Defensive wrapper around h5py.File that enforces immutability on Tier-A artifacts."""
    check_h5_immutability(path, mode=mode)
    return h5py.File(path, mode=mode, **kwargs)
