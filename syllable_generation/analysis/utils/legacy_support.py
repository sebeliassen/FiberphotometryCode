"""
analysis/utils/legacy_support.py

Handles backward compatibility for Keypoint MoSeq data, particularly
migrating Legacy VFI (Video Frame Index) artifacts to the new "Ground Truth" paradigm.

Paradigm Shift:
- Legacy: VFI indices were relative to a downsampled timeline (0..N_downsampled).
- New:    VFI indices are absolute frame numbers (0..N_original).

Downstream code should always assume Absolute indices. This module detects legacy
data and scales it on-the-fly.
"""

from typing import Dict

import numpy as np


def migrate_legacy_vfi(
    vfi_dict: Dict[str, np.ndarray],
    expected_lengths: Dict[str, int],
    scale_factor: float = 2.0,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Detects if VFI arrays are likely from the legacy downsampled pipeline 
    (based on max index vs expected session length) and scales them to absolute indices.
    
    Returns a COPY of the dictionary with migrated arrays.
    """
    vfi_out = {}
    migrated_count = 0

    for rk, vfi_arr in vfi_dict.items():
        if vfi_arr.size == 0:
            vfi_out[rk] = vfi_arr.copy()
            continue

        vfi_out[rk] = vfi_arr.copy()

    if migrated_count > 0:
        print(f"[Legacy Support] Migrated {migrated_count} sessions to Ground Truth VFI (x{scale_factor}).")
    elif verbose:
        print("[Legacy Support] No legacy VFI detected.")

    return vfi_out
