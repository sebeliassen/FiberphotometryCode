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

import numpy as np
from typing import Dict, Iterable, Optional

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
            
        target_len = expected_lengths.get(rk)
        if not target_len:
            # Cannot validate, assume safe or pass through
            vfi_out[rk] = vfi_arr.copy()
            continue
            
        v_max = vfi_arr.max()
        
        # HEURISTIC:
        # If the max index is significantly smaller than the session length 
        # (e.g. half size for Rate=2), it is likely a legacy downsampled VFI.
        # Threshold: < (Target / (Scale * 0.75)) allowing for 25% buffer.
        threshold = target_len / (scale_factor * 0.75)
        
        if v_max < threshold:
            # LEGACY DETECTED
            if verbose:
                print(f"[Legacy VFI] Detected {rk}: Max={v_max} << Target={target_len}. Scaling x{scale_factor}.")
            
            # Scale and round to nearest int to map to absolute frames
            # e.g. [0, 1, 2] * 2.0 -> [0, 2, 4]
            new_arr = np.round(vfi_arr * scale_factor).astype(int)
            vfi_out[rk] = new_arr
            migrated_count += 1
        else:
            # Already absolute (or some other rate we don't handle)
            vfi_out[rk] = vfi_arr.copy()
            
    if migrated_count > 0:
        print(f"[Legacy Support] Migrated {migrated_count} sessions to Ground Truth VFI (x{scale_factor}).")
    elif verbose:
        print("[Legacy Support] No legacy VFI detected.")
        
    return vfi_out
