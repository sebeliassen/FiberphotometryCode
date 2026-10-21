"""Preserve the latest historical VFI copy policy without rate inference.

The historical API name is retained for old callers. It does not detect legacy
clocks or validate that an input uses native video frames. Rescaling was removed
before the refactor because length heuristics corrupted trimmed frame mappings.
"""

from typing import Dict

import numpy as np


def migrate_legacy_vfi(
    vfi_dict: Dict[str, np.ndarray],
    expected_lengths: Dict[str, int],
    scale_factor: float = 2.0,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Return independent array copies without changing indices or their dtype.

    expected_lengths and scale_factor remain accepted for historical callers.
    The last pre-refactor policy removed rate inference and automatic scaling;
    this entry point does not convert relative indices to native video frames.
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
