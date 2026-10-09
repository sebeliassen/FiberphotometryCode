"""
fpcode.kpms
===========
Keypoint-MoSeq (KPMS) pipeline modules: model evaluation, stable label reindexing,
identity auditing, and behavioral metrics.
"""

from .evaluation import (
    IncompatibleModelsError,
    invert_permutation,
    is_permutation,
    mapping_hash,
    old_to_new_map,
    read_checkpoint_identity,
    reconstruct_mask,
    relabel_states,
    stable_frequency_order,
    validate_model_compatibility,
)
from .metrics import (
    extract_syllable_statistics,
    get_frequencies,
)

__all__ = [
    "stable_frequency_order",
    "invert_permutation",
    "is_permutation",
    "mapping_hash",
    "relabel_states",
    "old_to_new_map",
    "reconstruct_mask",
    "read_checkpoint_identity",
    "validate_model_compatibility",
    "IncompatibleModelsError",
    "get_frequencies",
    "extract_syllable_statistics",
]
