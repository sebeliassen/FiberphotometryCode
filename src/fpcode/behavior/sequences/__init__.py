"""Sequence operations with separate repetition and timeline policies."""

from .tiling import tile_results
from .timelines import (
                        LEGACY_DENSE_BOTH_ENDS,
                        LEGACY_SPARSE_NO_FILL,
                        flatten_results_sparse_legacy,
                        flatten_results_to_timeline,
)

__all__ = ["tile_results", "flatten_results_to_timeline", "flatten_results_sparse_legacy",
           "LEGACY_DENSE_BOTH_ENDS", "LEGACY_SPARSE_NO_FILL"]
