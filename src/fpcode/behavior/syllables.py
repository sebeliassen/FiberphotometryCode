"""Compatibility exports for canonical bouts and sequence operations."""

from .primitives.bouts import extract_bouts, extract_syllable_runs
from .sequences.tiling import tile_results
from .sequences.timelines import (
                        LEGACY_DENSE_BOTH_ENDS,
                        LEGACY_SPARSE_NO_FILL,
                        flatten_results_sparse_legacy,
                        flatten_results_to_timeline,
)

__all__ = ["extract_bouts", "extract_syllable_runs", "tile_results", "flatten_results_to_timeline",
           "flatten_results_sparse_legacy", "LEGACY_DENSE_BOTH_ENDS", "LEGACY_SPARSE_NO_FILL"]
