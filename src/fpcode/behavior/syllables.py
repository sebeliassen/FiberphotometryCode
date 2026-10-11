"""Compatibility exports for canonical bouts and sequence operations."""

from .bouts import extract_bouts, extract_syllable_runs
from .sequences import (
                        LEGACY_DENSE_BOTH_ENDS,
                        LEGACY_SPARSE_NO_FILL,
                        flatten_results_sparse_legacy,
                        flatten_results_to_timeline,
                        tile_results,
)

__all__ = ["extract_bouts", "extract_syllable_runs", "tile_results", "flatten_results_to_timeline",
           "flatten_results_sparse_legacy", "LEGACY_DENSE_BOTH_ENDS", "LEGACY_SPARSE_NO_FILL"]
