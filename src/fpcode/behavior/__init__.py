"""
fpcode.behavior
===============
Behavioral analysis modules: kinematics, activity trimming,
Markov syllable transitions, and bout run-length dynamics.
"""

from .analysis.transitions import (
    compute_stationary_distribution,
    compute_transition_matrix,
)
from .observations.kinematics import (
    clean_short_runs,
    presence_mask_from_conf,
    smooth_centroid_median,
    smooth_heading_circular,
    subset_by_activity,
)
from .syllables import (
    extract_syllable_runs,
    flatten_results_to_timeline,
    tile_results,
)

__all__ = [
    "smooth_heading_circular",
    "smooth_centroid_median",
    "clean_short_runs",
    "presence_mask_from_conf",
    "subset_by_activity",
    "compute_transition_matrix",
    "compute_stationary_distribution",
    "extract_syllable_runs",
    "tile_results",
    "flatten_results_to_timeline",
]
