"""
fpcode.photometry
=================
Fiber photometry signal processing, response metrics, and temporal alignment.
"""

from .metrics import (
    calculate_signal_response_metrics,
    calculate_signal_response_metrics_matrix,
)
from .processing import (
    compute_mean_ci,
    group_signals,
    merge_phot_bonsai_cam,
)
from .alignment import (
    build_fixed_moseq_dfs,
    combined_photometry_and_syllables,
)

__all__ = [
    "calculate_signal_response_metrics",
    "calculate_signal_response_metrics_matrix",
    "group_signals",
    "compute_mean_ci",
    "merge_phot_bonsai_cam",
    "combined_photometry_and_syllables",
    "build_fixed_moseq_dfs",
]
