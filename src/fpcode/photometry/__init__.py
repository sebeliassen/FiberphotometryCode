"""
fpcode.photometry
=================
Fiber photometry signal processing, response metrics, and temporal alignment.
"""

from importlib import import_module

from fpcode.statistics import compute_mean_ci

from .aggregation import group_signals
from .response_metrics import (
    calculate_signal_response_metrics,
    calculate_signal_response_metrics_matrix,
)

_LEGACY_EXPORTS = {
    "merge_phot_bonsai_cam": "fpcode.photometry.processing",
    "combined_photometry_and_syllables": "fpcode.photometry.alignment",
    "build_fixed_moseq_dfs": "fpcode.photometry.alignment",
}

__all__ = [
    "calculate_signal_response_metrics",
    "calculate_signal_response_metrics_matrix",
    "group_signals",
    "compute_mean_ci",
    "merge_phot_bonsai_cam",
    "combined_photometry_and_syllables",
    "build_fixed_moseq_dfs",
]


def __getattr__(name):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LEGACY_EXPORTS[name]), name)
    globals()[name] = value
    return value
