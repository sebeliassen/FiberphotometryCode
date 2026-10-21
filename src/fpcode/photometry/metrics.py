"""Compatibility exports; new code imports fpcode.photometry.response_metrics."""

from .response_metrics import (
    METRIC_NAMES,
    calculate_signal_response_metrics,
    calculate_signal_response_metrics_matrix,
    legacy_signal_response_metrics,
    legacy_signal_response_metrics_matrix,
)

__all__ = ["METRIC_NAMES", "calculate_signal_response_metrics", "calculate_signal_response_metrics_matrix",
           "legacy_signal_response_metrics", "legacy_signal_response_metrics_matrix"]
