"""Keep historical response APIs while numerical kernels live in fpcode."""

from fpcode.photometry.response_metrics import (
    legacy_signal_response_metrics as calculate_signal_response_metrics,
    legacy_signal_response_metrics_matrix as calculate_signal_response_metrics_matrix,
)

__all__ = ["calculate_signal_response_metrics", "calculate_signal_response_metrics_matrix"]
