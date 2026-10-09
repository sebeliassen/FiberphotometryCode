"""
fpcode.photometry.metrics
=========================
Pure numerical response metrics and signal calculation for fiber photometry traces.
"""

from typing import Dict, Sequence, Tuple, Union

import numpy as np


def calculate_signal_response_metrics(
    signal: np.ndarray,
    interval: Union[Tuple[int, int], Sequence[int], range] = None,
) -> Dict[str, float]:
    """
    Computes summary response metrics on a 1D signal trace:
      - slope_up: rate of rise from start to peak
      - slope_down: rate of fall from peak to end
      - maximal_value: peak signal amplitude
      - peak_timing: index of peak relative to interval start
      - auc: trapezoidal area under curve
    """
    signal = np.asarray(signal)
    if interval is not None:
        start, end = interval[0], interval[1]
        adjusted_signal = signal[start : end + 1]
    else:
        adjusted_signal = signal

    if len(adjusted_signal) == 0:
        return {
            "slope_up": 0.0,
            "slope_down": 0.0,
            "maximal_value": 0.0,
            "peak_timing": 0.0,
            "auc": 0.0,
        }

    peak_idx = int(np.argmax(adjusted_signal))
    maxima = float(adjusted_signal[peak_idx])
    left_minima = float(adjusted_signal[0])
    right_minima = float(adjusted_signal[-1])

    slope_up = (maxima - left_minima) / peak_idx if peak_idx != 0 else float("inf")
    slope_down = (
        (maxima - right_minima) / (len(adjusted_signal) - peak_idx)
        if peak_idx != len(adjusted_signal)
        else float("inf")
    )

    try:
        auc_val = float(np.trapezoid(adjusted_signal, dx=1))
    except AttributeError:
        auc_val = float(np.trapz(adjusted_signal, dx=1))

    return {
        "slope_up": slope_up,
        "slope_down": slope_down,
        "maximal_value": maxima,
        "peak_timing": float(peak_idx),
        "auc": auc_val,
    }


def calculate_signal_response_metrics_matrix(
    signals: np.ndarray,
    interval: Union[Tuple[int, int], Sequence[int], range] = None,
    fps: float = 60.0,
) -> np.ndarray:
    """
    Batch vectorized computation of response metrics across a 2D matrix (trials x time).
    Returns (trials, 5) array: [slope_up, slope_down, maxima, peak_time_sec, auc].
    """
    signals = np.asarray(signals)
    if interval is not None:
        start, end = interval[0], interval[1]
        signals = signals[:, start : end + 1]

    if signals.shape[1] == 0:
        return np.zeros((signals.shape[0], 5), dtype=np.float64)

    maxima = np.max(signals, axis=1)
    left_minima = signals[:, 0]
    right_minima = signals[:, -1]

    peak_indices = np.argmax(signals, axis=1) / fps

    slope_up = np.divide(maxima - left_minima, peak_indices, where=peak_indices != 0)
    slope_down_dxs = (signals.shape[1] / fps) - peak_indices
    slope_down = np.divide(maxima - right_minima, slope_down_dxs, where=slope_down_dxs != 0)

    try:
        auc = np.trapezoid(signals, dx=1, axis=1)
    except AttributeError:
        auc = np.trapz(signals, dx=1, axis=1)

    return np.column_stack((slope_up, slope_down, maxima, peak_indices, auc))
