import numpy as np
from tqdm import tqdm
from analysis.timepoint_analysis import aggregate_signals


def calculate_signal_response_metrics(signal, interval):
    # Assuming 'interval' is a slice or range, adjust signal accordingly
    adjusted_signal = signal[interval[0]:interval[1]+1] if isinstance(interval, (list, tuple, range)) else signal

    # Calculate peak timing once, as it's used multiple times
    peak_idx = np.argmax(adjusted_signal)

    # Slopes calculation
    maxima = adjusted_signal[peak_idx]
    left_minima = adjusted_signal[0]
    right_minima = adjusted_signal[-1]

    slope_up = ((maxima - left_minima) / peak_idx if
                peak_idx != 0 else float('inf'))  # Avoid division by zero
    slope_down = ((maxima - right_minima) / (len(adjusted_signal) - peak_idx) if
                  peak_idx != len(adjusted_signal) else float('inf'))  # Adjust for index

    response_metrics = {
        'slope_up': slope_up,
        'slope_down': slope_down,
        'maximal_value': maxima,
        'peak_timing': peak_idx,
        'auc': np.trapz(adjusted_signal, dx=1)
    }

    return response_metrics