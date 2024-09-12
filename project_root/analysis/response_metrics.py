import numpy as np
from tqdm import tqdm
import config

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


def calculate_signal_response_metrics_matrix(signals, interval):
    # Adjust signals according to interval
    if isinstance(interval, (list, tuple, range)):
        signals = signals[:, interval[0]:interval[1]+1]
    fps = config.PLOTTING_CONFIG['fps']
    
    maxima = np.max(signals, axis=1)
    left_minima = signals[:, 0]
    right_minima = signals[:, -1]
    
    peak_indices = np.argmax(signals, axis=1) / fps
    
    # Calculating slopes with np.divide to handle division by zero gracefully
    slope_up = np.divide(maxima - left_minima, peak_indices, where=peak_indices!=0)
    slope_down_dxs = signals.shape[1] / fps - peak_indices
    slope_down = np.divide(maxima - right_minima, slope_down_dxs, where=slope_down_dxs!=0)

    auc = np.trapz(signals, dx=1, axis=1)
    
    result_matrix = np.vstack((slope_up, slope_down, maxima, peak_indices, auc)).T
    
    #TODO: remember that you know have saved the response metric - index 
    # correspondance in the config
    metric_names = {
        0: 'slope_up',
        1: 'slope_down',
        2: 'maximal_value',
        3: 'peak_timing',
        4: 'auc'
    }

    return result_matrix, metric_names
