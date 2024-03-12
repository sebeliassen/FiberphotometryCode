import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from data.data_loading import DataContainer
from config import *
from collections import defaultdict
from utils import find_start_end_idxs

from scipy.signal import savgol_filter
import plotly.graph_objects as go

interval_start = peak_interval_config["interval_start"]
interval_end = peak_interval_config["interval_end"]


def get_signal_around_event(session, event_type, brain_region, add_response_metrics_to_session=False):
    if add_response_metrics_to_session:
        from analysis.response_metrics import calculate_signal_response_metrics
        response_interval = find_start_end_idxs(event_type)
        if not hasattr(session, 'response_metrics'):
            session.response_metrics = defaultdict(list)            

    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"This brain region is not available. \
                         Available ones are {list(session.fiber_to_region.values())}")
    
    # Ensure event are fetched from the event_idxs_container
    event_type_idxs = session.event_idxs_container.get_data(event_type)
    if event_type_idxs is None or len(event_type_idxs) == 0:
        raise ValueError(f"No event_type_idxs found for {event_type}")

    raw_df = session.df_container.get_data("raw")
    phot_df = session.df_container.get_data("photwrit_470")
    phot_times = phot_df['SecFromZero_FP3002'].values
    
    all_data = np.zeros((len(event_type_idxs), interval_start + interval_end))

    for i, event_idx in enumerate(event_type_idxs):
        event_time = raw_df['SecFromZero_FP3002'].loc[event_idx]
        event_idx = np.searchsorted(phot_times, event_time, side='right')

        # Check if there are enough data points after this event_time to form a complete interval
        if event_idx == len(phot_times) or event_idx + interval_end > len(phot_times):
            continue  
        
        # Assuming interval_start is defined, ensure there's enough data before the event_time
        if event_idx < interval_start:
            continue  
        
        start_idx = max(0, event_idx - interval_start)  
        end_idx = event_idx + interval_end  
        
        signal = phot_df[f'{brain_region}_phot_zF'].iloc[start_idx:end_idx].values
        if add_response_metrics_to_session:
            curr_response_metrics = calculate_signal_response_metrics(signal, response_interval)
            for response_metric, value in curr_response_metrics.items():
                session.response_metrics[(event_type, brain_region.split('_')[0], response_metric)].append(value)
        all_data[i, :len(signal)] = signal  
    time_axis = np.arange(-interval_start, interval_end)

    return time_axis[:all_data.shape[1]], all_data


def collect_signals(sessions, event_type, regions_to_aggregate, 
                    aggregate_by_session=False, add_response_metrics_to_sessions=False):
    fps = PLOTTING_CONFIG['fps']
    time_axis = np.arange(-interval_start, interval_end) / fps

    start_time, end_time = attr_interval_dict[event_type]
    # start_time, end_time = -2.5, 2.5
    start_event_idx = int(start_time * fps + interval_start)
    end_event_idx = int(end_time * fps + interval_start)
    
    all_signals = []
    for session in sessions:
        if len(session.event_idxs_container.data.get(event_type, [])) == 0:
            continue
        
        for region in regions_to_aggregate:
            if region in session.brain_regions:
                _, signal_data = get_signal_around_event(session, event_type, region, add_response_metrics_to_sessions)
                if aggregate_by_session:
                    signal_data = signal_data.mean(axis=0, keepdims=True)
                all_signals.append(signal_data)

    all_signals = np.vstack(all_signals)
    return time_axis, all_signals, (start_event_idx, end_event_idx)


def aggregate_signals(sessions, event_type, regions_to_aggregate, aggregate_by_session=False, 
                      normalize_baseline=True, n=None):
    time_axis, all_signals, (start_event_idx, end_event_idx) = \
        collect_signals(sessions, event_type, regions_to_aggregate, aggregate_by_session)

    if normalize_baseline:
        all_signals -= np.mean(all_signals[:, start_event_idx-7:start_event_idx+7], axis=1, keepdims=True)

    if all_signals.size > 0:
        if n:
            sample_idxs = np.random.choice(all_signals.shape[0], size=n, replace=False)
            all_signals = all_signals[sample_idxs]
        mean_signal = np.mean(all_signals, axis=0)
        
        # Smoothing
        window_length = 5
        window = np.ones(window_length) / window_length
        smoothed_mean_signal = np.convolve(mean_signal, window, 'same')

        sem_signal = stats.sem(all_signals, axis=0)
        ci_95 = sem_signal * stats.t.ppf((1 + 0.95) / 2., len(all_signals)-1)

        lower_bound = smoothed_mean_signal - ci_95
        upper_bound = smoothed_mean_signal + ci_95

        return time_axis, smoothed_mean_signal, lower_bound, upper_bound, (start_event_idx, end_event_idx), len(all_signals)
    else:
        print(f"No sessions contained timepoints for '{event_type}', so no plot was generated.")