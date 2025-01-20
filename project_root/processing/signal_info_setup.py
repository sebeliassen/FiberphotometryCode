from collections import defaultdict
from analysis.response_metrics import calculate_signal_response_metrics_matrix
from utils import find_start_end_idxs
import config
import numpy as np

interval_start = config.peak_interval_config["interval_start"]
interval_end = config.peak_interval_config["interval_end"]


def get_brain_region_event_signal_info(session, event_type, brain_region):
    fps = config.PLOTTING_CONFIG[session.session_type]['fps']
    response_interval = find_start_end_idxs(event_type, fps)

    if not hasattr(session, 'response_metrics'):
        session.response_metrics = defaultdict(list)            

    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"This brain region is not available. \
                         Available ones are {list(session.fiber_to_region.values())}")
    
    # Ensure event are fetched from the event_idxs_container
    event_type_idxs = session.event_idxs_container.get_data(event_type)
    if len(session.event_idxs_container.data.get(event_type, [])) == 0:
        raise ValueError(f"No event_type_idxs found for {event_type}")

    fiber_color = brain_region[-1]
    curr_phot_freq = config.LETTER_TO_FREQS[fiber_color]

    raw_df = session.dfs.get_data("raw")
    phot_df = session.dfs.get_data(f"phot_{curr_phot_freq}")
    phot_times = phot_df['SecFromZero_FP3002'].values
    
    signal_matrix = np.zeros((len(event_type_idxs), interval_start + interval_end))
    signal_idx_ranges = []

    for i, event_idx in enumerate(event_type_idxs):
        event_time = raw_df['SecFromZero_FP3002'].loc[event_idx]
        event_idx = np.searchsorted(phot_times, event_time, side='right')

        # Check if there are enough data points after this event_time to form a complete interval
        if event_idx == len(phot_times) or event_idx + interval_end > len(phot_times):
            continue  
        
        # Assuming interval_start is defined, ensure there's enough data before the event_time
        if event_idx < interval_start:
            continue  
        
        start_signal_idx = max(0, event_idx - interval_start)  
        end_signal_idx = event_idx + interval_end  
        
        signal = phot_df[brain_region + ('phot_zF',)].iloc[start_signal_idx:end_signal_idx].values.copy()
        signal_idx_ranges.append((start_signal_idx, end_signal_idx))

        start_event_idx = response_interval[0]
        normalized_signal = signal - np.mean(signal[start_event_idx-7:start_event_idx+7])

        signal_matrix[i] = normalized_signal

    #response_interval = find_start_end_idxs(event_type, fps)
    response_metrics = calculate_signal_response_metrics_matrix(signal_matrix, response_interval, fps)
    brain_region_event_signal_info = {
        'signal_matrix': signal_matrix, 
        'signal_idx_ranges': signal_idx_ranges, 
        'response_metrics': response_metrics,
        'phot_pointer': phot_df[brain_region + ('phot_zF',)]}
    
    return brain_region_event_signal_info

def get_session_signal_info(session):
    signal_info = {}

    for brain_region in session.brain_regions:
        for event_type in config.all_event_types:
            if len(session.event_idxs_container.data.get(event_type, [])) == 0:
                continue
            
            brain_region_event_signal_info = get_brain_region_event_signal_info(session, event_type, brain_region)
            
            # You probably need this for old sessions
            #br, side, color = brain_region.split('_')
            try:
                br, side, color = brain_region.split('_')
            except AttributeError:
                br, side, color = brain_region
            signal_info[(br, color, event_type)] = brain_region_event_signal_info
    return signal_info

def assign_sessions_signal_info(sessions):
    for session in sessions:
        session.signal_info = get_session_signal_info(session)