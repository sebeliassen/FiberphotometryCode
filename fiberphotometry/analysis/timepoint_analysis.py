import numpy as np
from collections import defaultdict
from itertools import chain
from fiberphotometry.utils import mouse_br_events_count

def collect_sessions_data(sessions, event_type, channel, regions_to_aggregate):
    if not isinstance(regions_to_aggregate, list):
        regions_to_aggregate = [regions_to_aggregate]
    all_signals = []
    all_resp_metrics = []
    mouse_ids = []

    for session in sessions:
        if len(session.event_idxs_container.data.get(event_type, [])) == 0:
            continue
        for brain_region in session.brain_regions:
            br_split = brain_region.split('_')
            if br_split[0] not in regions_to_aggregate:
                continue

            # Logic for legacy sessions i.e. sessions without explicit channels
            if len(br_split) == 3:
                brain_region, _, curr_channel = br_split
                if curr_channel != channel:
                    continue
                curr_signal_info = session.signal_info[(brain_region, channel, event_type)]
            elif len(br_split) == 2:
                brain_region, _ = br_split
                curr_signal_info = session.signal_info[(brain_region, event_type)]
            else:
                continue  # Skip invalid brain_region formats

            all_signals.append(curr_signal_info['signal_matrix'])
            all_resp_metrics.append(curr_signal_info['response_metrics'][0])
            mouse_ids.append(session.mouse_id)

    return all_signals, all_resp_metrics, mouse_ids
    

def sample_signals_and_metrics(sessions, brain_regions, channel, event_type, n=None, weight_method='events'):
    '''weight_method: 'mice', 'mice_events', 'events' \n
    returns all_signals, all_resp_metrics, curr_signal_info'''

    # res = collect_sessions_data(sessions, event_type, brain_regions)
    # if res:
    #     all_signals, all_resp_metrics, mouse_ids, curr_signal_info = res
    # else:
    #     return None
    all_signals, all_resp_metrics, mouse_ids = collect_sessions_data(sessions, event_type, channel, brain_regions)
    if len(all_signals) == 0:
        return None
    
    if weight_method == 'mice':
        all_signals_by_mouse = defaultdict(list)
        all_resp_metrics_by_mouse = defaultdict(list)
        for idx, mouse_id in enumerate(mouse_ids):
            all_signals_by_mouse[mouse_id].append(np.mean(all_signals[idx], axis=0))
            all_resp_metrics_by_mouse[mouse_id].append(np.mean(all_resp_metrics[idx], axis=0))
        
        all_signals = [np.mean(np.vstack(signals), axis=0) for signals in all_signals_by_mouse.values()]
        all_signals = np.vstack(all_signals)
        all_resp_metrics = [np.mean(np.vstack(signals), axis=0) for signals in all_resp_metrics_by_mouse.values()]
        all_resp_metrics = np.vstack(all_resp_metrics)

    elif weight_method == 'mice_events':
        all_signals_by_mouse = defaultdict(list)
        all_resp_metrics_by_mouse = defaultdict(list)

        for idx, mouse_id in enumerate(mouse_ids):
            all_signals_by_mouse[mouse_id].append(all_signals[idx])
            all_resp_metrics_by_mouse[mouse_id].append(all_resp_metrics[idx])

        all_signals = [np.vstack(signals) for signals in all_signals_by_mouse.values()]
        all_resp_metrics = [np.vstack(signals) for signals in all_resp_metrics_by_mouse.values()]

        if n is None:
            print(len([signals.shape[0] for signals in all_signals]))
            min_len = min([signals.shape[0] for signals in all_signals])
        else:
            min_len = n

        for i in range(len(all_signals)):
            sample_idxs = np.random.choice(all_signals[i].shape[0], size=min_len, replace=False)
            all_signals[i] = all_signals[i][sample_idxs]
            all_resp_metrics[i] = all_resp_metrics[i][sample_idxs]
        all_signals = np.vstack(all_signals)
        all_resp_metrics = np.vstack(all_resp_metrics)

    elif weight_method == 'events':    
        all_signals = np.vstack(all_signals)
        all_resp_metrics = np.vstack(all_resp_metrics)

    else:
        raise ValueError(f"Invalid weight method: {weight_method}")
    
    return all_signals, all_resp_metrics


def sample_low_and_high_signals(weight_method, performance_metric, brain_region, event, mouse_analyser, n=None):
    '''weight_method: 'mice', 'mice_events', 'events'\n
    returns low_signals, high_signals, low_resp_metrics, high_resp_metrics, resp_metric_names'''
    low_sessions, high_sessions = \
        mouse_analyser.sample_high_and_low_sessions(performance_metric, brain_region, event)
    
    if weight_method == 'mice_events':
        mouse_ids = {session.mouse_id for session in (low_sessions + high_sessions)}
        n = min(mouse_br_events_count(mouse_analyser.mice_dict[mouse_id], brain_region, event) for mouse_id in mouse_ids)

    low_signals, low_resp_metrics = sample_signals_and_metrics(low_sessions, event, brain_region, 
                                                                                  weight_method=weight_method, n=n)
    high_signals, high_resp_metrics = sample_signals_and_metrics(high_sessions, event, brain_region, 
                                                                    weight_method=weight_method, n=n)

    if weight_method == 'events' and n is not None:
        n = min(n, low_signals.shape[0], high_signals.shape[0])
        sample_idxs = np.random.choice(n, size=n, replace=False)

        low_signals = low_signals[sample_idxs]
        high_signals = high_signals[sample_idxs]

    return low_signals, high_signals, low_resp_metrics, high_resp_metrics


def get_injection_peak(signal, phot_times, blank_image_time):
    window = np.ones(100) / 100
    smoothed_signal = np.convolve(signal, window, mode='same')

    peak_idx_start = np.searchsorted(phot_times, blank_image_time - 10 * 60, side='left')
    peak_idx_end = np.searchsorted(phot_times, blank_image_time - 2.5 * 60, side='left')
    peak_x = np.argmax(smoothed_signal[peak_idx_start:peak_idx_end]) + peak_idx_start
    peak_y = smoothed_signal[peak_x]

    return peak_x, peak_y


def split_signal_by_injection_threshold(signal, phot_times, peak_y, blank_image_time):
    # Smooth heavily
    window = np.ones(1000) / 1000
    smoothed_signal = np.convolve(signal, window, mode='same')
    
    # Find the index corresponding to the injection time
    injection_idx = np.searchsorted(phot_times, blank_image_time, side='left')
    
    # Create a mask only for times after the injection
    signal_after_injection = smoothed_signal[injection_idx:]
    above_peak_mask = signal_after_injection > peak_y
    
    if np.sum(above_peak_mask) == 0:
        # No crossing found, return something meaning "no crossing"
        # E.g., return None, or the last index, or raise an error
        return len(signal) - 1
    else:
        # This is the first True index after injection
        first_cross_local_idx = np.argmax(above_peak_mask)
        
        # Convert back to an index in the full array
        threshold_x = injection_idx + first_cross_local_idx
        return threshold_x
    

def find_drug_split_x(session, brain_reg, find_relative_peak=False):
    phot_df = session.dfs.data['phot_470']
    raw_df = session.dfs.data['raw']
    
    # print(phot_df.columns)
    # ('mPFC', 'left', 'G', 'phot_zF')
    #print(brain_reg)
    #signal = phot_df[f'{brain_reg}_phot_zF']
    for signal_name, data in session.signal_meta.items():
        if data[0] == brain_reg:
            break
    signal = phot_df[signal_name + '_dff']
    phot_times = phot_df['sec_from_zero'].values
    blank_image_time = raw_df.iloc[session.cpt]['sec_from_zero']

    # 1) Find the injection peak
    peak_x, peak_y = get_injection_peak(signal, phot_times, blank_image_time)
    
    # 2) Determine threshold crossing, but only after injection
    threshold_x = split_signal_by_injection_threshold(
        signal=signal, 
        phot_times=phot_times, 
        peak_y=peak_y, 
        blank_image_time=blank_image_time
    )
    
    if find_relative_peak:
        return (phot_times[peak_x] - blank_image_time) / 60, peak_y
    else:
        return threshold_x