import numpy as np
from collections import defaultdict
from itertools import chain
from utils import mouse_br_events_count

def collect_sessions_data(sessions, event_type, regions_to_aggregate):
    if not isinstance(regions_to_aggregate, list):
        regions_to_aggregate = [regions_to_aggregate]
    all_signals = []
    all_resp_metrics = []
    mouse_ids = []

    for session in sessions:
        if len(session.event_idxs_container.data.get(event_type, [])) == 0:
            continue
        for brain_region in session.brain_regions:
            brain_region = brain_region.split('_')[0]
            if brain_region not in regions_to_aggregate:
                continue
            curr_signal_info = session.signal_info[(brain_region, event_type)]
            all_signals.append(curr_signal_info['signal_matrix'])
            all_resp_metrics.append(curr_signal_info['response_metrics'][0])
            mouse_ids.append(session.mouse_id)

    return all_signals, all_resp_metrics, mouse_ids, curr_signal_info['response_metrics'][1]

def sample_signals_and_metrics(sessions, event_type, brain_regions, n=None, weight_method='events'):
    '''weight_method: 'mice', 'mice_events', 'events' \n
    returns all_signals, all_resp_metrics, curr_signal_info'''
    if weight_method == 'mice_events' and n is None:
        raise ValueError("n must be set when using weight_method='mice_events'")

    all_signals, all_resp_metrics, mouse_ids, curr_signal_info = collect_sessions_data(sessions, event_type, brain_regions)
    
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
    
    return all_signals, all_resp_metrics, curr_signal_info


def sample_low_and_high_signals(weight_method, performance_metric, brain_region, event, mouse_analyser, n=None):
    '''weight_method: 'mice', 'mice_events', 'events'\n
    returns low_signals, high_signals, low_resp_metrics, high_resp_metrics, resp_metric_names'''
    low_sessions, high_sessions = \
        mouse_analyser.sample_high_and_low_sessions(performance_metric, brain_region, event)
    
    if weight_method == 'mice_events':
        mouse_ids = {session.mouse_id for session in (low_sessions + high_sessions)}
        n = min(mouse_br_events_count(mouse_analyser.mice_dict[mouse_id], brain_region, event) for mouse_id in mouse_ids)

    low_signals, low_resp_metrics, resp_metric_names = sample_signals_and_metrics(low_sessions, event, brain_region, 
                                                                                  weight_method=weight_method, n=n)
    high_signals, high_resp_metrics, _ = sample_signals_and_metrics(high_sessions, event, brain_region, 
                                                                    weight_method=weight_method, n=n)

    if weight_method == 'events' and n is not None:
        n = min(n, low_signals.shape[0], high_signals.shape[0])
        sample_idxs = np.random.choice(n, size=n, replace=False)

        low_signals = low_signals[sample_idxs]
        high_signals = high_signals[sample_idxs]

    return low_signals, high_signals, low_resp_metrics, high_resp_metrics, resp_metric_names