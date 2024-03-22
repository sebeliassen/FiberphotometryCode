import numpy as np
from collections import defaultdict
from itertools import chain

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
    '''weight_method: 'mice', 'mice_events', 'events' '''
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
        
    
    if n and weight_method == 'events':
        sample_idxs = np.random.choice(all_signals.shape[0], size=n, replace=False)
        all_signals = all_signals[sample_idxs]
        all_resp_metrics = all_signals[sample_idxs]
    
    return all_signals, all_resp_metrics, curr_signal_info

