import numpy as np

def sample_signals_and_metrics(sessions, event_type, regions_to_aggregate, n=None):
    regions_to_aggregate = [regions_to_aggregate]
    all_signals = []
    all_resp_metrics = []

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

    all_signals = np.vstack(all_signals)
    all_resp_metrics = np.vstack(all_resp_metrics)
    
    if n:
        sample_idxs = np.random.choice(all_signals.shape[0], size=n, replace=False)
        all_signals = all_signals[sample_idxs]
        all_resp_metrics = all_signals[sample_idxs]
    
    return all_signals, all_resp_metrics, curr_signal_info['response_metrics'][1]
