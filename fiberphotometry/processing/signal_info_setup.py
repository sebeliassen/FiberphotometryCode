from collections import defaultdict
from fiberphotometry.analysis.response_metrics import calculate_signal_response_metrics_matrix
from fiberphotometry.utils import find_start_end_idxs
from fiberphotometry import config
import numpy as np
from fiberphotometry.config import SYNC

interval_start = config.peak_interval_config["interval_start"]
interval_end = config.peak_interval_config["interval_end"]

def get_brain_region_event_signal_info(session, event_type, brain_region):
    fps = config.PLOTTING_CONFIG[session.session_type]['fps']
    response_interval = find_start_end_idxs(event_type, fps)

    if not hasattr(session, 'response_metrics'):
        session.response_metrics = defaultdict(list)            

    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"{brain_region} not in {session.fiber_to_region.values()}")

    # raw + photometry DFs
    raw_df  = session.dfs.get_data("raw")
    phot_df = session.dfs.get_data(f"phot_{config.LETTER_TO_FREQS[brain_region[-1]]}")

    event_type_idxs = session.event_idxs_container.get_data(event_type)
    if not event_type_idxs:
        raise ValueError(f"No event indices for {event_type}")

    # prepare a matrix of (n_events × window_length)
    n_events       = len(event_type_idxs)
    interval_start = config.peak_interval_config['interval_start']
    interval_end   = config.peak_interval_config['interval_end']
    window_len     = interval_start + interval_end
    signal_matrix  = np.zeros((n_events, window_len), dtype=float)
    signal_idx_ranges = []

    trial_time_col = SYNC['sec_trial_col'] # e.g., "SecFromTrialStart"
    phot_times = phot_df[trial_time_col].values

    for i, raw_idx in enumerate(event_type_idxs):
        # 1) get the actual event time from the RAW stream
        event_time = raw_df[trial_time_col].iloc[raw_idx] # raw_df["SecFromTrialStart"]

        # 2) find where that falls in the photometry timeline
        phot_idx = int(np.searchsorted(phot_times, event_time, side='right'))

        # 3) skip if there’s not enough pre- or post-data
        if phot_idx < interval_start or phot_idx + interval_end > len(phot_times):
            continue

        start_idx = phot_idx - interval_start
        end_idx   = phot_idx + interval_end
        signal_idx_ranges.append((start_idx, end_idx))

        # 4) pull out the data and normalize around the event
        # raw_trace = phot_df[brain_region + ('phot_zF',)].iloc[start_idx:end_idx].values.copy()
        raw_trace = phot_df[brain_region].iloc[start_idx:end_idx].values.copy()

        fit_start, fit_end = response_interval
        # center around the mean in a small window around the event
        # pre_mean = raw_trace[fit_start-7:fit_start+7].mean()
        pre_mean = 0
        signal_matrix[i] = raw_trace - pre_mean

    # 5) compute your metrics on the completed matrix
    response_metrics = calculate_signal_response_metrics_matrix(
        signal_matrix, response_interval, fps
    )

    return {
        'signal_matrix':     signal_matrix,
        'signal_idx_ranges': signal_idx_ranges,
        'response_metrics':  response_metrics,
        'phot_pointer':      phot_df[brain_region]
        # 'phot_pointer':      phot_df[brain_region + ('phot_zF',)]
    }

def get_session_signal_info(session):
    signal_info = {}
    import re
    # match keys ending in _<2letters><digits>, e.g. '_fr1' or '_cc3'
    suffix_re = re.compile(r'^(?P<base>.+)_(?P<suffix>[a-z]{2}\d+)$')

    for br_region in session.brain_regions:
        # unravel brain_region tuple or string
        try:
            br, side, color = br_region.split('_')
        except (AttributeError, ValueError):
            br, side, color = br_region

        for event_key, idxs in session.event_idxs_container.data.items():
            if not idxs:
                continue

            m = suffix_re.match(event_key)
            if m:
                base_key = m.group('base')

                # 1) full (suffixed) event
                info_full = get_brain_region_event_signal_info(session, event_key, br_region)
                signal_info[(br, color, event_key)] = info_full

                # 2) base event, if it actually has any indices
                if base_key in session.event_idxs_container.data and session.event_idxs_container.data[base_key]:
                    info_base = get_brain_region_event_signal_info(session, base_key, br_region)
                    signal_info[(br, color, base_key)] = info_base

            else:
                # unsuffixed: just include directly
                info = get_brain_region_event_signal_info(session, event_key, br_region)
                signal_info[(br, color, event_key)] = info

    return signal_info

def assign_sessions_signal_info(sessions):
    for session in sessions:
        session.signal_info = get_session_signal_info(session)