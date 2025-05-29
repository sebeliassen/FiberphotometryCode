from collections import defaultdict
import warnings
from fiberphotometry.analysis.response_metrics import calculate_signal_response_metrics_matrix
from fiberphotometry.utils import find_start_end_idxs
from fiberphotometry import config
import numpy as np
from fiberphotometry.config import SYNC



def get_brain_region_event_signal_info(session, event_type, brain_region):
    # --- config sanity ---------------------------------------------
    try:
        interval_start = config.peak_interval_config["interval_start"]
        interval_end   = config.peak_interval_config["interval_end"]
    except KeyError as e:
        raise KeyError(f"peak_interval_config missing key {e.args[0]!r}")

    if any(v < 0 for v in (interval_start, interval_end)):
        raise ValueError(f"interval_start/end must be ≥ 0 (got {interval_start}, {interval_end})")

    try:
        fps = config.PLOTTING_CONFIG[session.session_type]["fps"]
    except KeyError:
        raise KeyError(f"PLOTTING_CONFIG has no fps for session_type={session.session_type!r}")

    if fps <= 0:
        raise ValueError(f"fps must be positive (got {fps})")

    response_interval = find_start_end_idxs(event_type, fps)

    if not hasattr(session, 'response_metrics'):
        session.response_metrics = defaultdict(list)            

    # --- session structure -----------------------------------------
    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"{brain_region!r} not listed in session.fiber_to_region.values()")

    if not hasattr(session, "event_idxs_container"):
        raise AttributeError("session has no 'event_idxs_container' attribute")

    if not hasattr(session, "dfs"):
        raise AttributeError("session has no 'dfs' attribute holding DataFrames")
    
    event_type_idxs = session.event_idxs_container.get_data(event_type)
    if not event_type_idxs:
        raise ValueError(f"No event indices for {event_type}")

    # --- DataFrames and required columns ---------------------------
    raw_df = session.dfs.get_data("raw")
    if raw_df is None:
        raise ValueError("raw DataFrame is missing (dfs.get_data('raw') returned None)")

    freq   = config.LETTER_TO_FREQS.get(brain_region[-1])
    phot_df = session.dfs.get_data(f"phot_{freq}")
    if phot_df is None:
        raise ValueError(f"photometry DataFrame 'phot_{freq}' is missing")

    trial_time_col = SYNC["sec_trial_col"]
    for df_name, df in {"raw": raw_df, "phot": phot_df}.items():
        if trial_time_col not in df.columns:
            raise KeyError(f"Column {trial_time_col!r} absent from {df_name} DataFrame")

    if brain_region not in phot_df.columns:
        raise KeyError(f"Column {brain_region!r} absent from photometry DataFrame")

    phot_times = phot_df[trial_time_col].values
    if not np.all(np.diff(phot_times) >= 0):
        raise ValueError("phot_times are not monotonically increasing; cannot use searchsorted()")

    # prepare a matrix of (n_events × window_length)
    n_events       = len(event_type_idxs)
    window_len     = interval_start + interval_end
    signal_matrix  = np.zeros((n_events, window_len), dtype=float)
    signal_idx_ranges = []

    for i, raw_idx in enumerate(event_type_idxs):
        # 1) get the actual event time from the RAW stream
        event_time = raw_df[trial_time_col].iloc[raw_idx] # raw_df["SecFromTrialStart"]

        # 2) find where that falls in the photometry timeline
        phot_idx = int(np.searchsorted(phot_times, event_time, side='right'))

        # 3) skip if there’s not enough pre- or post-data
        if phot_idx < interval_start or phot_idx + interval_end > len(phot_times):
            # maybe this should just be a warning?
            warnings.warn(
                f"Skipping event {event_type!r} at raw index {raw_idx}: "
                f"phot_idx={phot_idx} in window [{-interval_start},{interval_end}] "
                "falls outside photometry length"
            )
            continue

        start_idx = phot_idx - interval_start
        end_idx   = phot_idx + interval_end
        signal_idx_ranges.append((start_idx, end_idx))

        # 4) pull out the data and normalize around the event
        # raw_trace = phot_df[brain_region + ('phot_zF',)].iloc[start_idx:end_idx].values.copy()
        raw_trace = phot_df[brain_region].iloc[start_idx:end_idx].values.copy()

        fit_start, fit_end = response_interval
        # center around the mean in a small window around the event
        pre_mean = raw_trace[fit_start-7:fit_start+7].mean()
        # pre_mean = 0
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
            br, side, color = (br_region.split("_") if isinstance(br_region, str)
                            else br_region)
        except ValueError:
            raise ValueError(
                f"brain_region {br_region!r} must split/iter to exactly 3 parts "
                "(region, side, color)"
            )

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