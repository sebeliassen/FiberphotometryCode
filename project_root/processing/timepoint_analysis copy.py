import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import peak_interval_config

interval_start = peak_interval_config["interval_start"]
interval_end = peak_interval_config["interval_end"]


def get_signal_around_timepoint(session, timepoint_name, brain_region):
    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"This brain region is not available. \
                         Available ones are {list(session.fiber_to_region.values())}")
    
    # Ensure timepoints are fetched from the timepoints_container
    timepoints = session.timepoints_container.get_data(timepoint_name)
    if timepoints is None:
        raise ValueError(f"No timepoints found for {timepoint_name}")

    raw_df = session.df_container.get_data("raw")
    # TODO: currently hardcoded to photwrit_470, which it shouldn't
    phot_df = session.df_container.get_data("photwrit_470")
    
    all_data = np.zeros((len(timepoints), interval_start + interval_end))
    
    for i, event_idx in enumerate(timepoints):
        #event_time = raw_df['SecFromZero_FP3002'].iloc[event_idx]
        event_time = raw_df['SecFromZero_FP3002'].loc[event_idx]
        event_sync = phot_df['SecFromZero_FP3002'] - event_time
        
        if len(event_sync[event_sync > 0]) <= interval_end:
            continue  # Skip events too close to the end of a session
        
        event_index = event_sync[event_sync > 0].idxmin()
        
        # Ensuring the interval does not go out of bounds
        start_idx = max(event_index - interval_start, 0)
        end_idx = min(event_index + interval_end, len(phot_df))
        
        signal = phot_df[f'{brain_region}_phot_zF'].iloc[start_idx:end_idx]
        all_data[i, :len(signal)] = signal.values  # Handle cases where signal length < interval

    time_axis = np.arange(-interval_start, interval_end)

    return time_axis[:all_data.shape[1]], all_data.mean(axis=0)


def aggregate_signals(sessions, timepoint_name, regions_to_aggregate, plot=True, brain_reg_name=None):
    aggregated_signals = []
    time_axis = np.arange(-interval_start, interval_end)

    for session in sessions:
        if timepoint_name not in session.timepoints_container.fetch_all_data_names():
            continue
        
        for region in regions_to_aggregate:
            if region in session.fiber_to_region.values():
                _, signal_data = get_signal_around_timepoint(session, timepoint_name, region)
                aggregated_signals.append(signal_data)

    # Convert list of signals into a NumPy array for vectorized operations
    aggregated_signals = np.array(aggregated_signals)

    if aggregated_signals.size > 0:
        # Calculate mean and standard error of the mean
        mean_signal = np.mean(aggregated_signals, axis=0)
        sem_signal = stats.sem(aggregated_signals, axis=0)
        # Calculate the upper and lower bounds of the 95% confidence interval
        ci_95 = sem_signal * stats.t.ppf((1 + 0.95) / 2., len(aggregated_signals)-1)
        lower_bound = mean_signal - ci_95
        upper_bound = mean_signal + ci_95

        if plot:
            plt.figure()
            plt.plot(time_axis / 20, mean_signal, label='Mean Signal')
            plt.fill_between(time_axis / 20, lower_bound, upper_bound, color='b', alpha=0.2, label='95% CI')
            plt.title(f"Averaged Peak plot for {brain_reg_name} around {timepoint_name}")
            plt.xlabel("Time from event (s)")
            plt.ylabel("Z-score")
            plt.legend()
            plt.show()
    else:
        print(f"No sessions contained timepoints for '{timepoint_name}', so no plot was generated.")