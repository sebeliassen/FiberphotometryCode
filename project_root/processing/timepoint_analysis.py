import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import *

from scipy.signal import savgol_filter
import plotly.graph_objects as go

interval_start = peak_interval_config["interval_start"]
interval_end = peak_interval_config["interval_end"]


def get_signal_around_timepoint(session, timepoint_name, brain_region):
    if brain_region not in session.fiber_to_region.values():
        raise ValueError(f"This brain region is not available. \
                         Available ones are {list(session.fiber_to_region.values())}")
    
    # Ensure timepoints are fetched from the timepoints_container
    timepoints = session.timepoints_container.get_data(timepoint_name)
    if timepoints is None or len(timepoints) == 0:
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

    return time_axis[:all_data.shape[1]], all_data#, all_data.mean(axis=0)


def aggregate_signals(sessions, timepoint_name, regions_to_aggregate, ax, 
                      aggregate_by_session=True, normalize_baseline=False, 
                      color='C0', subtitle=''):
    aggregated_signals = []
    time_axis = np.arange(-interval_start, interval_end)  # Ensure interval_start and interval_end are defined

    for session in sessions:
        if len(session.timepoints_container.data.get(timepoint_name, [])) == 0:
            continue
        
        for region in regions_to_aggregate:
            if region in session.brain_regions:
                _, signal_data = get_signal_around_timepoint(session, timepoint_name, region)
                if aggregate_by_session:
                    signal_data = signal_data.mean(axis=0)
                aggregated_signals.append(signal_data)

    # Convert list of signals into a NumPy array for vectorized operations
    if aggregate_by_session:
        aggregated_signals = np.array(aggregated_signals)
    else:
        aggregated_signals = np.vstack(aggregated_signals)

    if normalize_baseline:
        aggregated_signals -= np.mean(aggregated_signals[:, :150], axis=1, keepdims=True)

    if aggregated_signals.size > 0:
        # Calculate mean and standard error of the mean
        mean_signal = np.mean(aggregated_signals, axis=0)
        sem_signal = stats.sem(aggregated_signals, axis=0)
        # Calculate the upper and lower bounds of the 95% confidence interval
        ci_95 = sem_signal * stats.t.ppf((1 + 0.95) / 2., len(aggregated_signals)-1)
        lower_bound = mean_signal - ci_95
        upper_bound = mean_signal + ci_95

        # Plot directly to the provided ax object
        ax.plot(time_axis / 20, mean_signal, label='Mean Signal', color=color)
        ax.fill_between(time_axis / 20, lower_bound, upper_bound, color=color, alpha=0.2, label='95% CI')    
        ax.set_title(subtitle)
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Z-score")
        ax.legend()
        return lower_bound.min(), upper_bound.max()
    else:
        print(f"No sessions contained timepoints for '{timepoint_name}', so no plot was generated.")


def add_aucs(sessions):
    for session in sessions:
        brain_regs = session.fiber_to_region.values()
        # right now just look at the brain reg that is lowest in the alphabet
        brain_reg = sorted(brain_regs)[0]

        phot_times = session.df_container.data['photwrit_470']['SecFromZero_FP3002'].values
        signal = session.df_container.data['photwrit_470'][brain_reg]
        raw_df = session.df_container.data['raw']

        # Initialize a dictionary to store the AUC results for each attribute
        auc_results = {}

        for attr in actions_attr_dict.values():
            # Get before and after times
            before_times = raw_df.loc[session.timepoints_container.data[f'before_dispimg_{attr}']]['SecFromZero_FP3002'].values
            after_times = raw_df.loc[session.timepoints_container.data[f'after_dispimg_{attr}']]['SecFromZero_FP3002'].values


            # Find the index of the next closest time in phot_times
            before_indices = np.searchsorted(phot_times, before_times, side='right')
            after_indices = np.searchsorted(phot_times, after_times, side='right')

            # Calculate the AUC for each pair of before and after indices
            auc_list = []
            for start_idx, end_idx in zip(before_indices, after_indices):
                # Extract the signal and time slices between the start and end indices
                signal_slice = signal[start_idx:end_idx]
                time_slice = phot_times[start_idx:end_idx]

                # Calculate the AUC using the trapezoidal rule
                auc_value = np.trapz(signal_slice, x=time_slice)
                
                # Append a tuple with start index, end index, and auc value to the list
                auc_list.append((start_idx, end_idx, auc_value))

            # Store the list of tuples in the dictionary with attr as the key
            auc_results[attr] = auc_list
        session.auc_results = auc_results


def plot_session_events_and_signal(session, brain_reg, fig, row, col, title_suffix=""):
    phot_df = session.df_container.data['photwrit_470']
    raw_df = session.df_container.data['raw']
    filtered_df = raw_df.loc[session.events_of_interest_df["index"]]

    signal = phot_df[brain_reg]
    signal = savgol_filter(signal, 10, 3)  # Apply Savitzky-Golay filter
    phot_times = phot_df['SecFromZero_FP3002'].values
    event_times = filtered_df['SecFromZero_FP3002'].values
    event_names = filtered_df['Item_Name'].values

    event_color_map = {
        'Display Image': 'purple',
        'Missed Hit': 'red',
        'Correct Rejection': 'blue',
        'Hit': 'green',
        'Mistake': 'orange',
        'Correction Trial Correct Rejection': 'cyan',
    }

    # Plot the photometry signal
    fig.add_trace(go.Scatter(x=phot_times, y=signal, mode='lines', name='Signal', line=dict(color='black')), row=row, col=col)

    # Scatter plot for each event
    for time, name in zip(event_times, event_names):
        color = event_color_map.get(name, 'gray')
        fig.add_trace(go.Scatter(x=[time], y=[signal[np.searchsorted(phot_times, time, side='left')]], mode='markers',
                                 marker=dict(color=color), name=name), row=row, col=col)

    # Calculate limits for x-axis based on the event times
    first_event_time = min(event_times)
    last_event_time = max(event_times)
    # Optionally, you can expand the range slightly to ensure all points are comfortably within the view
    x_range = [first_event_time - (last_event_time - first_event_time) * 0.05, 
            last_event_time + (last_event_time - first_event_time) * 0.05]

    # Find indices for the range of event times to calculate y-axis limits
    start_index = np.searchsorted(phot_times, first_event_time, side='left')
    end_index = np.searchsorted(phot_times, last_event_time, side='right')

    # Calculate limits for y-axis based on the signal within the event time range
    y_min = min(signal[start_index:end_index])
    y_max = max(signal[start_index:end_index])
    # Expanding the y-axis range slightly for visual comfort
    y_range = [y_min - (y_max - y_min) * 0.05, 
            y_max + (y_max - y_min) * 0.05]

    # Updating layout for each subplot individually if needed
    fig.update_xaxes(title_text='Time (s)', row=row, col=col, range=x_range)
    fig.update_yaxes(title_text='Signal', row=row, col=col, range=y_range)