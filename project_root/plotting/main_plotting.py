import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from analysis.timepoint_analysis import aggregate_signals


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

    added_legend = set()  # Keep track of which event types have been added to the legend
    for time, name in zip(event_times, event_names):
        color = event_color_map.get(name, 'gray')
        # If we've already added this event type to the legend, don't add it again
        showlegend = name not in added_legend
        fig.add_trace(go.Scatter(
            x=[time], y=[signal[np.searchsorted(phot_times, time, side='left')]],
            mode='markers',
            marker=dict(color=color),
            name=name,
            showlegend=showlegend
        ), row=row, col=col)
        # Remember that we've added this event type
        added_legend.add(name)


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

def adjust_and_plot(ax, data, title, ylim, color='blue', label='Mean Signal', shading_boundaries=None):
    """Adjusts the y-limits based on provided bounds, plots the data, and shades areas outside specified boundaries."""
    time_axis, mean_signal, lower_bound, upper_bound = data
    ax.plot(time_axis, mean_signal, label=label, color=color)
    ax.fill_between(time_axis, lower_bound, upper_bound, color=color, alpha=0.2, label='95% CI')
    ax.set_title(title)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Z-score")
    ax.legend()
    ax.set_ylim(ylim)
    ax.set_xlim(time_axis.min(), time_axis.max())
    ax.grid()
    
    # Shade areas outside the specified boundaries, if provided
    if shading_boundaries is not None:
        ax.axvspan(time_axis.min(), shading_boundaries[0], color='grey', alpha=0.2)
        ax.axvspan(shading_boundaries[1], time_axis.max(), color='grey', alpha=0.2)

def main_plotting_function(outs, subtitles, suptitle, color, shading_boundaries=None, fname=None):
    """Plots aggregated signal data from multiple session groups with optional shading outside specified boundaries.
    
    Parameters:
    - outs: A list of tuples, each containing (time_axis, mean_signal, lower_bound, upper_bound) for a session group.
    - suptitle: Super title for the plot.
    - colors: List of colors for each session group plot.
    - shading_boundaries: Tuple indicating the start and end of the region not to be shaded.
    """
    _, axs = plt.subplots(figsize=(10, 5), ncols=len(outs), dpi=300)
    if len(outs) == 1:  # Ensure axs is iterable for a single subplot
        axs = [axs]
    
    # Determine global y-limits
    _, _, all_lbs, all_ubs = zip(*outs)
    global_lb = min(lb.min() for lb in all_lbs)
    global_ub = max(ub.max() for ub in all_ubs)
    if (global_ub - global_lb) < 2:
        global_ylim = ((global_lb + global_ub) / 2 - 1, (global_lb + global_ub) / 2 + 1)
    else:
        global_ylim = (global_lb, global_ub)

    # Plotting with adjusted y-limits and optional shading
    for data, ax, subtitle in zip(outs, axs, subtitles):
        adjust_and_plot(ax, data, title=subtitle, ylim=global_ylim, color=color, shading_boundaries=shading_boundaries)

    plt.suptitle(suptitle)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()