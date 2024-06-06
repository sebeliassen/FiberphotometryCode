import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import config


def plot_session_events_and_signal(session, brain_reg, fig, row, col, title_suffix=""):
    phot_df = session.df_container.data['photwrit_470']
    raw_df = session.df_container.data['raw']
    filtered_df = raw_df.loc[session.events_of_interest_df["index"]]

    signal = phot_df[f'{brain_reg}_phot_zF']
    # signal = savgol_filter(signal, 10, 3)  # Apply Savitzky-Golay filter
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
    fig.add_trace(go.Scatter(x=phot_times, y=signal, mode='lines', line=dict(color='black'), showlegend=False), row=row, col=col)

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
            showlegend=False
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
    fig.update_layout(height=400, width=400, title_text=title_suffix)
    fig.update_xaxes(title_text='Time (s)', row=row, col=col, range=x_range)
    fig.update_yaxes(title_text='zF - score', row=row, col=col, range=y_range)

def adjust_and_plot(ax, xs, ys, lb, ub, title, ylim, color='blue', label='Mean Signal', shading_boundaries=None, line_style='-'):
    """Adjusts the y-limits based on provided bounds, plots the data with specified line style, shades areas outside specified boundaries, and optionally adds vertical lines on the x-axis."""
    ax.plot(xs, ys, label=label, color=color, linestyle=line_style)
    ax.fill_between(xs, lb, ub, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Z-score")
    # ax.legend()
    ax.set_ylim(ylim)
    ax.set_xlim(xs.min(), xs.max())
    ax.grid()
    
    if shading_boundaries is not None:
        ax.axvspan(xs.min(), shading_boundaries[0], color='grey', alpha=0.2)
        ax.axvspan(shading_boundaries[1], xs.max(), color='grey', alpha=0.2)

def preprocess_signals(all_signals, smoothing_len):
    interval_start = config.peak_interval_config["interval_start"]
    interval_end = config.peak_interval_config["interval_end"]
    fps = config.PLOTTING_CONFIG['fps']
    
    xs = np.arange(-interval_start, interval_end) / fps
    
    all_ys = []
    all_lbs = []
    all_ubs = []
    for signals in all_signals:
        # Smooth the mean signal
        ys = np.mean(signals, axis=0)
        window = np.ones(smoothing_len) / smoothing_len
        ys = np.convolve(ys, window, 'same')

        # Calculate the standard deviation of the mean
        std_signal = np.std(signals, axis=0) / np.sqrt(len(signals))

        # Use scipy.stats.norm.interval to get the 95% confidence interval
        alpha = 0.95
        ci_lower, ci_upper = stats.norm.interval(alpha, loc=ys, scale=std_signal)

        # The lower and upper bounds
        lb = ci_lower
        ub = ci_upper
        all_ys.append(ys)
        all_lbs.append(lb)
        all_ubs.append(ub)

    global_lb = min([lb.min() for lb in all_lbs])
    global_ub = max([ub.max() for ub in all_ubs])
    if (global_ub - global_lb) < 2:
        global_ylim = ((global_lb + global_ub) / 2 - 1, (global_lb + global_ub) / 2 + 1)
    else:
        global_ylim = (global_lb, global_ub)
    
    return xs, all_ys, all_lbs, all_ubs, global_ylim

def plot_signals(all_signals, subtitles, suptitle, color, smoothing_len, shading_boundaries, scatters=None, plot_all=False, fname=None):
    xs, all_ys, all_lbs, all_ubs, global_ylim = preprocess_signals(all_signals, smoothing_len)
    _, axs = plt.subplots(figsize=(10, 10), ncols=len(all_signals), nrows=2, dpi=300)

    scatter_min_y = min(scatter[1].min() for scatter in scatters)
    scatter_max_y = min(scatter[1].max() for scatter in scatters)

    for ax_main, ax_scatter, ys, lb, ub, subtitle, scatter_data in zip(axs[0], axs[1], all_ys, all_lbs, all_ubs, subtitles, scatters):
        adjust_and_plot(ax_main, xs, ys, lb, ub, title=subtitle, ylim=global_ylim, color=color, shading_boundaries=shading_boundaries)

        # If scatter data is provided, plot it
        # if plot_all:
        #     for signal in all:
        #         smoothed_signal = np.convolve(signal, np.ones(5)/5, mode='same')
        #         ax_scatter.plot(xs, smoothed_signal)
        #     ax_scatter.grid()

        #     ymin, ymax = min(signals.min() for signals in all_signals), max(signals.max() for signals in all_signals)
        #     ymin = min(ymin, -1)  # Ensure ymin is at least -1
        #     ymax = max(ymax, 1)   # Ensure ymax is at least 1
        #     ax_scatter.set_ylim(ymin, ymax)  # Set the adjusted y-axis limits

        # else:
        if scatter_data is not None:
            ax_scatter.plot(xs, ys, color=color)
            ax_scatter.scatter(scatter_data[0], scatter_data[1], color=color, s=10, alpha=0.6)
            ax_scatter.set_xlim(xs.min(), xs.max())
            ax_scatter.set_ylim(min(scatter_min_y, global_ylim[0]), max(scatter_max_y, global_ylim[1]) + 0.5)  # Set y-limits for scatter plot if needed
            # ax_scatter.axis('off')  # Hide the axis for the scatter plot
            ax_scatter.grid()

    plt.suptitle(suptitle)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()


def plot_signals_p_values(all_signals, subtitles, suptitle, color, smoothing_len, shading_boundaries, p_values, fname=None):
    xs, all_ys, all_lbs, all_ubs, global_ylim = preprocess_signals(all_signals, smoothing_len)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)  # Changed to use a single primary axis
    
    labels = ['lower mice', 'upper mice']
    line_styles = ['--', '-']  # Solid for the first, dashed for the second

    # Superimposed signal plots using adjust_and_plot
    for ys, lb, ub, subtitle, label, line_style in zip(all_ys, all_lbs, all_ubs, subtitles, labels, line_styles):
        adjust_and_plot(ax1, xs, ys, lb, ub, title=subtitle, ylim=global_ylim, color=color, shading_boundaries=shading_boundaries, label=label, line_style=line_style)
    
    
    # Creating a secondary y-axis for p-values
    ax2 = ax1.twinx()
    
    # Plotting p-values on the secondary y-axis
    p_values = np.convolve(p_values, np.ones(10)/10, mode='same')
    # ax2.plot(xs, p_values_smoothed, label='P-Value (Smoothed)', color='black', alpha=0.8)
    min_x, max_x = shading_boundaries
    filtered_xs = [x for x in xs if min_x <= x <= max_x]
    filtered_p_values = [p for x, p in zip(xs, p_values) if min_x <= x <= max_x]

    # ax2.plot(xs, p_values, label='P-Value', color='black', alpha=0.8)
    ax2.plot(filtered_xs, filtered_p_values, label='P-Value', color='black', alpha=0.68)
    ax2.axhline(y=0.05, color='gold', linestyle='--', label='P<0.05')
    ax2.axhline(y=0.01, color='orange', linestyle='--', label='P<0.01')
    ax2.axhline(y=0.001, color='red', linestyle='--', label='P<0.001')

    # Get handles and labels from ax1
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    # Create upper right legend with ax1's handles and labels
    legend1 = ax2.legend(handles=handles1, labels=labels1, loc='upper right')
    ax2.add_artist(legend1)  # Add the first legend back (it will handle ax1's legends)

    # Get handles and labels from ax2 and create another legend for it
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles2, labels=labels2, loc='lower right')

    ax2.set_yscale('log')  # Setting logarithmic scale
    ax2.set_ylim(1, np.min(p_values))  # Adjust this range as needed
    ax2.set_ylabel('P-Value (log scale)')
    ax2.invert_yaxis()  # Invert y-axis to have 0.05 at the top and smaller values at the bottom
    # ax1.legend(loc='upper right')
    # ax2.legend(loc='lower right')

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if fname:
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()