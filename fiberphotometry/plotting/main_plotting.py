import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import config


def plot_session_events_and_signal(session, brain_reg, fig, row, col, title_suffix="", smooth=False):
    phot_df = session.df_container.data['photwrit_470']
    raw_df = session.df_container.data['raw']
    filtered_df = raw_df.loc[session.events_of_interest_df["index"]]

    signal = phot_df[f'{brain_reg}_phot_zF']
    phot_times = phot_df['SecFromZero_FP3002'].values
    event_times = filtered_df['SecFromZero_FP3002'].values
    event_names = filtered_df['Item_Name'].values
    blank_image_time = raw_df.iloc[session.cpt]['SecFromZero_FP3002']

    event_color_map = {
        'Display Image': 'purple',
        'Missed Hit': 'red',
        'Correct Rejection': 'blue',
        'Hit': 'green',
        'Mistake': 'orange',
        'Correction Trial Correct Rejection': 'cyan',
    }
    

    #signal = np.convolve(signal, window, mode='same')
    # Plot the photometry signal
    fig.add_trace(go.Scatter(x=phot_times, y=signal, mode='lines', line=dict(color='black'), showlegend=False), 
                  row=row, col=col)

    event_traces = {}
    for time, name in zip(event_times, event_names):
        color = event_color_map.get(name, 'gray')
        if name not in event_traces:
            event_traces[name] = {
                'x': [],
                'y': [],
                'color': color,
            }
        event_traces[name]['x'].append(time)
        event_traces[name]['y'].append(signal[np.searchsorted(phot_times, time, side='left')])

    for name, data in event_traces.items():
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'],
            mode='markers',
            marker=dict(color=data['color']),
            name=name,
            showlegend=True
        ), row=row, col=col)

    # Add a dark grey dashed vertical line at blank image time
    fig.add_vline(
        x=blank_image_time, 
        line=dict(color='darkgrey', dash='dash'), 
        annotation_text='CPT start', 
        annotation_position='top right', 
        row=row, col=col
    )

    # Calculate limits for x-axis based on the event times
    first_event_time = min(event_times)
    last_event_time = max(event_times)
    x_range = [first_event_time - (last_event_time - first_event_time) * 0.05, 
               last_event_time + (last_event_time - first_event_time) * 0.05]

    start_index = np.searchsorted(phot_times, first_event_time, side='left')
    end_index = np.searchsorted(phot_times, last_event_time, side='right')

    y_min = min(signal[start_index:end_index])
    y_max = max(signal[start_index:end_index])
    y_range = [y_min - (y_max - y_min) * 0.05, 
               y_max + (y_max - y_min) * 0.05]

    fig.update_layout(height=400, width=1000, title_text=title_suffix)
    fig.update_xaxes(title_text='Time (s)', row=row, col=col)
    fig.update_yaxes(title_text='zF - score', row=row, col=col)

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


def plot_signals(all_signals, subtitles, suptitle, color, smoothing_len, shading_boundaries, scatters=None, plot_all=False, fname=None):
    xs, all_ys, all_lbs, all_ubs, global_ylim = preprocess_signals(all_signals, smoothing_len)
    _, axs = plt.subplots(figsize=(10, 5), ncols=len(all_signals), nrows=1, dpi=300)

    for ax_main, ys, lb, ub, subtitle in zip(axs, all_ys, all_lbs, all_ubs, subtitles):
        adjust_and_plot(ax_main, xs, ys, lb, ub, title=subtitle, ylim=global_ylim, color=color, shading_boundaries=shading_boundaries)

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