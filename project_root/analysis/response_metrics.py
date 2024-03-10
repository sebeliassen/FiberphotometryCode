import numpy as np
from tqdm import tqdm
from processing.timepoint_analysis import aggregate_signals


def find_interval_for_auc(xs, ys):
    # Fit the polynomial to the data
    # (time_axis / 20)[100:-75], mean_signal[100:-75]
    coefficients = np.polyfit(xs, ys, 8)
    p = np.poly1d(coefficients)

    # Find the first derivative and its roots for extrema
    p_derivative = np.polyder(p)
    extrema_roots = np.sort(p_derivative.roots[np.isreal(p_derivative.roots)].real)

    # Find c_x: the x-value of the extrema closest to zero
    if extrema_roots.size > 0:
        c_x = extrema_roots[np.argmin(np.abs(extrema_roots))]

        # Second derivative for inflection points
        p_double_derivative = np.polyder(p_derivative)
        inflection_roots = np.sort(p_double_derivative.roots[np.isreal(p_double_derivative.roots)].real)

        # Determine the extrema closest to c_x on both sides
        left_extrema = extrema_roots[extrema_roots < c_x]
        right_extrema = extrema_roots[extrema_roots > c_x]

        points_to_plot = []

        # Check left side for extrema or fallback to inflection point
        if left_extrema.size > 0:
            left_point = left_extrema[-1]  # Closest on the left
        else:
            left_inflections = inflection_roots[inflection_roots < c_x]
            left_point = left_inflections[-1] if left_inflections.size > 0 else None

        if left_point is not None:
            points_to_plot.append((left_point, p(left_point)))

        # Check right side for extrema or fallback to inflection point
        if right_extrema.size > 0:
            right_point = right_extrema[0]  # Closest on the right
        else:
            right_inflections = inflection_roots[inflection_roots > c_x]
            right_point = right_inflections[0] if right_inflections.size > 0 else None

        if right_point is not None:
            points_to_plot.append((right_point, p(right_point)))

        # Plot the determined points
            # for point in points_to_plot:
            #     ax.scatter(*point, color='red', zorder=5)
        return points_to_plot        


def calculate_signal_response_metrics(signal, interval):
    # Assuming 'interval' is a slice or range, adjust signal accordingly
    adjusted_signal = signal[interval[0]:interval[1]+1] if isinstance(interval, (list, tuple, range)) else signal

    # Calculate peak timing once, as it's used multiple times
    peak_idx = np.argmax(adjusted_signal)

    # Slopes calculation
    maxima = adjusted_signal[peak_idx]
    left_minima = adjusted_signal[0]
    right_minima = adjusted_signal[-1]

    slope_up = ((maxima - left_minima) / peak_idx if
                peak_idx != 0 else float('inf'))  # Avoid division by zero
    slope_down = ((maxima - right_minima) / (len(adjusted_signal) - peak_idx) if
                  peak_idx != len(adjusted_signal) else float('inf'))  # Adjust for index

    response_metrics = {
        'slope_up': slope_up,
        'slope_down': slope_down,
        'maximal_value': maxima,
        'peak_timing': peak_idx,
        'auc': np.trapz(adjusted_signal, dx=1)
    }

    return response_metrics


def is_relevant_session(brain_regions, event_type, session):
    return ((brain_regions[0] in session.brain_regions 
             or brain_regions[1] in session.brain_regions)
             and session.timepoints_container.get_data(event_type))


def assign_responses(mice, disable_tqdm=False):
    for mouse in tqdm(mice, disable=disable_tqdm):
        # Initialize a flat dictionary for each mouse
        flattened_responses = {}
        
        # Iterate over combinations of brain regions and event types
        for brain_regions in [['VS_left', 'VS_right'], ['DMS_left', 'DMS_right'], ['DLS_left', 'DLS_right']]:
            for event_type in ['hit', 'mistake', 'miss', 'cor_reject', 'reward_collect']:
                filtered_sessions = [session for session in mouse.sessions if is_relevant_session(brain_regions, event_type, session)]

                if filtered_sessions:
                    _, ys, _, _, interval = aggregate_signals(filtered_sessions, event_type, brain_regions, 
                                                            aggregate_by_session=False, normalize_baseline=True)
                    response_metrics = calculate_signal_response_metrics(ys, interval)

                    # Generate a unique key by combining brain region (prefix) and event type
                    unique_key_prefix = brain_regions[0].split('_')[0]  # Extract region prefix (e.g., "VS")
                    unique_key = f"{unique_key_prefix}_{event_type}"
                    
                    # Add suffixes to each metric and merge them into the flattened_responses dictionary
                    for metric_name, metric_value in response_metrics.items():
                        flattened_key = f"{unique_key}_{metric_name}"
                        flattened_responses[flattened_key] = metric_value
        
        # Assign the flattened dictionary to the current mouse
        mouse.response_metrics = flattened_responses