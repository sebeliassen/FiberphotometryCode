# plotting.py
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'     # titles
plt.rcParams['axes.labelcolor'] = 'black'     # axis labels (optional)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import pandas as pd
import cv2
# Removed scipy.signal import (moved to photometry_processor)
import matplotlib.ticker as ticker
from typing import Tuple, Optional, List
import logging # Use logging

import config

# Import helpers from photometry_processor (or a dedicated utils module if created)
from photometry_processor import find_nearest_iloc, _estimate_target_behavior_timestamp

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Helper Functions find_nearest_iloc and _estimate_target_behavior_timestamp are REMOVED ---
# They are now in photometry_processor.py

# --- filter_photometry_segment function is REMOVED ---
# Its logic is now in photometry_processor.py


def generate_trace_plot_frame(
    filtered_segment_df: Optional[pd.DataFrame],
    plot_ylim: Optional[Tuple[float, float]],
    current_original_frame_index: int,
    left_region_name: str,
    right_region_name: str,
    frame_range_original_start: int,
    cam_ts_ns: Optional[np.ndarray] = None,     # <— ADD THIS
    log_prefix: str = ""
) -> np.ndarray:
    """
    Generates a single frame containing scrolling photometry traces (left and right regions).

    Args:
        filtered_segment_df: DataFrame containing the pre-filtered Z-score data
                             for the relevant segment, including 'BehaviorTimestamp'
                             and '{region}_filtered' columns.
        plot_ylim: Tuple (min, max) for the Y-axis limits.
        current_original_frame_index: The frame index (in original 60fps video)
                                      for which to generate the plot.
        left_region_name: Name of the left brain region (used for column lookup).
        right_region_name: Name of the right brain region.
        frame_range_original_start: The starting frame index of the entire segment
                                     (in original 60fps) used for timestamp estimation.
        log_prefix: String prefix for log messages.


    Returns:
        A NumPy array (BGR format) representing the plot image, or a blank white frame on error.
    """
    target_width = config.QUADRANT_WIDTH
    target_height = config.QUADRANT_HEIGHT
    blank_bgr_frame = np.full((target_height, target_width, 3), 255, dtype=np.uint8) # White background

    is_debug_frame = current_original_frame_index < frame_range_original_start + 10 # Log extra detail for first few frames
    if is_debug_frame:
         logger.debug(f"{log_prefix}--- Generating plot frame for Original Index: {current_original_frame_index} ---")
         logger.debug(f"{log_prefix}  Input plot_ylim: {plot_ylim}")

    # --- Input Validation ---
    if filtered_segment_df is None or filtered_segment_df.empty:
        if is_debug_frame: logger.warning(f"{log_prefix}  Input filtered_segment_df is None or empty. Returning blank frame.")
        return blank_bgr_frame
    if "BehaviorTimestamp" not in filtered_segment_df.columns:
        logger.error(f"{log_prefix}  'BehaviorTimestamp' column missing in filtered_segment_df. Returning blank frame.")
        return blank_bgr_frame
    if plot_ylim is None or not (isinstance(plot_ylim, tuple) and len(plot_ylim) == 2):
        logger.error(f"{log_prefix}  Invalid plot_ylim provided: {plot_ylim}. Returning blank frame.")
        return blank_bgr_frame
    if not np.isfinite(plot_ylim[0]) or not np.isfinite(plot_ylim[1]) or plot_ylim[0] >= plot_ylim[1]:
         logger.error(f"{log_prefix}  Non-finite or invalid plot_ylim range: {plot_ylim}. Returning blank frame.")
         return blank_bgr_frame

    behavior_timestamps_segment = filtered_segment_df["BehaviorTimestamp"].values
    if len(behavior_timestamps_segment) == 0:
        if is_debug_frame: logger.warning(f"{log_prefix}  BehaviorTimestamp array in segment is empty. Returning blank frame.")
        return blank_bgr_frame

    # --- Find Timestamp and Index for Current Frame ---
    # Estimate the 'BehaviorTimestamp' corresponding to the current original frame index
    if cam_ts_ns is not None and current_original_frame_index < len(cam_ts_ns):
        target_timestamp = cam_ts_ns[current_original_frame_index]   # ns
    else:
        target_timestamp = _estimate_target_behavior_timestamp(
            behavior_timestamps_segment,
            current_original_frame_index,
            frame_range_original_start
        )
    if target_timestamp is None:
        logger.warning(f"{log_prefix}  Failed to estimate target timestamp for frame {current_original_frame_index}. Returning blank frame.")
        return blank_bgr_frame

    # Find the index (iloc) within the *segment DataFrame* closest to the target timestamp
    target_iloc_in_segment = find_nearest_iloc(behavior_timestamps_segment, target_timestamp)
    if target_iloc_in_segment == -1:
        logger.warning(f"{log_prefix}  Failed to find nearest iloc for target timestamp {target_timestamp}. Returning blank frame.")
        return blank_bgr_frame

    if is_debug_frame:
        logger.debug(f"{log_prefix}  Target Timestamp: {target_timestamp:.2f}")
        logger.debug(f"{log_prefix}  Target iloc in segment: {target_iloc_in_segment}")


    # --- Determine Data Window for Plotting ---
    # Show a window centered around the current time point, with more data shown from the past.
    window_total_size_points = config.TRACE_PLOT_WINDOW_SIZE # Number of data points in the plot window
    points_past = int(window_total_size_points * 0.8) # e.g., 80% of window shows past data
    points_future = window_total_size_points - points_past

    # Calculate window boundaries (iloc) within the segment DataFrame
    start_iloc_plot = max(0, target_iloc_in_segment - points_past)
    end_iloc_plot = min(len(filtered_segment_df), target_iloc_in_segment + points_future)

    # Adjust start if end hit the boundary first, to maintain window size if possible
    start_iloc_plot = max(0, end_iloc_plot - window_total_size_points)

    window_df_plot = filtered_segment_df.iloc[start_iloc_plot:end_iloc_plot]

    if window_df_plot.empty:
        if is_debug_frame: logger.warning(f"{log_prefix}  Plotting window is empty (iloc [{start_iloc_plot}:{end_iloc_plot}]). Returning blank frame.")
        return blank_bgr_frame

    if is_debug_frame:
        logger.debug(f"{log_prefix}  Plotting Window iloc: [{start_iloc_plot}, {end_iloc_plot})")
        logger.debug(f"{log_prefix}  Plotting Window shape: {window_df_plot.shape}")

    # --- Calculate Relative Time X-axis for Plotting ---
    TIMESTAMP_SCALE_FACTOR = 1e9 # ns -> s
    window_timestamps = window_df_plot["BehaviorTimestamp"].values
    # Time relative to the 'current' frame's estimated timestamp
    x_values_relative_time_sec = (window_timestamps - target_timestamp) / TIMESTAMP_SCALE_FACTOR

    # Get actual time range for setting x-limits
    # Avoid issues if x_values array is empty (shouldn't happen due to checks above)
    min_rel_time = x_values_relative_time_sec.min() if len(x_values_relative_time_sec) > 0 else -1.0
    max_rel_time = x_values_relative_time_sec.max() if len(x_values_relative_time_sec) > 0 else 1.0
    # Add slight padding to x-limits? Maybe not needed if window size is fixed.
    plot_xlim = (min_rel_time, max_rel_time)

    # --- Setup Matplotlib Figure ---
    # Create figure and axes ONCE per frame generation
    fig, axes = plt.subplots(
        2, 1, # Two vertically stacked plots
        figsize=(target_width / 100, target_height / 100), # inches
        dpi=100, # Renders at target_width x target_height pixels
        sharex=True # Share the x-axis (relative time)
    )
    # Ensure axes is always a list/array, even if subplots returns a single Axes object
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor('white') # White background for the figure area outside axes


    # --- Plotting Loop ---
    regions_to_plot = [left_region_name, right_region_name]
    colors_to_plot = [config.TRACE_PLOT_COLOR_LEFT, config.TRACE_PLOT_COLOR_RIGHT]
    plotted_anything = False # Flag to track if any valid data was plotted

    for i, (ax, region, color) in enumerate(zip(axes, regions_to_plot, colors_to_plot)):
        filtered_col_name = f"{region}_filtered" # Column name convention from photometry_processor
        if filtered_col_name not in window_df_plot.columns:
            logger.warning(f"{log_prefix}  Filtered column '{filtered_col_name}' not found in plotting window.")
            # Optionally plot a flat line or leave blank
            ax.text(0.5, 0.5, f'{region}\nData Missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=10)
            y_values_to_plot = [] # Empty list
        else:
            y_values_to_plot = window_df_plot[filtered_col_name].values

            # Check if data exists for plotting
            if len(x_values_relative_time_sec) != len(y_values_to_plot):
                 logger.error(f"{log_prefix}  Mismatch between X ({len(x_values_relative_time_sec)}) and Y ({len(y_values_to_plot)}) "
                              f"lengths for {region}. Skipping plot for this region.")
                 continue # Skip plotting this region

            # Check for all NaNs before plotting
            if np.all(np.isnan(y_values_to_plot)):
                 if is_debug_frame: logger.debug(f"{log_prefix}  All Y values are NaN for {region}. Plotting empty.")
                 # Plot nothing, let the axes setup handle appearance
            else:
                 # Plot the filtered data
                 ax.plot(x_values_relative_time_sec, y_values_to_plot, color=color, linewidth=config.TRACE_PLOT_LINEWIDTH)
                 plotted_anything = True
                 if is_debug_frame:
                    y_min, y_max = np.nanmin(y_values_to_plot), np.nanmax(y_values_to_plot)
                    logger.debug(f"{log_prefix}  Plotting {region}: {len(y_values_to_plot)} points, Range [{y_min:.2f}, {y_max:.2f}]")


        # --- Configure Axes Appearance ---
        ax.set_ylabel(f"{region}\n(Z-Score)", fontsize=9)
        ax.set_ylim(plot_ylim) # Apply the calculated global Y limits

        # Add vertical line at t=0 (current frame time)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Set x-limits based on the window's relative time range
        ax.set_xlim(plot_xlim)

        # Tick styling
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both')) # Auto ticks, limit number
        ax.tick_params(axis='x', which='major', labelsize=8)

        # Grid and background
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_facecolor('#FAFAFA') # Light gray background for plot area

        # Add X-axis label only to the bottom plot
        if i == len(axes) - 1:
            ax.set_xlabel("Time relative to current frame (s)", fontsize=9)
        else:
            # Hide x-tick labels for upper plots if sharing x-axis
            ax.tick_params(axis='x', labelbottom=False)


    # Add overall title (optional) - using Behavior Frame index
    axes[0].set_title(f"Behavior Frame: {current_original_frame_index}", fontsize=10, pad=6)
    # axes[0].set_title(f"Behavior Frame: {current_original_frame_index}",
    #                   fontsize=10, y=1.02)

    # Adjust layout to prevent labels overlapping
    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for title/xlabel
    except ValueError:
        logger.warning(f"{log_prefix}  tight_layout failed. Plot appearance might be suboptimal.")


    # --- Render Matplotlib Figure to NumPy Array ---
    # Use a try/finally block to ensure plt.close() is called
    img_bgr = blank_bgr_frame # Default to blank if rendering fails
    try:
        canvas = FigureCanvas(fig)
        canvas.draw() # Render the canvas

        # Get image data as RGB buffer
        img_rgb = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8) # Use buffer_rgba for compatibility

        # Reshape based on canvas dimensions, handling potential RGBA -> RGB conversion
        width, height = canvas.get_width_height()
        if img_rgb.size == width * height * 4: # RGBA buffer
             img_rgb = img_rgb.reshape(height, width, 4)
             img_rgb = img_rgb[:, :, :3] # Drop alpha channel
        elif img_rgb.size == width * height * 3: # RGB buffer
             img_rgb = img_rgb.reshape(height, width, 3)
        else:
             raise ValueError(f"Unexpected buffer size: {img_rgb.size} != {width*height*3} or {width*height*4}")

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if is_debug_frame:
            logger.debug(f"{log_prefix}  Rendered plot shape: {img_bgr.shape}")
            # Check if rendered image is blank
            if np.all(img_bgr == 255): logger.warning(f"{log_prefix}  -> Rendered plot is all white!")
            elif np.all(img_bgr == 0): logger.warning(f"{log_prefix}  -> Rendered plot is all black!")


    except Exception as e:
        logger.error(f"{log_prefix}Failed to render matplotlib canvas for frame {current_original_frame_index}: {e}", exc_info=True)
        # Return the blank frame initialized earlier
    finally:
        # IMPORTANT: Close the figure to release memory
        plt.close(fig)

    return img_bgr