"""Median smoothing kernels for circular angles and two-coordinate arrays."""

import numpy as np
from scipy.ndimage import median_filter


def circular_median(theta, window):
    sin_smooth = median_filter(np.sin(theta), size=window, mode="nearest")
    cos_smooth = median_filter(np.cos(theta), size=window, mode="nearest")
    return np.arctan2(sin_smooth, cos_smooth)


def coordinate_median(coordinates, window):
    return np.column_stack([
        median_filter(coordinates[:, 0], size=window, mode="nearest"),
        median_filter(coordinates[:, 1], size=window, mode="nearest"),
    ])
