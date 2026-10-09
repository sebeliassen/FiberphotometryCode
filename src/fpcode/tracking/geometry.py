"""Spatial transforms for tracking, independent of camera acquisition files."""

import numpy as np


def rot2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def apply_affine(coordinates, matrix, translation):
    coordinates = np.asarray(coordinates)
    matrix = np.asarray(matrix, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2 or matrix.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("affine transform requires coordinates(...,2), matrix(2,2), translation(2,)")
    if not np.isfinite(matrix).all() or not np.isfinite(translation).all():
        raise ValueError("affine calibration must be finite")
    return coordinates @ matrix.T + translation


def reflect_horizontal(coordinates, *, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("reflection width must be finite and positive")
    return apply_affine(coordinates, np.diag([-1.0, 1.0]), [width, 0])
