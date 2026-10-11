"""Spatial transforms for tracking, independent of camera acquisition files."""

import numpy as np


def rot2d(theta, *, dtype=None):
    """Rotation matrices with shape ``theta.shape + (2, 2)``.

    Compute trigonometry at the input precision and cast the matrix on output.
    Legacy tracking consumers explicitly request float32 matrices.
    """
    theta = np.asarray(theta)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.empty(theta.shape + (2, 2), dtype=dtype or np.result_type(cosine, sine))
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = -sine
    rotation[..., 1, 0] = sine
    rotation[..., 1, 1] = cosine
    return rotation


def apply_affine(coordinates, matrix, translation, *, dtype=None):
    """Apply an affine map; explicit dtype controls each arithmetic operand.

    The default retains the foundation API's float64 calibration arithmetic.
    """
    coordinates = np.asarray(coordinates, dtype=dtype)
    matrix = np.asarray(matrix, dtype=float if dtype is None else dtype)
    translation = np.asarray(translation, dtype=float if dtype is None else dtype)
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2 or matrix.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("affine transform requires coordinates(...,2), matrix(2,2), translation(2,)")
    if not np.isfinite(matrix).all() or not np.isfinite(translation).all():
        raise ValueError("affine calibration must be finite")
    return coordinates @ matrix.T + translation


def reflect_horizontal(coordinates, *, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("reflection width must be finite and positive")
    return apply_affine(coordinates, np.diag([-1.0, 1.0]), [width, 0])


def project_reflected_points(coordinates, matrix, translation, *, width, dtype=float):
    """Mirror x, then apply an affine map with the chosen arithmetic precision.

    Keep the two operations separate: combining them into one matrix can change
    float32 rounding relative to the fused tracking artifact.
    """
    if not np.isfinite(width) or width <= 0:
        raise ValueError("reflection width must be finite and positive")
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError("projection arithmetic dtype must be floating point")
    reflected = np.asarray(coordinates, dtype=dtype).copy()
    if reflected.ndim < 1 or reflected.shape[-1] != 2:
        raise ValueError("reflection requires coordinates(...,2)")
    reflected[..., 0] = np.asarray(width, dtype=dtype) - reflected[..., 0]
    return apply_affine(reflected, matrix, translation, dtype=dtype)
