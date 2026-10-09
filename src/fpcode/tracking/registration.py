"""Affine registration of matched two-dimensional calibration landmarks."""

from dataclasses import dataclass

import numpy as np

from fpcode.contracts.timelines import frozen_array
from .geometry import apply_affine


@dataclass(frozen=True)
class AffineRegistration:
    matrix: np.ndarray
    translation: np.ndarray
    residual_rms: float

    def __post_init__(self):
        object.__setattr__(self, "matrix", frozen_array(self.matrix))
        object.__setattr__(self, "translation", frozen_array(self.translation))

    def apply(self, coordinates):
        return apply_affine(coordinates, self.matrix, self.translation)


def fit_affine(source_points, target_points):
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or source.shape != target.shape or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("at least three matched two-dimensional landmarks are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("calibration landmarks must be finite")
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("source landmarks are collinear; affine calibration is underdetermined")
    matrix, translation = coefficients[:2].T, coefficients[2]
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("target landmarks yield a singular transform")
    residual = apply_affine(source, matrix, translation) - target
    return AffineRegistration(matrix, translation, float(np.sqrt(np.mean(residual**2))))
