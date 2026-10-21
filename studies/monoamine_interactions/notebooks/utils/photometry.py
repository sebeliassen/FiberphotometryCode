"""Compatibility imports for the historical monoamine correction algorithm."""

from fpcode.photometry.preprocessing import (
    biexponential,
    monoexponential,
    fit_biexponential_dff,
    motion_correct_biexponential,
)

__all__ = ["biexponential", "monoexponential", "fit_biexponential_dff", "motion_correct_biexponential"]
