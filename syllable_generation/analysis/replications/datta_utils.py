"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta_utils.py
========================================================================
Relocated to: fpcode.photometry.alignment
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta_utils is deprecated; "
    "import from fpcode.photometry.alignment instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.photometry.alignment")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

combined_photometry_and_syllables = _real.combined_photometry_and_syllables
build_fixed_moseq_dfs = _real.build_fixed_moseq_dfs
