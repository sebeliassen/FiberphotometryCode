"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py
=================================================================================================
Relocated to: fpcode.kpms.runtime
"""

import importlib
import sys
import warnings

warnings.warn(
    "utils.resample_batching is deprecated; import from fpcode.kpms.runtime instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.runtime")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

# The module alias already preserves every canonical attribute and its identity.
# Eager attribute bindings would defeat the runtime facade's lazy backend loading.
