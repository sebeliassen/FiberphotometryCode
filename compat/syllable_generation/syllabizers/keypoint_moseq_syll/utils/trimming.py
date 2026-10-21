"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/trimming.py
========================================================================================
Relocated to: fpcode.kpms.data
"""

import importlib
import sys
import warnings

warnings.warn(
    "utils.trimming is deprecated; import from fpcode.kpms.data instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.data")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

_clean_short_runs = _real._clean_short_runs
presence_mask_from_conf = _real.presence_mask_from_conf
filter_with_masks = _real.filter_with_masks
clamp_conf_nonneg = _real.clamp_conf_nonneg
subset_spread_blocks = _real.subset_spread_blocks
subset_after_filter = _real.subset_after_filter
subset_by_activity = _real.subset_by_activity
