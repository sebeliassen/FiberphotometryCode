"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/batch_init.py
========================================================================================
Relocated to: fpcode.kpms.init_model
"""

import importlib
import sys
import warnings

warnings.warn(
    "utils.batch_init is deprecated; import from fpcode.kpms.init_model instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.init_model")
sys.modules[__name__] = _real
