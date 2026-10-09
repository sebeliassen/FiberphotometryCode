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

BatchConfig = _real.BatchConfig
CONFIG = _real.CONFIG
precompute_optimal_batch_config = _real.precompute_optimal_batch_config
apply_optimizations = _real.apply_optimizations
restore_original_kernels = _real.restore_original_kernels
kpms_optimizations = _real.kpms_optimizations
KPMSStateCache = _real.KPMSStateCache
