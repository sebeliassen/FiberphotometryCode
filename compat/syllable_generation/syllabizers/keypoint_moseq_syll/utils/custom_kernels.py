"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/custom_kernels.py
==================================================
Relocated to: vendor/keypoint_moseq_syll/utils/custom_kernels.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.syllabizers.keypoint_moseq_syll.utils.custom_kernels is deprecated; import from vendor.keypoint_moseq_syll.utils.custom_kernels instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("vendor.keypoint_moseq_syll.utils.custom_kernels")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
