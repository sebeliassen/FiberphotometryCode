"""
[DEPRECATED SHIM] core/pair_matching.py
==========================================
Relocated to: fpcode.tracking.pair_matching
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.pair_matching is deprecated; import from fpcode.tracking.pair_matching instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.tracking.pair_matching")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
