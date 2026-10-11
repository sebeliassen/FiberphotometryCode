"""
[DEPRECATED SHIM] core/static_fusion_engine.py
==========================================
Relocated to: fpcode.tracking.static_fusion
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.static_fusion_engine is deprecated; import from fpcode.tracking.static_fusion instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.tracking.static_fusion")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
