"""
[DEPRECATED SHIM] core/h5_guard.py
==========================================
Relocated to: fpcode.common.h5_guard
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.h5_guard is deprecated; import from fpcode.common.h5_guard instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.h5_guard")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
