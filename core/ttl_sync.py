"""
[DEPRECATED SHIM] core/ttl_sync.py
==========================================
Relocated to: fpcode.tracking.clock_sync
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.ttl_sync is deprecated; import from fpcode.tracking.clock_sync instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.tracking.clock_sync")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
