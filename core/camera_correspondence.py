"""
[DEPRECATED SHIM] core/camera_correspondence.py
==========================================
Relocated to: fpcode.tracking.clock_sync
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.camera_correspondence is deprecated; import from fpcode.tracking.clock_sync instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.tracking.camera_correspondence")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
