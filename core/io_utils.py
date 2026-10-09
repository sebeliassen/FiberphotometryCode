"""
[DEPRECATED SHIM] core/io_utils.py
==========================================
Relocated to: fpcode.common.io
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.io_utils is deprecated; import from fpcode.common.io instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.io")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
