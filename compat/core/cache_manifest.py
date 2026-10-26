"""
[DEPRECATED SHIM] core/cache_manifest.py
==========================================
Relocated to: fpcode.common.cache_manifest
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.cache_manifest is deprecated; import from fpcode.common.cache_manifest instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.cache_manifest")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
