"""
[DEPRECATED SHIM] core/paths.py
==========================================
Relocated to: fpcode.common.paths
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.paths is deprecated; import from fpcode.common.paths instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.paths")
sys.modules[__name__] = _real
