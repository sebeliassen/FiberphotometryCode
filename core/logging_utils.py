"""
[DEPRECATED SHIM] core/logging_utils.py
==========================================
Relocated to: fpcode.common.logging
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.logging_utils is deprecated; import from fpcode.common.logging instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.logging")
sys.modules[__name__] = _real
