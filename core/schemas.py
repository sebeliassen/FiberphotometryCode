"""
[DEPRECATED SHIM] core/schemas.py
==========================================
Relocated to: fpcode.common.schemas
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.schemas is deprecated; import from fpcode.common.schemas instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.schemas")
sys.modules[__name__] = _real
