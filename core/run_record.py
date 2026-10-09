"""
[DEPRECATED SHIM] core/run_record.py
==========================================
Relocated to: fpcode.common.run_record
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.run_record is deprecated; import from fpcode.common.run_record instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.common.run_record")
sys.modules[__name__] = _real
