"""
[DEPRECATED SHIM] core/static_fusion_engine.py
==========================================
Relocated to: fpcode.tracking.fusion_engine
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.static_fusion_engine is deprecated; import from fpcode.tracking.fusion_engine instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.tracking.fusion_engine")
sys.modules[__name__] = _real
