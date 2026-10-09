"""
[DEPRECATED SHIM] core/evaluation_metrics.py
==========================================
Relocated to: fpcode.kpms.metrics
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.evaluation_metrics is deprecated; import from fpcode.kpms.metrics instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.metrics")
sys.modules[__name__] = _real
