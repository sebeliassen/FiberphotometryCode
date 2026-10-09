"""
[DEPRECATED SHIM] core/eval_artifacts.py
==========================================
Relocated to: fpcode.kpms.evaluation
"""

import importlib
import sys
import warnings

warnings.warn(
    "core.eval_artifacts is deprecated; import from fpcode.kpms.evaluation instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.evaluation")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
