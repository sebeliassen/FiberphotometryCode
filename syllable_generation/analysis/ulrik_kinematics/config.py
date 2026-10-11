"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/config.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/config.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.config is deprecated; import from studies.behavior.kinematics_ulrik.config instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.config")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
