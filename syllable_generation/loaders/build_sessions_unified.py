"""
[DEPRECATED SHIM] syllable_generation/loaders/build_sessions_unified.py
==================================================
Relocated to: studies/behavior/loaders/build_sessions_unified.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.loaders.build_sessions_unified is deprecated; import from studies.behavior.loaders.build_sessions_unified instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.loaders.build_sessions_unified")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
