"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/lib/evaluation/kinematic_encoding.py
==================================================
Relocated to: studies/behavior/language_models/lib/evaluation/kinematic_encoding.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.lib.evaluation.kinematic_encoding is deprecated; import from studies.behavior.language_models.lib.evaluation.kinematic_encoding instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.lib.evaluation.kinematic_encoding")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
