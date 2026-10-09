"""
[DEPRECATED SHIM] syllable_generation/analysis/attach_karen_final_model.py
==================================================
Relocated to: studies/behavior/karen_studies/attach_karen_final_model.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.attach_karen_final_model is deprecated; import from studies.behavior.karen_studies.attach_karen_final_model instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.attach_karen_final_model")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
