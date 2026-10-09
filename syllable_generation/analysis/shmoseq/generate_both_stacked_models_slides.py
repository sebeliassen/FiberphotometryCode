"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/generate_both_stacked_models_slides.py
==================================================
Relocated to: studies/behavior/shmoseq/generate_both_stacked_models_slides.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.generate_both_stacked_models_slides is deprecated; import from studies.behavior.shmoseq.generate_both_stacked_models_slides instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.generate_both_stacked_models_slides")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
