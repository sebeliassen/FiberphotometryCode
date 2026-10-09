"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/lib/visualization/vibe_passports.py
==================================================
Relocated to: studies/behavior/language_models/lib/visualization/vibe_passports.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.lib.visualization.vibe_passports is deprecated; import from studies.behavior.language_models.lib.visualization.vibe_passports instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.lib.visualization.vibe_passports")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
