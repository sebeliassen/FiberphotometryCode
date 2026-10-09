"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/find_similar_syllables.py
==================================================
Relocated to: studies/behavior/karen_studies/find_similar_syllables.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.find_similar_syllables is deprecated; import from studies.behavior.karen_studies.find_similar_syllables instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.find_similar_syllables")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
