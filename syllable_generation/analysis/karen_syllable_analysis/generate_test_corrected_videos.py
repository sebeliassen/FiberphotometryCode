"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/generate_test_corrected_videos.py
==================================================
Relocated to: studies/behavior/karen_studies/generate_test_corrected_videos.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.generate_test_corrected_videos is deprecated; import from studies.behavior.karen_studies.generate_test_corrected_videos instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.generate_test_corrected_videos")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
