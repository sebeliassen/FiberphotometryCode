"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/02_frame_vs_bout/test_2x2_cross_prediction.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/02_frame_vs_bout/test_2x2_cross_prediction.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.02_frame_vs_bout.test_2x2_cross_prediction is deprecated; import from studies.behavior.language_models.pipelines.02_frame_vs_bout.test_2x2_cross_prediction instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.02_frame_vs_bout.test_2x2_cross_prediction")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
