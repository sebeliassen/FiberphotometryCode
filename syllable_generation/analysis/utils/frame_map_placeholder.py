"""
[DEPRECATED SHIM] syllable_generation/analysis/utils/frame_map_placeholder.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/utils/frame_map_placeholder.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.utils.frame_map_placeholder is deprecated; import from studies.behavior.misc.syllable_generation.analysis.utils.frame_map_placeholder instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.utils.frame_map_placeholder")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
