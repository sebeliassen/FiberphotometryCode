"""
[DEPRECATED SHIM] syllable_generation/kpms_studio/launch_gui.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/kpms_studio/launch_gui.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.kpms_studio.launch_gui is deprecated; import from studies.behavior.misc.syllable_generation.kpms_studio.launch_gui instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.kpms_studio.launch_gui")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
