"""
[DEPRECATED SHIM] fiberphotometry/kpms/syllable_generator.py
==================================================
Relocated to: studies/photometry/exploratory/kpms/syllable_generator.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.kpms.syllable_generator is deprecated; import from studies.photometry.exploratory.kpms.syllable_generator instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.kpms.syllable_generator")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
