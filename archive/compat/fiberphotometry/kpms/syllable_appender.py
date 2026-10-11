"""
[DEPRECATED SHIM] fiberphotometry/kpms/syllable_appender.py
==================================================
Relocated to: studies/photometry/exploratory/kpms/syllable_appender.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.kpms.syllable_appender is deprecated; import from studies.photometry.exploratory.kpms.syllable_appender instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.kpms.syllable_appender")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
