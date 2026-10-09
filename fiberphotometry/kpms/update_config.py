"""
[DEPRECATED SHIM] fiberphotometry/kpms/update_config.py
==================================================
Relocated to: studies/photometry/exploratory/kpms/update_config.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.kpms.update_config is deprecated; import from studies.photometry.exploratory.kpms.update_config instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.kpms.update_config")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
