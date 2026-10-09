"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/panel_1z/modules/geometry.py
==================================================
Relocated to: studies/behavior/replications_datta/panel_1z/modules/geometry.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.panel_1z.modules.geometry is deprecated; import from studies.behavior.replications_datta.panel_1z.modules.geometry instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.panel_1z.modules.geometry")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
