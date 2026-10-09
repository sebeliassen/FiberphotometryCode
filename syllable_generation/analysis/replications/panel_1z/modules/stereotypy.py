"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/panel_1z/modules/stereotypy.py
==================================================
Relocated to: studies/behavior/replications_datta/panel_1z/modules/stereotypy.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.panel_1z.modules.stereotypy is deprecated; import from studies.behavior.replications_datta.panel_1z.modules.stereotypy instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.panel_1z.modules.stereotypy")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
