"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta/panel_1e/plot_panel_1e_unified.py
==================================================
Relocated to: studies/behavior/replications_datta/datta/panel_1e/plot_panel_1e_unified.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta.panel_1e.plot_panel_1e_unified is deprecated; import from studies.behavior.replications_datta.datta.panel_1e.plot_panel_1e_unified instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.datta.panel_1e.plot_panel_1e_unified")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
