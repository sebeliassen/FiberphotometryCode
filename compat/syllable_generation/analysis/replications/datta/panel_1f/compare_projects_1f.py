"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta/panel_1f/compare_projects_1f.py
==================================================
Relocated to: studies/behavior/replications_datta/datta/panel_1f/compare_projects_1f.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta.panel_1f.compare_projects_1f is deprecated; import from studies.replications_datta.datta.panel_1f.compare_projects_1f instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.replications_datta.datta.panel_1f.compare_projects_1f")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
