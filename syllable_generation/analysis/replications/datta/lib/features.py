"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta/lib/features.py
==================================================
Relocated to: studies/behavior/replications_datta/datta/lib/features.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta.lib.features is deprecated; import from studies.behavior.replications_datta.datta.lib.features instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.datta.lib.features")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
