"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta_orig/processing/01_generate_datta_corrs.py
==================================================
Relocated to: studies/behavior/replications_datta/datta_orig/processing/01_generate_datta_corrs.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta_orig.processing.01_generate_datta_corrs is deprecated; import from studies.behavior.replications_datta.datta_orig.processing.01_generate_datta_corrs instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.datta_orig.processing.01_generate_datta_corrs")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
