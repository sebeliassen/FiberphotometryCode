"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta_orig/panels/1e/digest_shuffles.py
==================================================
Relocated to: studies/behavior/replications_datta/datta_orig/panels/1e/digest_shuffles.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta_orig.panels.1e.digest_shuffles is deprecated; import from studies.replications_datta.datta_orig.panels.1e.digest_shuffles instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.replications_datta.datta_orig.panels.1e.digest_shuffles")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
