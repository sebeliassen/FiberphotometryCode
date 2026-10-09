"""
[DEPRECATED SHIM] syllable_generation/analysis/deep_research_experiments/peth_lateralized_da_virtual.py
==================================================
Relocated to: studies/behavior/deep_research_experiments/peth_lateralized_da_virtual.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.peth_lateralized_da_virtual is deprecated; import from studies.behavior.deep_research_experiments.peth_lateralized_da_virtual instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.deep_research_experiments.peth_lateralized_da_virtual")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
