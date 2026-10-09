"""
[DEPRECATED SHIM] syllable_generation/analysis/deep_research_experiments/plot_lateralized_da.py
==================================================
Relocated to: studies/behavior/deep_research_experiments/plot_lateralized_da.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.plot_lateralized_da is deprecated; import from studies.behavior.deep_research_experiments.plot_lateralized_da instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.deep_research_experiments.plot_lateralized_da")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
