"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/hierarchical_pilot/test_continuous_emissions.py
==================================================
Relocated to: studies/behavior/sequence_viz/hierarchical_pilot/test_continuous_emissions.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.hierarchical_pilot.test_continuous_emissions is deprecated; import from studies.behavior.sequence_viz.hierarchical_pilot.test_continuous_emissions instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.hierarchical_pilot.test_continuous_emissions")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
