"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/hierarchical_pilot/generate_4state_interactive_html.py
==================================================
Relocated to: studies/behavior/sequence_viz/hierarchical_pilot/generate_4state_interactive_html.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.hierarchical_pilot.generate_4state_interactive_html is deprecated; import from studies.behavior.sequence_viz.hierarchical_pilot.generate_4state_interactive_html instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.hierarchical_pilot.generate_4state_interactive_html")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
