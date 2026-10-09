"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/run_relational_topology_and_heading_reset_movies.py
==================================================
Relocated to: studies/behavior/sequence_viz/run_relational_topology_and_heading_reset_movies.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.run_relational_topology_and_heading_reset_movies is deprecated; import from studies.behavior.sequence_viz.run_relational_topology_and_heading_reset_movies instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.run_relational_topology_and_heading_reset_movies")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
