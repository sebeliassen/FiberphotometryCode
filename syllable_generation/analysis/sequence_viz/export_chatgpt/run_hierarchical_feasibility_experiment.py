"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/export_chatgpt/run_hierarchical_feasibility_experiment.py
==================================================
Relocated to: studies/behavior/sequence_viz/export_chatgpt/run_hierarchical_feasibility_experiment.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.export_chatgpt.run_hierarchical_feasibility_experiment is deprecated; import from studies.behavior.sequence_viz.export_chatgpt.run_hierarchical_feasibility_experiment instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.export_chatgpt.run_hierarchical_feasibility_experiment")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
