"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/extract_level1_chain_quantitative_results.py
==================================================
Relocated to: studies/behavior/sequence_viz/extract_level1_chain_quantitative_results.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.extract_level1_chain_quantitative_results is deprecated; import from studies.behavior.sequence_viz.extract_level1_chain_quantitative_results instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.extract_level1_chain_quantitative_results")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
