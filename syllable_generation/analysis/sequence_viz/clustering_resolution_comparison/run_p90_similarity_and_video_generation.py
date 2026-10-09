"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/run_p90_similarity_and_video_generation.py
==================================================
Relocated to: studies/behavior/sequence_viz/clustering_resolution_comparison/run_p90_similarity_and_video_generation.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_p90_similarity_and_video_generation is deprecated; import from studies.behavior.sequence_viz.clustering_resolution_comparison.run_p90_similarity_and_video_generation instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.clustering_resolution_comparison.run_p90_similarity_and_video_generation")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
