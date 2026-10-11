"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/24_generate_exemplar_breakdown_tables.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/24_generate_exemplar_breakdown_tables.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.24_generate_exemplar_breakdown_tables is deprecated; import from studies.behavior.language_models.pipelines.24_generate_exemplar_breakdown_tables instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.24_generate_exemplar_breakdown_tables")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
