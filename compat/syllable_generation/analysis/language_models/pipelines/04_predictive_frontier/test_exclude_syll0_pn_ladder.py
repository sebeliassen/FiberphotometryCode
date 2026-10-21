"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/test_exclude_syll0_pn_ladder.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/04_predictive_frontier/test_exclude_syll0_pn_ladder.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.04_predictive_frontier.test_exclude_syll0_pn_ladder is deprecated; import from studies.behavior.language_models.pipelines.04_predictive_frontier.test_exclude_syll0_pn_ladder instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.04_predictive_frontier.test_exclude_syll0_pn_ladder")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
