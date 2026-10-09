"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/diagnose_tandem_tile_heterogeneity.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/diagnose_tandem_tile_heterogeneity.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.diagnose_tandem_tile_heterogeneity is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.diagnose_tandem_tile_heterogeneity instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.diagnose_tandem_tile_heterogeneity")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
