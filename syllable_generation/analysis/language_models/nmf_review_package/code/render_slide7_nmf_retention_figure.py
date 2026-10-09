"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/nmf_review_package/code/render_slide7_nmf_retention_figure.py
==================================================
Relocated to: studies/behavior/language_models/nmf_review_package/code/render_slide7_nmf_retention_figure.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.nmf_review_package.code.render_slide7_nmf_retention_figure is deprecated; import from studies.behavior.language_models.nmf_review_package.code.render_slide7_nmf_retention_figure instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.nmf_review_package.code.render_slide7_nmf_retention_figure")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
