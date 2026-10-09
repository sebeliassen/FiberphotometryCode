"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/export_tandem_cohort_production_results.py
===============================================================================================================================
Relocated to: fpcode.kpms.export
"""

import importlib
import sys
import warnings

warnings.warn(
    "export_tandem_cohort_production_results is deprecated; import from fpcode.kpms.export instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.export")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

export_tandem_cohort_results = _real.export_tandem_cohort_results

if __name__ == "__main__":
    from fpcode.common.paths import REPO_ROOT
    results_h5 = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production/tandem_cohort_118sess_50ar_200slds/results.h5"
    dirty_pkl = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
    output_prod_pkl = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl"
    output_data_pkl = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl"
    k240k_ref = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
    _real.export_tandem_cohort_results(results_h5, dirty_pkl, output_prod_pkl, output_data_pkl, k240k_ref)
