"""Historical KPMS export API; session composition lives in workflows."""

from fpcode.adapters.legacy_sessions import LEGACY_TANDEM_PHOTOMETRY_COLUMNS as TEMPLATE_PHOT_COLS
from fpcode.kpms.results import extract_and_export_results
from fpcode.workflows.session_export import export_tandem_sessions


def export_tandem_cohort_results(results_h5, dirty_pkl, output_pkl, output_data_pkl=None,
                                 k240k_ref_pkl=None, *, pairs=None):
    """Keep the five positional arguments and historical cohort discovery defaults.

    Explicit matched pairs avoid directory discovery. Row-prefix camera sampling,
    60 Hz kinematics and legacy 470 photometry interpolation remain unchanged.
    """
    if pairs is None:
        from fpcode.tracking.pair_matching import find_matched_pairs
        pairs = find_matched_pairs()
    return export_tandem_sessions(results_h5, dirty_pkl, output_pkl, pairs=pairs,
                                  output_data_pkl=output_data_pkl, k240k_ref_pkl=k240k_ref_pkl)


__all__ = ["TEMPLATE_PHOT_COLS", "extract_and_export_results", "export_tandem_cohort_results"]
