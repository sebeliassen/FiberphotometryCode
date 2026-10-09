"""
[DEPRECATED SHIM] fiberphotometry/kpms/load_and_analyze_session_state.py
==================================================
Relocated to: studies/photometry/exploratory/kpms/load_and_analyze_session_state.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.kpms.load_and_analyze_session_state is deprecated; import from studies.photometry.exploratory.kpms.load_and_analyze_session_state instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.kpms.load_and_analyze_session_state")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
