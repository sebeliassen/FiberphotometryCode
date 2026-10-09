"""
[DEPRECATED SHIM] fiberphotometry/notebooks/sebastian/signal_action_prediction.py
==================================================
Relocated to: studies/photometry/exploratory/notebooks/sebastian/signal_action_prediction.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.notebooks.sebastian.signal_action_prediction is deprecated; import from studies.photometry.exploratory.notebooks.sebastian.signal_action_prediction instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.notebooks.sebastian.signal_action_prediction")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
