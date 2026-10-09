"""
eval_artifacts.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.kpms.evaluation.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.kpms.evaluation")
sys.modules[__name__] = _real
