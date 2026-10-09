"""
evaluation_metrics.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.kpms.metrics.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.kpms.metrics")
sys.modules[__name__] = _real
