"""
run_record.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.run_record.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.common.run_record")
sys.modules[__name__] = _real
