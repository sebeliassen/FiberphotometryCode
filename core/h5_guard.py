"""
h5_guard.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.h5_guard.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.common.h5_guard")
sys.modules[__name__] = _real
