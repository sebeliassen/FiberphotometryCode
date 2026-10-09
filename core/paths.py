"""
paths.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.paths.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.common.paths")
sys.modules[__name__] = _real
