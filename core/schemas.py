"""
schemas.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.schemas.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.common.schemas")
sys.modules[__name__] = _real
