"""
io_utils.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.io.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.common.io")
sys.modules[__name__] = _real
