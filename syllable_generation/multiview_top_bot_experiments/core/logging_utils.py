"""
logging_utils.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.logging.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.common.logging")
sys.modules[__name__] = _real
