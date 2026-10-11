"""
logging_utils.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.logging.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.common.logging")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
