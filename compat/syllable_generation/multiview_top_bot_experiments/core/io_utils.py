"""
io_utils.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.io.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.common.io")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
