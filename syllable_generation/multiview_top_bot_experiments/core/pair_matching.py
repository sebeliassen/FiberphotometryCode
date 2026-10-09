"""
pair_matching.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.tracking.pair_matching.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.tracking.pair_matching")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
