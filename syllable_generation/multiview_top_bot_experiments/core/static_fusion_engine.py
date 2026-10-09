"""
static_fusion_engine.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.tracking.static_fusion.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.tracking.static_fusion")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
