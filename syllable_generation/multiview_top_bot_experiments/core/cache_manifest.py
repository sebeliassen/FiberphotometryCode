"""
cache_manifest.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.common.cache_manifest.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.common.cache_manifest")
sys.modules[__name__] = _real
