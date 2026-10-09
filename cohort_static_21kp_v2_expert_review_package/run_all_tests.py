#!/usr/bin/env python3
"""
run_all_tests.py
================
Self-contained synthetic unit test runner for the Multi-View Tandem Review Package.
Runs all synthetic tests purely on CPU without requiring external project data.
"""

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Also support subdirectories if pipeline scripts are invoked
pipeline_dir = PACKAGE_ROOT / "pipeline"
if pipeline_dir.exists() and str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

def main():
    print("=" * 80)
    print("🧪 RUNNING MULTI-VIEW TANDEM SYNTHETIC REGRESSION TEST SUITE")
    print(f"  Package Root: {PACKAGE_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PACKAGE_ROOT / "tests"), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"✅ ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        return 0
    else:
        print(f"❌ FAILURES OCCURRED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
