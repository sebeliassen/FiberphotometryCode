#!/usr/bin/env python3
"""[FORWARDER] Uses python -m fpcode.cli.train --config configs/training/pilot_static_comparison.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODEL_NAME = "pilot_static_comparison"
TARGET_FPS = 60
NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 100
TOTAL_ITERS = 150
INIT_KAPPA = 1.0e6
DECREASE_KAPPA_FACTOR = 10.0

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/training/pilot_static_comparison.yaml"
    sys.exit(subprocess.call([sys.executable, "-m", "fpcode.cli.train", "--config", str(cfg)] + sys.argv[1:]))
