#!/usr/bin/env python3
"""[FORWARDER] Relocated to pipelines/train.py --config configs/training/pilot_17kp_vs_21kp_comparison.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODEL_NAME = "pilot_17kp_vs_21kp_comparison"
TARGET_FPS = 60
NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 150
TOTAL_ITERS = 200
SAVE_EVERY_N_ITERS = 25
INIT_KAPPA = 1.0e6
SLDS_KAPPA = 1.5e12

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/training/pilot_17kp_vs_21kp_comparison.yaml"
    sys.exit(subprocess.call([sys.executable, str(REPO_ROOT / "pipelines/train.py"), "--config", str(cfg)] + sys.argv[1:]))
