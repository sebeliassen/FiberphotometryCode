#!/usr/bin/env python3
"""[FORWARDER] Relocated to pipelines/train.py --config configs/training/cohort_static_21kp_v2_seed101.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
TARGET_FPS = 60
NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 300
TOTAL_ITERS = 350
SAVE_EVERY_N_ITERS = 25

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/training/cohort_static_21kp_v2_seed101.yaml"
    sys.exit(subprocess.call([sys.executable, str(REPO_ROOT / "pipelines/train.py"), "--config", str(cfg)] + sys.argv[1:]))
