#!/usr/bin/env python3
"""[FORWARDER] Relocated to pipelines/train.py --config configs/training/tandem_cohort_production_seed101.yaml"""
import subprocess
import sys
from fpcode.common.paths import REPO_ROOT

MODEL_NAME = "tandem_cohort_118sess_50ar_200slds"
TARGET_FPS = 60
NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 200
TOTAL_ITERS = 250
SAVE_EVERY_N_ITERS = 25
INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = 7.5e12


if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/training/tandem_cohort_production_seed101.yaml"
    sys.exit(subprocess.call([sys.executable, str(REPO_ROOT / "pipelines/train.py"), "--config", str(cfg)] + sys.argv[1:]))
