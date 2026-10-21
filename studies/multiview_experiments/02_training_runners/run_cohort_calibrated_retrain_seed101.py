#!/usr/bin/env python3
"""[FORWARDER] Uses python -m fpcode.cli.train --config configs/training/cohort_calibrated_retrain_seed101.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODEL_NAME = "tandem_cohort_calibrated_50ar_200slds_seed101"
TARGET_FPS = 60
NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 200
TOTAL_ITERS = 250
SAVE_EVERY_N_ITERS = 25
INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = 7.5e12

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/training/cohort_calibrated_retrain_seed101.yaml"
    sys.exit(subprocess.call([sys.executable, "-m", "fpcode.cli.train", "--config", str(cfg)] + sys.argv[1:]))
