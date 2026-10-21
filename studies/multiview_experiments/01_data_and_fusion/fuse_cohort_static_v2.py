#!/usr/bin/env python3
"""[FORWARDER] Uses python -m fpcode.cli.fuse --config configs/fusion/cohort_static_21kp_v2.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODE = "static_21kp_v2"
ARENA_WIDTH = 972.0

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/fusion/cohort_static_21kp_v2.yaml"
    sys.exit(subprocess.call([sys.executable, "-m", "fpcode.cli.fuse", "--config", str(cfg)] + sys.argv[1:]))
