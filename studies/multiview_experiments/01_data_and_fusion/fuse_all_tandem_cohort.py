#!/usr/bin/env python3
"""[FORWARDER] Relocated to pipelines/fuse.py --config configs/fusion/all_tandem_cohort.yaml"""
import subprocess
import sys

from fpcode.common.paths import REPO_ROOT

MODE = "tandem_23kp"
ARENA_WIDTH = 972.0

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/fusion/all_tandem_cohort.yaml"
    sys.exit(subprocess.call([sys.executable, str(REPO_ROOT / "pipelines/fuse.py"), "--config", str(cfg)] + sys.argv[1:]))
