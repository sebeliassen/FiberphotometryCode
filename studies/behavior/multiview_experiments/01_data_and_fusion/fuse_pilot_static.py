#!/usr/bin/env python3
"""[FORWARDER] Relocated to pipelines/fuse.py --config configs/fusion/pilot_static.yaml"""
import subprocess
import sys
from fpcode.common.paths import REPO_ROOT

MODE = "pilot_static"
ARENA_WIDTH = 972.0

if __name__ == "__main__":
    cfg = REPO_ROOT / "configs/fusion/pilot_static.yaml"
    sys.exit(subprocess.call([sys.executable, str(REPO_ROOT / "pipelines/fuse.py"), "--config", str(cfg)] + sys.argv[1:]))
