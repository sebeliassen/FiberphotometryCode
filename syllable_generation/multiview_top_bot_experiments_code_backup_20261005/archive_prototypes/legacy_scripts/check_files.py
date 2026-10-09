#!/usr/bin/env python3
import sys
from pathlib import Path

repo = Path("/home/fsp585/Desktop/FiberphotometryCode")
dlc_prod = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")

top_dir = repo / "karen_project" / "h5_files"
t1_top = sorted(list(top_dir.glob("*T1*05.03.26*.h5")))
print("TopCam T1 files:")
for f in t1_top:
    print(f"  {f.name}")

print("\nBotCam completed T1 files:")
bot_t1 = sorted(list(dlc_prod.glob("*T1*050326*/*.h5")))
for f in bot_t1:
    print(f"  {f.parent.name} -> {f.name}")
