#!/usr/bin/env python3
"""
03_setup_multiview_kpms.py
--------------------------
Sets up the dedicated Keypoint-MoSeq project for Multi-View (TopCam + BotCam) pilot:
1. Initializes directory: syllable_generation/multiview_top_bot_experiments/kpms_multiview_pilot
2. Configures config.yml with:
   - Unified 23 bodyparts
   - Selected use_bodyparts (spine + 4 paws)
   - Anatomical skeleton linking dorsal shoulders/hips to ventral paws
   - Orientation priors (tail_base -> nose)
   - Optimized AR-SLDS hyperparameters (latent_dim=10, fps=40, kappa=15000)
"""

import os
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation" / "multiview_top_bot_experiments"
PILOT_PROJECT_DIR = EXPERIMENT_DIR / "kpms_multiview_pilot"
PILOT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PILOT_PROJECT_DIR / "config.yml"

def create_multiview_config():
    all_bodyparts = [
        "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
        "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
        "tail_base", "left_shoulder", "left_midside", "left_hip",
        "right_shoulder", "right_midside", "right_hip", "head_midpoint",
        "left_front_paw", "right_front_paw",
        "left_hind_paw", "right_hind_paw"
    ]

    use_bodyparts = [
        "nose", "head_midpoint", "neck", "mid_back",
        "mouse_center", "mid_backend", "tail_base",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_front_paw", "right_front_paw",
        "left_hind_paw", "right_hind_paw"
    ]

    skeleton = [
        ["nose", "head_midpoint"],
        ["head_midpoint", "neck"],
        ["neck", "mid_back"],
        ["mid_back", "mouse_center"],
        ["mouse_center", "mid_backend"],
        ["mid_backend", "tail_base"],
        ["neck", "left_shoulder"],
        ["neck", "right_shoulder"],
        ["tail_base", "left_hip"],
        ["tail_base", "right_hip"],
        # Biological limb connections: dorsal anchors to ventral paws
        ["left_shoulder", "left_front_paw"],
        ["right_shoulder", "right_front_paw"],
        ["left_hip", "left_hind_paw"],
        ["right_hip", "right_hind_paw"]
    ]

    config_dict = {
        "video_dir": str(PILOT_PROJECT_DIR / "videos"),
        "bodyparts": all_bodyparts,
        "use_bodyparts": use_bodyparts,
        "skeleton": skeleton,
        "anterior_bodyparts": ["tail_base"],
        "posterior_bodyparts": ["nose"],
        "fps": 40,
        "downsample_rate": 1,
        "latent_dim": 10,
        "conf_threshold": 0.5,
        "added_noise_level": 0.1,
        "PCA_fitting_num_frames": 20000,
        "whiten": True,
        "kappa": 15000,
        "num_ar_iters": 50,
        "num_pca_iters": 25,
        "verbose": True
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Created Multi-View KPMS config at: {CONFIG_PATH}")
    print(f"Total Bodyparts:     {len(all_bodyparts)}")
    print(f"Modeled Bodyparts:   {len(use_bodyparts)} (includes all 4 ventral paws)")
    print(f"Limb Skeleton Edges: {len(skeleton)}")

def main():
    print("=" * 70)
    print("INITIALIZING MULTI-VIEW KEYPOINT-MOSEQ PILOT PROJECT")
    print(f"Project Root: {PILOT_PROJECT_DIR}")
    print("=" * 70)
    create_multiview_config()
    print("=" * 70)

if __name__ == "__main__":
    main()
