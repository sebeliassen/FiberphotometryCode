#!/usr/bin/env python3
"""
setup_multiview_project.py
--------------------------
Sets up the Keypoint-MoSeq project directory and config.yml
for 23-keypoint multi-view modeling.
"""

from pathlib import Path

import yaml

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# 23 Bodyparts
BODYPARTS = [
    # TopCam (17)
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint",
    # BotCam (6)
    "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw",
    "bot_nose_tip", "bot_tail_base"
]

SKELETON = [
    # Dorsal spine and head
    ["nose", "head_midpoint"],
    ["nose", "left_ear_tip"],
    ["nose", "right_ear_tip"],
    ["head_midpoint", "left_ear_tip"],
    ["head_midpoint", "right_ear_tip"],
    ["head_midpoint", "neck"],
    ["neck", "mid_back"],
    ["mid_back", "mouse_center"],
    ["mouse_center", "mid_backend"],
    ["mid_backend", "mid_backend2"],
    ["mid_backend2", "mid_backend3"],
    ["mid_backend3", "tail_base"],
    # Dorsal shoulders and hips
    ["neck", "left_shoulder"],
    ["left_shoulder", "left_midside"],
    ["left_midside", "left_hip"],
    ["neck", "right_shoulder"],
    ["right_shoulder", "right_midside"],
    ["right_midside", "right_hip"],
    ["tail_base", "right_hip"],
    ["tail_base", "left_hip"],
    # Ventral paw links to joints
    ["left_shoulder", "left_front_paw"],
    ["right_shoulder", "right_front_paw"],
    ["left_hip", "left_hind_paw"],
    ["right_hip", "right_hind_paw"],
    # Ventral anchors to dorsal anchors
    ["nose", "bot_nose_tip"],
    ["tail_base", "bot_tail_base"],
]

config = {
    "bodyparts": BODYPARTS,
    "use_bodyparts": BODYPARTS,
    "skeleton": SKELETON,
    "anterior_bodyparts": ["tail_base"],
    "posterior_bodyparts": ["nose"],
    "added_noise_level": 0.1,
    "PCA_fitting_num_frames": 10000,
    "conf_threshold": 0.5,
    "error_estimator": {
        "intercept": 0.8642441486445335,
        "slope": -0.5487375020951059
    },
    "obs_hypparams": {
        "nu_s": 5,
        "nu_sigma": 100000.0,
        "sigmasq_0": 0.1,
        "sigmasq_C": 0.1
    },
    "ar_hypparams": {
        "K_0_scale": 10.0,
        "S_0_scale": 0.01,
        "latent_dim": 10,
        "nlags": 3
    },
    "trans_hypparams": {
        "alpha": 5.7,
        "gamma": 1000.0,
        "kappa": 1000000.0,
        "num_states": 100
    },
    "cen_hypparams": {
        "sigmasq_loc": 1.277378255756416
    },
    "recording_name_suffix": "",
    "verbose": True,
    "conf_pseudocount": 0.001,
    "video_dir": "/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files",
    "keypoint_colormap": "autumn",
    "whiten": True,
    "fix_heading": False,
    "fps": 60,
    "outlier_scale_factor": 6.0
}

config_path = PROJECT_DIR / "config.yml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created multi-view KPMS config: {config_path}")
