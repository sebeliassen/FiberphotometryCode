#!/usr/bin/env python3
"""
episode_pilot_config.py
=============================================================================
Configuration and Shared Constants for the seqNMF-Adjacent Probabilistic
Behavioral Episode Pilot.
=============================================================================
"""

import sys

import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric
from pathlib import Path

import numpy as np

# Paths
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/probabilistic_episode_pilot"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = EXP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = EXP_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_DIR = EXP_DIR / "movies"
MOVIES_DIR.mkdir(parents=True, exist_ok=True)

THUMBS_DIR = EXP_DIR / "thumbnails"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

KPMS_RESULTS_PATH = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/results.h5"
VFI_PATH = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/video_frame_indexes.npz"
SESSIONS_CSV = REPO_ROOT / "syllable_generation/projects/jonathan_openfield_amph/overrides/sessions.csv"
ANIMALS_CSV = REPO_ROOT / "syllable_generation/projects/jonathan_openfield_amph/overrides/animals.csv"

SEQNMF_RESULTS_PATH = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf/clustered_seqnmf_model_results.pkl"
CLUSTERED_DATASET_PATH = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf/clustered_sequence_dataset.pkl"

# Animal Train / Validation Cohort Split (10 Train / 5 Validation)
# Strict animal-level grouping: all sessions of an animal are grouped together
TRAIN_ANIMALS = ["728", "740", "1680", "1759", "1765", "1658", "1664", "1666", "1694", "1749"]
VAL_ANIMALS = ["1767", "1773", "1775", "1771", "1781"]

# Level 4 Meta-Syllable Definitions (14 states)
META_NAMES = [
    "Walk_Right",          # 0
    "Walk_Left",           # 1
    "Walk_Forward",        # 2
    "Sprint_Run",          # 3
    "Turn_Right",          # 4
    "Turn_Left",           # 5
    "Rear_Elevate",        # 6
    "Touchdown_Descent",   # 7
    "Sniff_Investigate",   # 8
    "Look_Orient",         # 9
    "Groom_Module",        # 10
    "Stationary_Rest",     # 11
    "Tracker_Artifact",    # 12
    "Tail_Unlabeled"       # 13
]

# Level 2 Conservative Synonym Mapping (93 states)
# Merges near-identical sub-variants: 11->1, 23->3, 38->19, 43->32, 49->32, 9->8, 52->27
LEVEL2_MAP = np.arange(100)
LEVEL2_MAP[11] = 1   # walk_right syn
LEVEL2_MAP[23] = 3   # walk_left syn
LEVEL2_MAP[38] = 19  # walk_fwd syn
LEVEL2_MAP[43] = 32  # rest syn
LEVEL2_MAP[49] = 32  # rest syn
LEVEL2_MAP[9] = 8    # sniff syn
LEVEL2_MAP[52] = 27  # core groom syn

# Three established behavioral themes on Level 2 clustering
THEMES = {
    "theme_A": {
        "theme_id": "theme_A",
        "title": "Theme A: Wall-Rear & Descent Locomotion",
        "target_seq_l2": [15, 14, 4, 3],
        "idealized_chain": "sniff up -> rear/lean -> back+turn L -> walk left",
        "seqnmf_motif_id": 7,
        "seqnmf_factor_idx": 6,
        "target_tokens": ["Sprint_Run", "Turn_Right", "Rear_Elevate", "Touchdown_Descent", "Sprint_Run"],
        "target_meta_ids": [3, 4, 6, 7, 3],
        "n_phases": 4,
        "max_dur_bouts": 12,
        "enrichment": "KI Mutant Enriched"
    },
    "theme_B": {
        "theme_id": "theme_B",
        "title": "Theme B: Sprint & Turning Redirection",
        "target_seq_l2": [2, 0, 13, 6],
        "idealized_chain": "turn right -> run right -> dash -> turn left",
        "seqnmf_motif_id": 4,
        "seqnmf_factor_idx": 3,
        "target_tokens": ["Tail_Unlabeled", "Sniff_Investigate", "Sprint_Run", "Turn_Left", "Sprint_Run"],
        "target_meta_ids": [13, 8, 3, 5, 3],
        "n_phases": 4,
        "max_dur_bouts": 12,
        "enrichment": "KI Mutant Enriched"
    },
    "theme_C": {
        "theme_id": "theme_C",
        "title": "Theme C: Olfactory Inspection & Rest",
        "target_seq_l2": [8, 32, 8, 32],
        "idealized_chain": "sniff -> rest -> sniff -> rest",
        "seqnmf_motif_id": 1,
        "seqnmf_factor_idx": 0,
        "target_tokens": ["Sniff_Investigate", "Stationary_Rest", "Sniff_Investigate", "Groom_Module", "Sniff_Investigate"],
        "target_meta_ids": [8, 11, 8, 10, 8],
        "n_phases": 4,
        "max_dur_bouts": 15,
        "enrichment": "WT Control Enriched"
    }
}

# Video generation settings
VIDEO_FPS = 40
CANONICAL_RIG_CHAMBERS = ["top_left", "bottom_left", "bottom_right", "top_right"]
CH_TO_IDX = {"top_left": 0, "bottom_left": 1, "bottom_right": 2, "top_right": 3}
