import os
import sys
import time

import numpy as np

# Compatibility alias for numpy 2.x with bokeh/panel
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

from kpms_pipeline import run_pipeline
from run_pipeline import get_sessions

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "test_2_sessions_model"

def main():
    print("=" * 70)
    print("         2-SESSION REAL-DATA BENCHMARK (3 ITERATIONS)")
    print("=" * 70)

    sessions = get_sessions()[:2]
    print(f"Loaded {len(sessions)} real sessions.")

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": "deeplabcut",
        "cam": "center_cam",
        "subroutine": "jonathan",
        "latent_dim": 10,
        "num_ar_iters": 0,
        "total_iters": 3,
        "pca_mode": "load",
        "model_name": TEST_MODEL_NAME,
        "init_kappa": 60000,
        "decrease_kappa_factor": 10,
        "mixed_map_iters": 24,
        "seed": 42,
        "downsample_rate": 2,
        "kappa": 6000,
    }

    t0 = time.perf_counter()
    results = run_pipeline(sessions, **pipeline_kwargs)
    t_tot = time.perf_counter() - t0
    print("\n=======================================================================")
    print(f" Completed 3 iterations across 2 real sessions in {t_tot:.2f} seconds total!")
    print("=======================================================================")

if __name__ == "__main__":
    main()
