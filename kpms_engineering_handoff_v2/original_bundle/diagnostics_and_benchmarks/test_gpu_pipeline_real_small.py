import os
import sys
import time

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers/keypoint_moseq_syll")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, KPMS_DIR)

from kpms_pipeline import run_pipeline
from run_pipeline import get_sessions

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "test_gpu_small_model"

def main():
    print("=== Testing 100% GPU-Batched Pipeline on 3 Real Karen Sessions ===")
    sessions = get_sessions()[:3]
    print(f"Loaded {len(sessions)} test sessions: {sessions}")

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": "deeplabcut",
        "cam": "center_cam",
        "subroutine": "jonathan",
        "latent_dim": 10,
        "num_ar_iters": 0,
        "total_iters": 2,
        "pca_mode": "load",
        "model_name": TEST_MODEL_NAME,
        "init_kappa": 60000,
        "decrease_kappa_factor": 10,
        "mixed_map_iters": 24,
        "seed": 42,
        "downsample_rate": 2,
        "new_fps": 40,
        "use_batched_init": True,
        "use_memory_patch": True,
        "flipping_mode": None,
        "subset_method": "spread",
        "subset_fraction": 1.0,
        "save_every_n_iters": None,
    }

    try:
        t0 = time.time()
        results = run_pipeline(sessions, **pipeline_kwargs)
        total_time = time.time() - t0
        print(f"\n🎉 Test on 3 sessions succeeded in {total_time:.2f}s without errors!")
    finally:
        import shutil
        temp_dir = os.path.join(PROJECT_DIR, TEST_MODEL_NAME)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up {temp_dir}")

if __name__ == "__main__":
    main()
