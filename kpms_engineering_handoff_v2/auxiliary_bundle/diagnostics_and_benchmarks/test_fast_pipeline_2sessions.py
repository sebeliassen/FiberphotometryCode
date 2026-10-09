import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import sys
import time

import numpy as np
import pandas as pd

np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation/syllabizers/keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

from kpms_pipeline import run_pipeline

SESSIONS_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")
PROJECT_DIR = os.path.join(KPMS_DIR, "demo_project")

def get_2_sessions():
    all_sessions = pd.read_pickle(SESSIONS_PKL)
    filtered = []
    for s in all_sessions:
        if hasattr(s, 'tracking_refs_df') and s.tracking_refs_df is not None and not s.tracking_refs_df.empty:
            h5_path = s.tracking_refs_df['path'].iloc[0]
            if h5_path and os.path.exists(h5_path):
                filtered.append(s)
                if len(filtered) == 2:
                    break
    print(f"[Setup] Loaded {len(filtered)} valid sessions for test pipeline run.")
    return filtered

def main():
    print("=" * 80)
    print("      FAST PRECISE PIPELINE VERIFICATION RUN (2 SESSIONS, 2 AR + 2 SLDS)")
    print("=" * 80)

    sessions = get_2_sessions()

    t0 = time.perf_counter()
    run_pipeline(
        sessions=sessions,
        project_dir=PROJECT_DIR,
        detector="deeplabcut",
        cam="center_cam",
        subroutine="jonathan",
        latent_dim=10,
        num_ar_iters=2,
        total_iters=4,  # 2 AR + 4 SLDS = 6 total iterations
        pca_mode="load_or_fit",
        model_name="test_fast_exact_run",
        init_kappa=60000,
        decrease_kappa_factor=10,
        downsample_rate=2,
        new_fps=40,
        use_batched_init=True,
        use_memory_patch=True,  # Activates the fastest precise modules
    )
    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Pipeline test run completed in {total_time:.2f} s!")
    print("=" * 80)

if __name__ == "__main__":
    main()
