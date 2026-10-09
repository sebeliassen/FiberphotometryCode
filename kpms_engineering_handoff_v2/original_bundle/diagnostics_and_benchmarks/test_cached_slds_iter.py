import os
import sys
import time

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import keypoint_moseq as kpms
from diagnostics.fast_state_cache import load_init_state
from utils.resample_batching import apply_optimizations

CACHE_FILE = "/tmp/karen_slds_init_state.pkl"
PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "cached_test_model"

def main():
    print("=" * 80)
    print("   FAST INSTANT SLDS BENCHMARK VIA SERIALIZED CACHE")
    print("=" * 80)

    # 1. Apply optimizations
    apply_optimizations()

    # 2. Instant load from cache
    t_load0 = time.perf_counter()
    model, data, metadata = load_init_state(CACHE_FILE, put_on_device=True)
    t_load = time.perf_counter() - t_load0
    print(f"[1/2] Cache loaded in {t_load:.2f}s (Bypassed 3 minutes of preprocessing)!")

    # 3. Fit 1 SLDS Iteration
    print("\n[2/2] Running 1 Full SLDS Gibbs Iteration on 1,004 videos...")
    t_fit0 = time.perf_counter()
    model, model_name = kpms.fit_model(
        model, data, metadata, PROJECT_DIR,
        model_name=TEST_MODEL_NAME,
        ar_only=False,
        start_iter=0,
        num_iters=1,
        parallel_message_passing=False,
        save_every_n_iters=None
    )
    t_fit = time.perf_counter() - t_fit0

    # 4. Check for NaNs
    has_nans = False
    for k, v in model["states"].items():
        if hasattr(v, "shape"):
            nan_count = int(np.isnan(np.asarray(v)).sum())
            if nan_count > 0:
                has_nans = True
                print(f"  ❌ WARNING: {nan_count} NaNs in states/{k}")
            else:
                print(f"  ✅ states/{k}: clean (0 NaNs)")

    print("\n" + "=" * 80)
    print("                    INSTANT BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Cache Load Time          : {t_load:.2f} s")
    print(f"Full 1,004-Video SLDS Iter: {t_fit:.2f} s ({t_fit/60.0:.2f} min)")
    print(f"Numerical Health (NaNs)  : {'FAILED' if has_nans else 'PERFECT (0 NaNs)'}")
    print("=" * 80)

    # Cleanup temporary test folder
    temp_dir = os.path.join(PROJECT_DIR, TEST_MODEL_NAME)
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
