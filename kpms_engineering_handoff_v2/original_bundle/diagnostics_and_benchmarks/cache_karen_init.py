import os
import sys

import jax.random as jr

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import keypoint_moseq as kpms
from diagnostics.fast_state_cache import save_init_state
from run_pipeline import get_sessions
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
CACHE_FILE = "/tmp/karen_slds_init_state.pkl"

def cache_init():
    print("=" * 70)
    print("   GENERATING CACHED INITIALIZATION FOR FULL KAREN DATASET")
    print(f"   Target: {CACHE_FILE}")
    print("=" * 70)

    sessions = get_sessions()
    print(f"Loaded {len(sessions)} valid sessions.")

    # 1. Derive paths and load raw keypoints
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam")
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, "deeplabcut")

    # 2. Presence masks and filtering
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}
    filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
        coordinates, confidences, all_masks, PROJECT_DIR,
        return_video_frame_indexes=True, downsample_rate=2,
        **kpms.load_config(PROJECT_DIR)
    )
    del coordinates, confidences, all_masks
    clamp_conf_nonneg(filtered_confs)

    # Remove sequences with < 5 frames
    filtered_coords = {k: v for k, v in filtered_coords.items() if len(v) >= 5}
    filtered_confs = {k: v for k, v in filtered_confs.items() if len(v) >= 5}
    video_frame_indexes = {k: v for k, v in video_frame_indexes.items() if k in filtered_coords}

    # 3. Format data
    data, metadata = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(PROJECT_DIR))

    # 4. Batched init
    pca = kpms.load_pca(PROJECT_DIR)
    init_config = kpms.load_config(PROJECT_DIR)
    rng_key = jr.PRNGKey(42)
    model = batched_init_model_by_video(
        data, pca, init_config,
        params_subset_size=5, batch_size=20, seed=rng_key
    )

    # 5. Update kappa for SLDS
    kappa = 60000.0 / 10.0
    model = kpms.update_hypparams(model, kappa=kappa)

    # 6. Save to disk
    save_init_state(model, data, metadata, cache_path=CACHE_FILE)
    print(f"\n[Success] Initial state cached to {CACHE_FILE}!")

if __name__ == "__main__":
    cache_init()
