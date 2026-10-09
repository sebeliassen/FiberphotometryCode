import os

# --- GPU MEMORY OPTIMIZATION ---
# Force CPU usage for movie generation to keep GPU free for training/extractions
# This MUST be set before ANY jax/kpms imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import keypoint_moseq as kpms
import numpy as np
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

# --- Global Constants ---
# Video directory from config.yml or manual override
VIDEO_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/amph_vids_ssd")
OUTPUT_FPS = 40 # Config says 40 for new_fps
SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl"
DEBUG_ONLY = False # IF TRUE, PRINTS AND EXITS BEFORE GENERATION

# Add project root to path for unpickling sessions
sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")

def run_for_project(project_dir_str, model_name):
    PROJECT_DIR = Path(project_dir_str)
    MODEL_NAME = model_name

    print("="*60)
    print("Starting Grid Movie Generation for:")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Model:   {MODEL_NAME}")
    print("="*60)

    print(f"Checking for completion of {MODEL_NAME}...")
    summary_path = PROJECT_DIR / MODEL_NAME / "model_summary.json"
    if not summary_path.exists():
        print(f"SKIPPING: {MODEL_NAME} is not finished (no model_summary.json found).")
        return

    print(f"Loading checkpoint from {PROJECT_DIR / MODEL_NAME}")
    try:
        model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return

    print("Loading config...")
    config = kpms.load_config(str(PROJECT_DIR))
    if 'use_bodyparts' in config:
        config['bodyparts'] = config['use_bodyparts']

    print("Extracting results...")
    results = kpms.extract_results(model, metadata, str(PROJECT_DIR), MODEL_NAME, save_results=False)

    print("Estimating coordinates and centroids...")
    # Estimate Y from model states
    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids, headings = kpms.get_centroids_headings(coordinates, **config)

    # Load Video Frame Indexes to map frames to videos
    vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
    print(f"Loading VFI from {vfi_path}")
    with np.load(vfi_path) as z:
        video_frame_indexes = {k: z[k] for k in z.files}

    # Map parts to videos
    print(f"Loading sessions from {SESSIONS_PKL}...")
    try:
        with open(SESSIONS_PKL, 'rb') as f:
            sessions = pickle.load(f)
    except Exception as e:
        print(f"Error loading sessions: {e}")
        return

    print("Building video map from session metadata...")

    # 1. Build a lookup: { VideoStem -> OriginalPath } AND { VideoStem -> AnimalID }
    stem_to_path = {}
    stem_to_animal = {}

    for s in sessions:
        if hasattr(s, 'video_refs_df') and not s.video_refs_df.empty:
            original_path = s.video_refs_df.iloc[0]['path']
            stem = Path(original_path).stem
            stem_to_path[stem] = str(original_path)

            # Get Animal ID
            aid = None
            if hasattr(s, 'metadata') and 'animal_id' in s.metadata:
                aid = s.metadata['animal_id']
            elif hasattr(s, 'fiber_refs_df') and 'animal_id' in s.fiber_refs_df.columns:
                 # Fallback
                 aid = s.fiber_refs_df.iloc[0]['animal_id']

            if aid is not None:
                stem_to_animal[stem] = str(aid)

    print(f"Loaded {len(stem_to_path)} video references from pickles.")

    video_paths = {}
    mapped_count = 0

    # filter keys for NON-FLIPPED only
    all_keys = list(results.keys())
    print(f"[DEBUG] Total keys in results: {len(all_keys)}")

    # Import optimal flip list for specific filtering
    try:
        from syllable_generation.syllabizers.keypoint_moseq_syll.utils.optimal_flip_list import OPTIMAL_FLIP_IDS
        print(f"Loaded {len(OPTIMAL_FLIP_IDS)} optimal flip IDs.")
        # Ensure string comparison
        OPTIMAL_FLIP_STRS = [str(x) for x in OPTIMAL_FLIP_IDS]
    except ImportError:
        print("WARNING: Could not import OPTIMAL_FLIP_IDS. Using default string filtering only.")
        OPTIMAL_FLIP_IDS = []
        OPTIMAL_FLIP_STRS = []

    filtered_vfi_keys = []
    skipped_count = 0
    skipped_optimal_flip_count = 0

    for k in all_keys:
        # 1. Standard "flipped" string check (for double_flip project)
        if "flipped" in k:
            skipped_count += 1
            continue

        # 2. Specific ID check for "optimized_flip" project
        if "optimized_flip" in MODEL_NAME:
            # Must first match key to stem to find Animal ID
            found_stem = None
            for stem in stem_to_path:
                if stem in k:
                    found_stem = stem
                    break

            should_skip = False
            if found_stem and found_stem in stem_to_animal:
                aid = stem_to_animal[found_stem]
                if aid in OPTIMAL_FLIP_STRS:
                    should_skip = True
                    # print(f"[DEBUG] Skipping key {k[:30]}... for Animal {aid}")

            # If we couldn't map it, we might be conservative?
            # But let's assume if we can't map it, we keep it (or it will fail video mapping later)

            if should_skip:
                skipped_optimal_flip_count += 1
                continue

        filtered_vfi_keys.append(k)

    print(f"[DEBUG] Filtered out {skipped_count} explicitly 'flipped' keys.")
    if "optimized_flip" in MODEL_NAME:
        print(f"[DEBUG] Filtered out {skipped_optimal_flip_count} keys based on OPTIMAL_FLIP_IDS.")
    print(f"[DEBUG] Remaining keys: {len(filtered_vfi_keys)}")


    # 2. Iterate filtered Keys and match to Session Video Stems
    for key in filtered_vfi_keys:
        match_stem = None
        for stem, original_path in stem_to_path.items():
            if stem in key:
                match_stem = stem

                # Prioritize SSD path if it exists
                ssd_path = VIDEO_DIR / f"{stem}.avi"
                if ssd_path.exists():
                    video_paths[key] = str(ssd_path)
                else:
                    # Fallback to original path if not yet copied/missing on SSD
                    video_paths[key] = str(original_path)

                mapped_count += 1
                break

        if not match_stem:
             print(f"[DEBUG] Failed to map key using session lookup: {key}")

    print(f"Mapped {mapped_count} / {len(filtered_vfi_keys)} sessions.")

    if mapped_count == 0:
        print("ERROR: No videos mapped. Cannot generate grid movies.")
        return

    # Filter all dictionaries in memory to ONLY mapped and NON-FLIPPED keys
    final_keys = list(video_paths.keys())
    print(f"Final filtering of all input dicts for {len(final_keys)} sessions...")

    results = {k: results[k] for k in final_keys}
    video_paths = {k: video_paths[k] for k in final_keys}

    # Verification: print first 3 paths
    print("Verification of mapped video paths (First 3):")
    for fk in final_keys[:3]:
        print(f"  {fk[:20]}... -> {video_paths[fk]}")

    coordinates = {k: coordinates[k] for k in final_keys}
    centroids = {k: centroids[k] for k in final_keys}
    headings = {k: headings[k] for k in final_keys}
    video_frame_indexes = {k: video_frame_indexes[k] for k in final_keys}

    if DEBUG_ONLY:
        print("\n*** DEBUG_ONLY MODE: VERIFICATION COMPLETE. EXITING. ***")
        return

    # Remove 'fps' and 'overlay_keypoints' from config if present to avoid conflict
    if 'fps' in config:
        config.pop('fps')
    if 'overlay_keypoints' in config:
        config.pop('overlay_keypoints')

    print(f"[DEBUG] Final check: Passing {len(results)} sessions to generate_grid_movies (Overlay=False)")
    kpms.generate_grid_movies(
        results=results,
        project_dir=str(PROJECT_DIR),
        model_name=MODEL_NAME,
        video_paths=video_paths,
        coordinates=coordinates,
        centroids=centroids,
        headings=headings,
        video_frame_indexes=video_frame_indexes,
        fps=OUTPUT_FPS,
        overlay_keypoints=False, # User requested WITHOUT overlay this time
        **config
    )
    print("Done with project!")


if __name__ == "__main__":
    # Define the sequence of projects to run
    # 1. Flipped Double Project
    run_for_project(
        "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/flipped_double_project",
        "double_flip_activity_k100k"
    )

    # 2. Optimized Flip Project
    print("\n\n" + "-"*60 + "\nProceeding to next project...\n" + "-"*60 + "\n")

    run_for_project(
        "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project",
        "optimized_flip_activity_k100k"
    )

    print("\nAll projects completed successfully.")
