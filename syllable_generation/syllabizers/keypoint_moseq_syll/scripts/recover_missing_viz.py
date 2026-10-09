import os

# --- GPU MEMORY OPTIMIZATION ---
# Force CPU usage for movie generation to keep GPU free for training/extractions
# This MUST be set before ANY jax/kpms imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pickle
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import keypoint_moseq as kpms
import numpy as np
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

# --- OpenCVReader Monkeypatch for Fast Seeking ---
# When VFI is downsampled (e.g. gaps of 1-3 frames), the default OpenCVReader.read()
# calls cap.set() which is excruciatingly slow. This monkeypatch intercepts the read
# and uses sequential cap.read() fast-forwarding for small gaps instead.
from vidio.read import OpenCVReader

_original_read = OpenCVReader.read

def _fast_read(self, framenum):
    if type(framenum) == slice:
        return _original_read(self, framenum)

    gap = framenum - self.fnum
    # If the requested frame is ahead by a small margin (e.g., skip 1-30 frames)
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret:
                break
        self.fnum = framenum

    return _original_read(self, framenum)

OpenCVReader.read = _fast_read
print("[INFO] Applied OpenCVReader fast-seek monkeypatch to vidio.read.")


# --- Constants ---
# Adjust root dir since we are in scripts/
# file: .../keypoint_moseq_syll/scripts/recover_missing_viz.py
# parent: .../scripts
# parent.parent: .../keypoint_moseq_syll
ROOT_DIR = Path(__file__).resolve().parent.parent
SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl"
VIDEO_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/amph_vids_ssd")
OUTPUT_FPS = 40

# Add project root to path for unpickling sessions
# ROOT_DIR: .../keypoint_moseq_syll
# parent: .../syllabizers
# parent.parent: .../syllable_generation
# parent.parent.parent: .../FiberphotometryCode
PROJECT_ROOT = ROOT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def ensure_vfi_exists(project_dir, model_name, sibling_model_name="double_flip_activity_k20k"):
    """
    Checks if video_frame_indexes.npz exists in the model folder.
    If not, attempts to copy it from a sibling model folder (e.g. k20k).
    """
    model_dir = project_dir / model_name
    vfi_path = model_dir / "video_frame_indexes.npz"

    if vfi_path.exists():
        print(f"[VFI] Found existing VFI at {vfi_path}")
        return True

    # Attempt copy
    sibling_vfi = project_dir / sibling_model_name / "video_frame_indexes.npz"
    if sibling_vfi.exists():
        print(f"[VFI] Missing locally. Copying from sibling: {sibling_vfi}")
        shutil.copy(sibling_vfi, vfi_path)
        return True

    # As a fallback for optimized_flip, try its own k20k
    if "optimized" in str(project_dir):
         sibling_vfi = project_dir / "optimized_flip_activity_k20k" / "video_frame_indexes.npz"
         if sibling_vfi.exists():
            print(f"[VFI] Missing locally. Copying from optimized sibling: {sibling_vfi}")
            shutil.copy(sibling_vfi, vfi_path)
            return True

    print(f"[ERROR] Could not find VFI in {vfi_path} or siblings.")
    return False

import argparse


def generate_missing(project_dir_str, model_name, do_plots=True, do_movies=True, test_mode=False, syllables_json_path=None):
    PROJECT_DIR = Path(project_dir_str)
    MODEL_DIR = PROJECT_DIR / model_name

    print("\n" + "="*60)
    print(f"Processing: {PROJECT_DIR.name} / {model_name}")
    print("="*60)

    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found: {MODEL_DIR}")
        return

    # 1. VFI Check & Copy
    if not ensure_vfi_exists(PROJECT_DIR, model_name):
        print("Skipping due to missing VFI.")
        return

    # 2. Completion Check (model_summary.json)
    if not (MODEL_DIR / "model_summary.json").exists():
        print(f"Skipping: {model_name} is unfinished (model_summary.json missing).")
        return

    # Load Model
    print("Loading checkpoint...")
    try:
        model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), model_name)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Loading config...")
    config = kpms.load_config(str(PROJECT_DIR))
    if 'use_bodyparts' in config:
        config['bodyparts'] = config['use_bodyparts']

    # Extract Results
    print("Extracting results...")
    results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name, save_results=False)

    # Estimate Coordinates
    print("Estimating coordinates...")
    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )
    coordinates = unbatch(Y_est, *metadata)

    # --- Trajectory Plots ---
    if do_plots:
        output_dir = MODEL_DIR / "trajectory_plots"
        if output_dir.exists() and (output_dir / "all_trajectories.gif").exists() and not test_mode:
            print("Trajectory Plots (all_trajectories.gif) already exist. Skipping.")
        else:
            print("Generating Trajectory Plots (all_trajectories.gif missing)...")
            output_dir.mkdir(exist_ok=True)

            # Use local dicts for calling to not mutate originals if running sequentially
            coords_to_use = coordinates
            results_to_use = results

            if test_mode:
                print("[TEST MODE] Limiting to 1 session for plots...")
                first_key = list(coordinates.keys())[0]
                coords_to_use = {first_key: coordinates[first_key]}
                # Results might not be needed for plots depending on kpms version, but passing subset
                if first_key in results:
                    results_to_use = {first_key: results[first_key]}

            kpms.generate_trajectory_plots(
                coordinates=coords_to_use,
                results=results_to_use,
                project_dir=str(PROJECT_DIR),
                output_dir=str(output_dir),
                min_frequency=0,
                min_duration=0,
                save_individually=True,
                save_gifs=True,
                save_mp4s=False,
                density_sample=False,
                **config
            )
            print(f"Plots saved to {output_dir}")
    else:
        print("Skipping Trajectory Plots (flag disabled).")

    # --- Grid Movies ---
    if do_movies:
        if (MODEL_DIR / "grid_movies").exists() and not test_mode and not syllables_json_path:
             print("Grid Movies already exist. Skipping.")
        else:
            print("Generating Grid Movies...")

            # Load VFI
            vfi_path = MODEL_DIR / "video_frame_indexes.npz"
            with np.load(vfi_path) as z:
                video_frame_indexes = {k: z[k] for k in z.files}

            centroids, headings = kpms.get_centroids_headings(coordinates, **config)

            # Load Sessions & Map Videos
            print(f"Loading sessions from {SESSIONS_PKL}...")
            with open(SESSIONS_PKL, 'rb') as f:
                sessions = pickle.load(f)

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
                        aid = s.fiber_refs_df.iloc[0]['animal_id']

                    if aid is not None:
                        stem_to_animal[stem] = str(aid)

            video_paths = {}
            for key in results.keys():
                for stem, original_path in stem_to_path.items():
                    if stem in key:
                        # Check SSD
                        ssd_path = VIDEO_DIR / f"{stem}.avi"
                        if ssd_path.exists():
                            video_paths[key] = str(ssd_path)
                        else:
                            video_paths[key] = str(original_path)
                        break

            if not video_paths:
                print("Error: No videos mapped. Skipping grid movies.")
                return

            # Import optimal flip list for specific filtering
            try:
                from syllable_generation.syllabizers.keypoint_moseq_syll.utils.optimal_flip_list import OPTIMAL_FLIP_IDS
                OPTIMAL_FLIP_STRS = [str(x) for x in OPTIMAL_FLIP_IDS]
            except ImportError:
                print("WARNING: Could not import OPTIMAL_FLIP_IDS._ Using default string filtering only.")
                OPTIMAL_FLIP_STRS = []

            # Filtering
            final_keys = []
            skipped_count = 0
            skipped_optimal_flip_count = 0

            for key in video_paths.keys():
                # Universal check for explicit "flipped" duplicates
                if "flipped" in key:
                    skipped_count += 1
                    continue

                # Check for sub-optimal flips in optimized_flip_project
                # The user explicitly wants to DROP any keys matching the 'flipped' animal IDs
                # from the base session list, leaving only the ~16 'KEEP' animals.
                if "optimized_flip" in model_name:
                    found_stem = next((s for s in stem_to_path if s in key), None)
                    if found_stem and found_stem in stem_to_animal:
                        aid = stem_to_animal[found_stem]
                        if aid in OPTIMAL_FLIP_STRS:
                            skipped_optimal_flip_count += 1
                            continue

                final_keys.append(key)

            print(f"[DEBUG] Filtered out {skipped_count} explicitly 'flipped' keys.")
            if "optimized_flip" in model_name:
                print(f"[DEBUG] Filtered out {skipped_optimal_flip_count} keys on optimal flip list.")

            if test_mode:
                print("[TEST MODE] Limiting to 1 session for grid movies...")
                final_keys = final_keys[:1]

            results_subset = {k: results[k] for k in final_keys}
            video_paths_subset = {k: video_paths[k] for k in final_keys}
            coordinates_subset = {k: coordinates[k] for k in final_keys}
            centroids_subset = {k: centroids[k] for k in final_keys}
            headings_subset = {k: headings[k] for k in final_keys}
            vfi_subset = {k: video_frame_indexes[k] for k in final_keys}

            # Optional filtering by explicit syllables json
            target_syllables = None
            if syllables_json_path:
                import json
                try:
                    with open(syllables_json_path, 'r') as f:
                        syll_map = json.load(f)
                        target_syllables = syll_map.get(model_name)
                except Exception as e:
                    print(f"Warning: Could not read syllable json {syllables_json_path}: {e}")

            # Monkeypatch routines if explicit syllables are required
            original_sample_instances = kpms.viz.sample_instances
            original_get_syllable_instances = kpms.viz.get_syllable_instances
            original_util_sample = kpms.util.sample_instances

            if target_syllables is not None:
                # 1. Bypass "min_instances" and "min_frequency" checks so rare syllables aren't dropped
                def _lenient_get_syllable_instances(syllables, pre, post, min_duration=0, min_frequency=0, min_instances=1):
                    return original_get_syllable_instances(
                        syllables, pre=pre, post=post, min_duration=min_duration,
                        min_frequency=0, min_instances=1
                    )

                # 2. Bypass "num_samples" assertion in util.sample_instances by padding instances
                #    Since viz imports it natively, we just replace viz.sample_instances with a wrapper
                def _filtering_and_lenient_sample_instances(syllable_instances, num_samples, *args, **kwargs):
                    # Pad
                    padded_instances = {}
                    for k, v in syllable_instances.items():
                        if len(v) == 0: continue
                        if len(v) < num_samples:
                            padded_instances[k] = v + [v[0]] * (num_samples - len(v))
                        else:
                            padded_instances[k] = v

                    # Sample (using original logic so it actually generates the required shape)
                    sampled = original_sample_instances(padded_instances, num_samples, *args, **kwargs)

                    # Filter
                    filtered = {k: v for k, v in sampled.items() if k in target_syllables}
                    print(f"Filtered grid movies down to {len(filtered)} target syllables.")
                    return filtered

                kpms.viz.get_syllable_instances = _lenient_get_syllable_instances
                kpms.viz.sample_instances = _filtering_and_lenient_sample_instances

            print(f"Generating movies for {len(final_keys)} session(s)...")

            # Remove conflicting config args
            if 'fps' in config: config.pop('fps')
            if 'overlay_keypoints' in config: config.pop('overlay_keypoints')

            try:
                kpms.generate_grid_movies(
                    results=results_subset,
                    project_dir=str(PROJECT_DIR),
                    model_name=model_name,
                    video_paths=video_paths_subset,
                    coordinates=coordinates_subset,
                    centroids=centroids_subset,
                    headings=headings_subset,
                    video_frame_indexes=vfi_subset,
                    fps=OUTPUT_FPS,
                    overlay_keypoints=False,
                    **config
                )
                print("Grid Movies generated.")
            finally:
                # Restore original functions
                kpms.viz.get_syllable_instances = original_get_syllable_instances
                kpms.util.sample_instances = original_util_sample
                kpms.viz.sample_instances = original_sample_instances
    else:
        print("Skipping Grid Movies (flag disabled).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate missing viz on CPU")
    parser.add_argument("--test", action="store_true", help="Run on 1 session only")
    parser.add_argument("--plots", action="store_true", help="Generate Trajectory Plots")
    parser.add_argument("--movies", action="store_true", help="Generate Grid Movies")
    parser.add_argument("--syllables_json", type=str, default=None, help="Path to JSON file specifying target syllables per model")

    args = parser.parse_args()

    # If no specific flags, do both (unless testing, then maybe best to still explicit?
    # Let's say if neither is set, do BOTH. If one is set, do that one.)
    if not args.plots and not args.movies:
        args.plots = True
        args.movies = True

    # 1. Flipped Double Project
    run_vals = [
        ("flipped_double_project", "double_flip_activity_k100k"),
        ("optimized_flip_project", "optimized_flip_activity_k100k")
    ]

    if args.test:
        print("Running in TEST MODE (first model only, 1 output each)")
        run_vals = run_vals[:1]

    for proj_name, model_name in run_vals:
        proj_dir = ROOT_DIR / proj_name

        # Default to local project's grid_syllables.json if none provided via CLI
        syll_path = args.syllables_json
        if syll_path is None:
            local_json = proj_dir / "grid_syllables.json"
            if local_json.exists():
                syll_path = str(local_json)
                print(f"[INFO] Auto-detected local syllable config: {syll_path}")

        generate_missing(str(proj_dir), model_name, do_plots=args.plots, do_movies=args.movies, test_mode=args.test, syllables_json_path=syll_path)

    print("\nAll tasks completed.")
