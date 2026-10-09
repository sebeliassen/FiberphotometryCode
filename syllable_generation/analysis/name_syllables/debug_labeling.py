import os
import keypoint_moseq as kpms
import pandas as pd
import numpy as np
import shutil
import jax.numpy as jnp
from jax_moseq.utils import unbatch

# --- CONFIGURATION (AMPH PROJECT) ---
base_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll"
project_dir = os.path.join(base_path, "amph_project")
model_name = "amph_prod_k120k_run_1 copy"
keypoint_dir = "/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_amph_workspace/h5_files"
format_type = "deeplabcut"

results_path = os.path.join(project_dir, model_name, "results.h5")
grid_movie_dir = os.path.join(project_dir, model_name, "grid_movies")

print(f"Targeting Project: {project_dir}")
print(f"Targeting Model:   {model_name}")

# 1. Cleanup for rigorous testing (optional, but good for verification)
if os.path.exists(grid_movie_dir):
    print(f"Cleaning up grid_movies at {grid_movie_dir}...")
    shutil.rmtree(grid_movie_dir)

# Load Verify Config
config = kpms.load_config(project_dir)
print("Config loaded.")

# 2. Extract Results (Ensure results.h5 is fresh/correct)
print("Loading checkpoint...")
checkpoint_tuple = kpms.load_checkpoint(project_dir, model_name)
model = checkpoint_tuple[0]
metadata = checkpoint_tuple[2]

if not os.path.exists(results_path):
    print("Extracting results from checkpoint...")
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    kpms.io.save_hdf5(results_path, results)
else:
    print("Loading existing results.h5...")
    results = kpms.load_results(project_dir, model_name)

# 3. Load Sync Data (Video Frame Indexes)
vfi_path = os.path.join(project_dir, model_name, "video_frame_indexes.npz")
video_frame_indexes = None
if os.path.exists(vfi_path):
    print(f"Loading video_frame_indexes from {vfi_path}...")
    vfi_data = np.load(vfi_path, allow_pickle=True)
    video_frame_indexes = {k: vfi_data[k] for k in vfi_data.files}
    
    # --- UPSAMPLING LOGIC (User Request) ---
    # The model likely runs at 30Hz, but videos are 60Hz.
    # Logic retrieved from verify_legacy_migration.py:
    # v_scaled = vfi * rate
    # v_rep = np.repeat(v_scaled, 2)
    # offsets = np.tile(np.arange(2), N)
    # v_final = v_rep + offsets
    
    # We detect rate based on config FPS vs Video FPS?
    # Or just assume 2.0 as per snippet.
    # Config says fps=60? The model usually downsamples by 2 if `use_bodyparts` logic implies it?
    # Or PCA fitting num frames implied 30Hz? 
    # Let's assume rate=2.0 for now as verified in migration script for this project.
    
    upsample_rate = 2
    print(f"Applying VFI Upsampling (Rate={upsample_rate})...")
    
    upsampled_vfi = {}
    for k, v in video_frame_indexes.items():
        # 1. Scale
        v_scaled = v * upsample_rate
        
        # 2. Repeat + Offset
        N = len(v_scaled)
        v_rep = np.repeat(v_scaled, upsample_rate)
        offsets = np.tile(np.arange(upsample_rate), N)
        v_final = (v_rep + offsets).astype(int)
        
        upsampled_vfi[k] = v_final
        
    video_frame_indexes = upsampled_vfi
    print(f"Upsampled {len(video_frame_indexes)} index arrays.")

else:
    print("CRITICAL WARNING: video_frame_indexes.npz missing!")

# --- 4. ADVANCED: Estimate Coordinates from Model (Sync Fix) ---
print("Estimating coordinates from model states (for perfect sync)...")
import jax.numpy as jnp
from jax_moseq.utils import unbatch

# Helper function from load_and_analyze_session_state.py
def map_syllable_parts_to_videos(video_frame_indexes, video_dir):
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_exts)]
    m = {}
    misses = 0
    for part_key in video_frame_indexes.keys():
        # Heuristic: base name before 'DLC_' or similar suffix
        base = part_key.split('DLC_')[0] 
        # Note: adjust split logic if keys differ
        # User code uses: base = part_key.split('DLC_')[0] 
        # or base = part_key.split('_part')[0]
        
        match = next((vf for vf in video_files if vf.startswith(base)), None)
        if match:
            m[part_key] = os.path.join(video_dir, match)
        else:
            misses += 1
            # print(f"No match for {part_key}")
            
    print(f"Mapped {len(m)} videos. Misses: {misses}")
    return m

try:
    from jax_moseq.models.keypoint_slds import estimate_coordinates
    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )
    coordinates = unbatch(Y_est, *metadata)
    print("Coordinates estimated from model.")
    
    print("Computing aligned centroids/headings...")
    cfg = dict(config)
    if 'use_bodyparts' in cfg:
        cfg['bodyparts'] = cfg['use_bodyparts']
        
    centroids, headings = kpms.get_centroids_headings(coordinates, **cfg)
    print("Centroids/Headings computed.")
    
except ImportError:
    print("Fallback to raw keypoints...")
    coordinates, _, _ = kpms.load_keypoints(filepath_pattern=keypoint_dir, format=format_type)
    centroids = None
    headings = None


# Video Mapping
print("Mapping video paths manually...")
video_dir_path = config.get('video_dir')
video_paths = map_syllable_parts_to_videos(video_frame_indexes, video_dir_path)

# --- 6. UPSAMPLE DATA (To match VFI) ---
# Since we upsampled VFI (N -> 2N), we must also upsample the corresponding data
# (results["syllable"], coordinates, centroids, headings)
print(f"Upsampling Data (x{upsample_rate}) to match VFI...")

# 1. Upsample Results (Syllables)
results_upsampled = {}
for k, v in results.items():
    if 'syllable' in v:
        # Repeat elements: [a, b] -> [a, a, b, b]
        syll_up = np.repeat(v['syllable'], upsample_rate)
        
        # We need to recreate the dict structure
        results_upsampled[k] = v.copy()
        results_upsampled[k]['syllable'] = syll_up
        
        # Also upsample latent state if present/needed? 
        # Grid movies mostly use 'syllable' to group frames.
    else:
        results_upsampled[k] = v

# 2. Upsample Coordinates / Centroids / Headings
# These are dictionaries of arrays (T, K, 2)
def upsample_array_dict(d, rate):
    out = {}
    for k, arr in d.items():
        # Repeat along time axis (0)
        out[k] = np.repeat(arr, rate, axis=0)
    return out

coordinates_upsampled = upsample_array_dict(coordinates, upsample_rate)
centroids_upsampled = upsample_array_dict(centroids, upsample_rate) # handles None gracefully?
headings_upsampled = upsample_array_dict(headings, upsample_rate)

results = results_upsampled
coordinates = coordinates_upsampled
centroids = centroids_upsampled
headings = headings_upsampled
print("Data upsampling complete.")

# 5. Generate Grid Movies
print(f"Generating Grid Movies (aligned + mapped + upsampled to {config['fps'] * upsample_rate} Hz)...")
try:
    kpms.generate_grid_movies(
        results,
        project_dir=project_dir, 
        model_name=model_name, 
        output_dir=grid_movie_dir,
        num_examples=10, 
        sampling_mode="random",
        plot_model_name=False,
        # IMPORTANT: Pass the new upsampled FPS (e.g. 80Hz)
        fps=config.get('fps', 40) * upsample_rate,
        
        # Explicit mapping instead of video_dir
        video_paths=video_paths,
        
        coordinates=coordinates,
        centroids=centroids,
        headings=headings,
        video_frame_indexes=video_frame_indexes,
        overlay_keypoints=False,
    )
    print("Grid movies generated successfully!")
except Exception as e:
    print(f"Grid generation failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug script complete.")
