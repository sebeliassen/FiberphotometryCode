#!/usr/bin/env python3
"""
generate_density_grid_movies.py

Step 2: Generate density-sampled grid movies (both with and without trajectory overlay)
for key syllables:
- Regulated Syllables: 6 (bimodal) and 72 (unimodal)
- Representative Frequent Syllables: 38 (bimodal), 43 (bimodal), 58 (unimodal), 81 (unimodal)

Using sample_instances(mode='density', n_neighbors=50) to sample the canonical
exemplar and its closest neighborhood mode in trajectory space.
"""

import time
from pathlib import Path

import numpy as np

np.bool8 = np.bool_
import cv2
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import OpenCVReader, get_edges, overlay_keypoints_on_image
from tqdm import tqdm

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2")
MODEL_NAME = "final_karen_model_k240k"
OUT_BASE = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_syllable_analysis/output/density_sampled_grid_movies")
DIR_CLEAN = OUT_BASE / "movies_without_trajectories"
DIR_OVERLAY = OUT_BASE / "movies_with_trajectories"
DIR_CLEAN.mkdir(parents=True, exist_ok=True)
DIR_OVERLAY.mkdir(parents=True, exist_ok=True)

TARGET_SYLLABLES = [6, 72, 38, 43, 58, 81]

print("=" * 80)
print("🎬 STEP 2: DENSITY-SAMPLED CANONICAL GRID MOVIE GENERATION")
print(f"   Target Syllables: {TARGET_SYLLABLES}")
print(f"   Output Clean:    {DIR_CLEAN}")
print(f"   Output Overlay:  {DIR_OVERLAY}")
print("=" * 80)

# 1. Load config and results
print("[1/5] Loading model checkpoint, results, and video metadata...")
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

fps = config.get("fps", 40)
pre_frames = 5
post_frames = 15
pre_sec = pre_frames / fps
post_sec = post_frames / fps
window_size = 256

# 2. Extract continuous coordinates
print("[2/5] Reconstructing continuous coordinates from latent states...")
t0 = time.time()
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
print(f"      Extracted coordinates for {len(coordinates)} recordings in {time.time()-t0:.2f}s")

# 3. Filter centroids and headings for clean egocentric frame alignment
print("[3/5] Pre-filtering session centroids and headings (size=9)...")
t0 = time.time()
centroids = {k: v["centroid"] for k, v in results.items()}
headings = {k: v["heading"] for k, v in results.items()}
syllables = {k: v["syllable"] for k, v in results.items()}

filt_centroids, filt_headings = kpms.viz.filter_centroids_headings(centroids, headings, filter_size=9)
print(f"      Pre-filtered {len(filt_centroids)} recordings in {time.time()-t0:.2f}s")

# 4. Find matching video paths and frame indexes
video_dir = config.get("video_dir", "/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")
video_paths = kpms.viz.find_matching_videos(results.keys(), video_dir, as_dict=True)
video_frame_indexes = config.get("video_frame_indexes", None)
if video_frame_indexes is None:
    video_frame_indexes = {k: np.arange(len(v)) for k, v in syllables.items()}

edges = get_edges(config.get("use_bodyparts", []), config.get("skeleton", []))

# 5. Extract syllable instances and perform DENSITY SAMPLING
print("[4/5] Computing density-sampled exemplars in trajectory space (mode='density')...")
t0 = time.time()
instances = kpms.util.get_syllable_instances(
    syllables, pre=pre_frames, post=post_frames, min_duration=0, min_frequency=0, min_instances=9
)

target_instances = {s: instances[s] for s in TARGET_SYLLABLES if s in instances}

sampled_instances = kpms.viz.sample_instances(
    target_instances,
    num_samples=9,
    mode="density",
    pca_samples=10000,
    pca_dim=4,
    n_neighbors=50,
    coordinates=coordinates,
    pre=pre_frames,
    post=post_frames,
    centroids=filt_centroids,
    headings=filt_headings
)
print(f"      Successfully density-sampled {len(sampled_instances)} syllables in {time.time()-t0:.2f}s")

# Helper function to write high-quality MP4 using OpenCV
def write_video_clip(frames, output_path, fps=40):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

# 6. Render Dual 3x3 Grid Movies (Clean and Overlay simultaneously)
print("[5/5] Rendering 3x3 Density-Sampled Grid Movies...")

for s in tqdm(TARGET_SYLLABLES, desc="Rendering Density Movies", ncols=80):
    inst_list = sampled_instances[s]
    total_frames = pre_frames + post_frames

    # Pre-allocate frames for 3x3 grid: 3*256 = 768 x 768
    grid_clean_frames = [np.zeros((3 * window_size, 3 * window_size, 3), dtype=np.uint8) for _ in range(total_frames)]
    grid_overlay_frames = [np.zeros((3 * window_size, 3 * window_size, 3), dtype=np.uint8) for _ in range(total_frames)]

    for slot_idx, (key, start, end) in enumerate(inst_list):
        r_idx = slot_idx // 3
        c_idx = slot_idx % 3
        y1, y2 = r_idx * window_size, (r_idx + 1) * window_size
        x1, x2 = c_idx * window_size, (c_idx + 1) * window_size

        cs = filt_centroids[key][start - pre_frames : start + post_frames]
        h_0 = filt_headings[key][start]
        c_0 = cs[pre_frames]

        # Egocentric rotation & alignment matrix
        r_mat = np.float32([[np.cos(h_0), np.sin(h_0)], [-np.sin(h_0), np.cos(h_0)]])
        c_rot = r_mat @ c_0 - window_size // 2
        M = np.float32([
            [np.cos(h_0), np.sin(h_0), -c_rot[0]],
            [-np.sin(h_0), np.cos(h_0), -c_rot[1]]
        ])

        syll_coords = coordinates[key][start - pre_frames : start + post_frames].copy()
        frame_ixs = video_frame_indexes[key][start - pre_frames : start + post_frames]

        reader = OpenCVReader(video_paths[key])
        raw_clip = [reader[ix] for ix in frame_ixs]
        reader.close()

        for t_i, (raw_f, coords_i) in enumerate(zip(raw_clip, syll_coords)):
            # Clean crop
            f_clean = cv2.warpAffine(raw_f, M, (window_size, window_size))
            grid_clean_frames[t_i][y1:y2, x1:x2] = f_clean

            # Overlay crop
            f_ov = overlay_keypoints_on_image(raw_f.copy(), coords_i, edges=edges)
            f_ov = cv2.warpAffine(f_ov, M, (window_size, window_size))
            grid_overlay_frames[t_i][y1:y2, x1:x2] = f_ov

    out_clean_path = DIR_CLEAN / f"syllable{s}_density.mp4"
    out_overlay_path = DIR_OVERLAY / f"syllable{s}_density.mp4"

    write_video_clip(grid_clean_frames, out_clean_path, fps=fps)
    write_video_clip(grid_overlay_frames, out_overlay_path, fps=fps)

print("\n" + "=" * 80)
print("✅ STEP 2 COMPLETE: Density-Sampled Grid Movies Generated!")
for s in TARGET_SYLLABLES:
    print(f"  • Syllable {s:2d}: {DIR_CLEAN / f'syllable{s}_density.mp4'} & {DIR_OVERLAY / f'syllable{s}_density.mp4'}")
print("=" * 80)
