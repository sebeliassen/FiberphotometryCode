#!/usr/bin/env python3
"""
generate_all_density_clean_grid_movies.py

Generates all 100 density-sampled clean grid movies (without trajectory overlay):
- Uses mode='density' via adaptive nearest-neighbor exemplar mode sampling
  (canonical modal realization of each syllable in trajectory space).
- Encoded using H.264 (yuv420p via imageio/ffmpeg) for 100% compatibility with
  the Antigravity IDE built-in video player and all web browsers.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
np.bool8 = np.bool_
import cv2
import imageio
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import jax.numpy as jnp

import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import OpenCVReader

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2")
MODEL_NAME = "final_karen_model_k240k"
OUT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_syllable_analysis/output/density_grid_movies_without_trajectories")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎬 GENERATING ALL 100 DENSITY-SAMPLED CLEAN GRID MOVIES (H.264 / yuv420p)")
print(f"   Destination: {OUT_DIR}")
print("=" * 80)

# 1. Load config, results, and checkpoint
print("[1/5] Loading model checkpoint and results...")
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

fps = config.get("fps", 40)
pre_frames = 5
post_frames = 15
total_frames = pre_frames + post_frames
window_size = 256

# 2. Extract continuous coordinates and pre-filter centroids/headings
print("[2/5] Reconstructing coordinates and pre-filtering centroids/headings (size=9)...")
t0 = time.time()
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)

centroids = {k: v["centroid"] for k, v in results.items()}
headings = {k: v["heading"] for k, v in results.items()}
syllables = {k: v["syllable"] for k, v in results.items()}

filt_centroids, filt_headings = kpms.viz.filter_centroids_headings(centroids, headings, filter_size=9)
print(f"      Reconstructed coordinates and filtered headings in {time.time()-t0:.2f}s")

# 3. Video paths and frame index mapping
video_dir = config.get("video_dir", "/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")
video_paths = kpms.viz.find_matching_videos(results.keys(), video_dir, as_dict=True)
video_frame_indexes = config.get("video_frame_indexes", None)
if video_frame_indexes is None:
    video_frame_indexes = {k: np.arange(len(v)) for k, v in syllables.items()}

# 4. Adaptive Density Sampling across all 100 syllables
print("[3/5] Extracting syllable instances and computing adaptive density exemplars...")
t0 = time.time()
instances = kpms.util.get_syllable_instances(
    syllables, pre=pre_frames, post=post_frames, min_duration=0, min_frequency=0, min_instances=1
)

# Extract instance trajectories for PCA density fitting
# Sample a subset of trajectories globally to train reference PCA
print("      Computing global trajectory embedding for density estimation...")
trajectories_dict = {}
all_trajs_list = []
rng = np.random.default_rng(42)

for s in range(100):
    inst_s = instances.get(s, [])
    if len(inst_s) == 0:
        continue
    # Subsample if very large to prevent OOM
    if len(inst_s) > 500:
        sub_idx = rng.choice(len(inst_s), 500, replace=False)
        sub_inst = [inst_s[i] for i in sub_idx]
    else:
        sub_inst = inst_s
        
    s_trajs = []
    for key, start, end in sub_inst:
        chunk = coordinates[key][start - pre_frames : start + post_frames]
        c = filt_centroids[key][start]
        h = filt_headings[key][start]
        s_trajs.append(kpms.util.np_io(kpms.util.inverse_rigid_transform)(chunk, c, h))
        
    trajectories_dict[s] = (sub_inst, np.array(s_trajs))
    all_trajs_list.extend(s_trajs)

all_X = np.array(all_trajs_list)
if len(all_X) > 20000:
    p_idx = rng.choice(len(all_X), 20000, replace=False)
    pca_train = all_X[p_idx]
else:
    pca_train = all_X

pca = PCA(n_components=4, random_state=42).fit(pca_train.reshape(pca_train.shape[0], -1))
all_X_pca = pca.transform(pca_train.reshape(pca_train.shape[0], -1))
all_nbrs = NearestNeighbors(n_neighbors=min(50, len(all_X_pca))).fit(all_X_pca)

sampled_instances = {}
for s in range(100):
    inst_s = instances.get(s, [])
    if len(inst_s) == 0:
        continue
        
    if len(inst_s) < 9:
        # Fewer than 9 instances: sample with replacement
        sub_idx = rng.choice(len(inst_s), 9, replace=True)
        sampled_instances[s] = [inst_s[i] for i in sub_idx]
        continue
        
    sub_inst, s_trajs = trajectories_dict[s]
    N = len(s_trajs)
    k_nn = min(50, N - 1)
    
    if k_nn < 9:
        sub_idx = rng.choice(len(inst_s), 9, replace=False)
        sampled_instances[s] = [inst_s[i] for i in sub_idx]
        continue
        
    X_pca = pca.transform(s_trajs.reshape(N, -1))
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    local_density = 1.0 / (distances.mean(1) + 1e-12)
    
    g_dist, _ = all_nbrs.kneighbors(X_pca)
    global_density = 1.0 / (g_dist.mean(1) + 1e-12)
    
    exemplar_idx = np.argmax(local_density / global_density)
    chosen_nbrs = indices[exemplar_idx]
    
    if len(chosen_nbrs) >= 9:
        ch_idx = rng.choice(chosen_nbrs, 9, replace=False)
    else:
        ch_idx = chosen_nbrs[:9]
        
    sampled_instances[s] = [sub_inst[i] for i in ch_idx]

print(f"      Adaptive density sampling completed for {len(sampled_instances)} syllables in {time.time()-t0:.2f}s")

# Helper function to write H.264 (yuv420p) video compatible with IDE
def write_h264_clip(frames, output_path, fps=40, quality=7):
    # frames: list of (H, W, 3) uint8 RGB arrays
    with imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=quality
    ) as writer:
        for f in frames:
            writer.append_data(f)

# 5. Batch Render All 100 Clean 3x3 Grid Movies
print("\n[4/5] Rendering all 100 clean density grid movies (768x768, H.264)...")
render_t0 = time.time()

for s in tqdm(range(100), desc="Rendering Clean Density Movies", ncols=80):
    if s not in sampled_instances:
        continue
        
    inst_list = sampled_instances[s]
    grid_clean_frames = [np.zeros((3 * window_size, 3 * window_size, 3), dtype=np.uint8) for _ in range(total_frames)]
    
    for slot_idx, (key, start, end) in enumerate(inst_list[:9]):
        r_idx = slot_idx // 3
        c_idx = slot_idx % 3
        y1, y2 = r_idx * window_size, (r_idx + 1) * window_size
        x1, x2 = c_idx * window_size, (c_idx + 1) * window_size
        
        cs = filt_centroids[key][start - pre_frames : start + post_frames]
        h_0 = filt_headings[key][start]
        c_0 = cs[pre_frames]
        
        r_mat = np.float32([[np.cos(h_0), np.sin(h_0)], [-np.sin(h_0), np.cos(h_0)]])
        c_rot = r_mat @ c_0 - window_size // 2
        M = np.float32([
            [np.cos(h_0), np.sin(h_0), -c_rot[0]],
            [-np.sin(h_0), np.cos(h_0), -c_rot[1]]
        ])
        
        frame_ixs = video_frame_indexes[key][start - pre_frames : start + post_frames]
        reader = OpenCVReader(video_paths[key])
        raw_clip = [reader[ix] for ix in frame_ixs]
        reader.close()
        
        for t_i, raw_f in enumerate(raw_clip):
            f_clean = cv2.warpAffine(raw_f, M, (window_size, window_size))
            grid_clean_frames[t_i][y1:y2, x1:x2] = f_clean
            
    out_path = OUT_DIR / f"syllable{s}.mp4"
    write_h264_clip(grid_clean_frames, out_path, fps=fps)

print("\n" + "=" * 80)
print(f"✅ ALL {len(sampled_instances)} CLEAN DENSITY GRID MOVIES SUCCESSFULLY GENERATED!")
print(f"   Time elapsed: {time.time()-render_t0:.2f}s")
print(f"   Location: {OUT_DIR}")
print("=" * 80)
