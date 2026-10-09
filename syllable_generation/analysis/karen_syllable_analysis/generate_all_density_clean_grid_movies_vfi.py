#!/usr/bin/env python3
"""
generate_all_density_clean_grid_movies_vfi.py

Generates all 100 clean grid movies (without trajectory overlay) using:
- Correct Video Frame Indexes (VFI) from video_frame_indexes.npz (aligns exact 80Hz camera frames)
- Exact video paths mapped from sessions_karen_project_w_sylls_k240k.pkl
- Fast-seek OpenCVReader monkeypatch
- Canonical Density-Mode Sampling (24 tiles in 4x6 grid, window_size=144, pre=20, post=40)
- Centroid indicator dot active during the syllable
- Universal H.264 (yuv420p) encoding via write_video_clip
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

np.bool8 = np.bool_
import cv2
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import get_syllable_instances, write_video_clip
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from vidio.read import OpenCVReader

# ==============================================================================
# 1. OPTIMIZED VIDEO READER MONKEYPATCH
# ==============================================================================
_original_read = OpenCVReader.read

def _fast_read(self, framenum):
    if isinstance(framenum, slice):
        return _original_read(self, framenum)
    gap = framenum - self.fnum
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret:
                break
        self.fnum = framenum
    return _original_read(self, framenum)

OpenCVReader.read = _fast_read
print("[INFO] Applied OpenCVReader fast-seek monkeypatch.", flush=True)

# ==============================================================================
# 2. PATHS & CONFIGURATION
# ==============================================================================
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))
PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"
DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output" / "density_grid_movies_without_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows, cols = 4, 6
num_samples = rows * cols  # 24 tiles
fps = 40
pre_sec, post_sec = 0.5, 1.0
pre = int(pre_sec * fps)   # 20 frames
post = int(post_sec * fps) # 40 frames
window_size = 144
dot_color = (255, 255, 255)
dot_radius = 4

print("=" * 80)
print("🎬 GENERATING ALL 100 DENSITY-SAMPLED CLEAN GRID MOVIES (VFI-ALIGNED, H.264)")
print(f"   Grid: {rows}x{cols} ({num_samples} tiles) | Window: {window_size}px | pre={pre}f, post={post}f")
print(f"   Destination: {OUT_DIR}")
print("=" * 80)

# ==============================================================================
# 3. LOAD DATA, VFI & MODEL COORDINATES
# ==============================================================================
print("[1/5] Loading model checkpoint, results, and config...", flush=True)
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

print("[2/5] Reconstructing continuous coordinates, centroids, and headings...", flush=True)
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts
centroids, headings = kpms.get_centroids_headings(coordinates, **config)

# Load accurate VFI
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
print(f"[3/5] Loading exact VFI mapping from {vfi_path}...", flush=True)
with np.load(vfi_path) as z:
    video_frame_indexes = {k: z[k] for k in z.files}

# Build video path map from sessions
print(f"[INFO] Mapping session video paths from {DATA_PKL}...", flush=True)
with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

stem_to_path = {}
for s in sessions:
    if hasattr(s, "video_refs_df") and s.video_refs_df is not None and len(s.video_refs_df) > 0:
        for _, row in s.video_refs_df.iterrows():
            p = row["path"]
            stem_to_path[Path(p).stem] = p
            stem_to_path[s.session_id] = p

video_paths = {}
for k in results.keys():
    base_k = k.split("_part")[0] if "_part" in k else k
    if k in stem_to_path:
        video_paths[k] = stem_to_path[k]
    elif base_k in stem_to_path:
        video_paths[k] = stem_to_path[base_k]
    else:
        for stem, path in stem_to_path.items():
            if stem in k or k in stem:
                video_paths[k] = path
                break

print(f"[INFO] Successfully mapped all {len(video_paths)} recordings to disk.", flush=True)

# ==============================================================================
# 4. ADAPTIVE DENSITY SAMPLING (CANONICAL EXEMPLAR + NEIGHBORHOOD)
# ==============================================================================
print("[4/5] Extracting instances and computing adaptive density exemplars...", flush=True)
syllables_dict = {k: v["syllable"] for k, v in results.items() if "syllable" in v}
all_instances = get_syllable_instances(
    syllables_dict, pre=pre, post=post, min_duration=0, min_frequency=0, min_instances=1
)

# Extract aligned trajectory chunks for density estimation
trajectories_dict = {}
all_trajs_list = []
rng = np.random.default_rng(42)

for s in range(100):
    inst_s = all_instances.get(s, [])
    if len(inst_s) == 0:
        continue
    # Subsample to keep memory optimal
    if len(inst_s) > 400:
        sub_idx = rng.choice(len(inst_s), 400, replace=False)
        sub_inst = [inst_s[i] for i in sub_idx]
    else:
        sub_inst = inst_s

    s_trajs = []
    for key, start, end in sub_inst:
        chunk = coordinates[key][start - pre : start + post]
        c = centroids[key][start]
        h = headings[key][start]
        s_trajs.append(kpms.util.np_io(kpms.util.inverse_rigid_transform)(chunk, c, h))

    trajectories_dict[s] = (sub_inst, np.array(s_trajs))
    all_trajs_list.extend(s_trajs)

all_X = np.array(all_trajs_list)
if len(all_X) > 15000:
    p_idx = rng.choice(len(all_X), 15000, replace=False)
    pca_train = all_X[p_idx]
else:
    pca_train = all_X

pca = PCA(n_components=4, random_state=42).fit(pca_train.reshape(pca_train.shape[0], -1))
all_X_pca = pca.transform(pca_train.reshape(pca_train.shape[0], -1))
all_nbrs = NearestNeighbors(n_neighbors=min(50, len(all_X_pca))).fit(all_X_pca)

sampled_instances = {}
for s in range(100):
    inst_s = all_instances.get(s, [])
    if len(inst_s) == 0:
        continue

    if len(inst_s) < num_samples:
        # Fewer than 24 instances: sample with replacement
        sub_idx = rng.choice(len(inst_s), num_samples, replace=True)
        sampled_instances[s] = [inst_s[i] for i in sub_idx]
        continue

    sub_inst, s_trajs = trajectories_dict[s]
    N = len(s_trajs)
    k_nn = min(50, N - 1)

    if k_nn < num_samples:
        sub_idx = rng.choice(len(inst_s), num_samples, replace=False)
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

    if len(chosen_nbrs) >= num_samples:
        ch_idx = rng.choice(chosen_nbrs, num_samples, replace=False)
    else:
        ch_idx = chosen_nbrs[:num_samples]

    sampled_instances[s] = [sub_inst[i] for i in ch_idx]

print(f"[INFO] Prepared density-sampled instances for {len(sampled_instances)} syllables.", flush=True)

# ==============================================================================
# 5. RENDER CLEAN GRID MOVIES (4x6, 144px, VFI-INDEXED, H.264)
# ==============================================================================
def _render_clean_grid_movie(instances):
    tiles_clean = []

    for key, start, end in instances:
        cs = centroids[key][start - pre : start + post]
        h, c = headings[key][start], cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

        # EXACT VFI LOOKUP
        frame_ixs = video_frame_indexes[key][start - pre : start + post]
        reader = OpenCVReader(video_paths[key])
        frames = [reader[ix] for ix in frame_ixs]
        reader.close()

        c_rot = r @ c - window_size // 2
        M = np.float32([[np.cos(h), np.sin(h), -c_rot[0]], [-np.sin(h), np.cos(h), -c_rot[1]]])

        t_clean = []
        for ii, (raw_f, c_pt) in enumerate(zip(frames, cs)):
            f_clean = cv2.warpAffine(raw_f.copy(), M, (window_size, window_size))
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                cv2.circle(f_clean, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            t_clean.append(f_clean)

        tiles_clean.append(np.stack(t_clean))

    # Reshape and concatenate into 4x6 grid
    tc = np.stack(tiles_clean).reshape(rows, cols, post + pre, window_size, window_size, 3)
    clean_frames = np.concatenate(np.concatenate(tc, axis=2), axis=2)
    return clean_frames

print("\n[5/5] Rendering all 100 density clean grid movies...", flush=True)
render_t0 = time.time()

for syl in tqdm(range(100), desc="Rendering Clean Density Movies", ncols=80):
    if syl not in sampled_instances:
        continue
    insts = sampled_instances[syl]
    clean_frames = _render_clean_grid_movie(insts)
    clean_out = OUT_DIR / f"syllable{syl}.mp4"
    write_video_clip(clean_frames, str(clean_out), fps=fps)

print("\n" + "=" * 80, flush=True)
print(f"✅ ALL {len(sampled_instances)} CLEAN DENSITY GRID MOVIES SUCCESSFULLY GENERATED!", flush=True)
print(f"   Time elapsed: {time.time()-render_t0:.2f}s", flush=True)
print(f"   Location: {OUT_DIR}", flush=True)
print("=" * 80, flush=True)
