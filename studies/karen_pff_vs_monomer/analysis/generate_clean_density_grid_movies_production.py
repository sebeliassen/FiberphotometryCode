#!/usr/bin/env python3
"""
generate_clean_density_grid_movies_production.py

Production pipeline for clean density-sampled grid movies:
1. EXCLUDES all pilot sessions/animals (removes the top source of tracking/translation drift).
2. Uses ANATOMICAL FORWARD HEADING (anterior: nose, posterior: tail_base).
3. Uses TORSO-BALANCED CENTROID (midpoint between neck and hips).
4. Uses EXACT VFI LOOKUPS (video_frame_indexes.npz) for frame-accurate timing.
5. Employs ADAPTIVE DENSITY SAMPLING (canonical exemplar mode in trajectory space).
6. Renders 24-tile (4x6) clean grid movies at 144px per tile, 1.5s duration, in universal H.264.
7. Automatically packages all 100 movies into a clean master zip deliverable.
"""

import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

np.bool8 = np.bool_
import cv2
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import get_syllable_instances, write_video_clip
from vidio.read import OpenCVReader

# Fast-seek reader
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

PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"
DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output" / "density_grid_movies_without_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows, cols = 4, 6
num_samples = rows * cols  # 24 tiles
fps = 40
pre, post = 20, 40         # 60 frames = 1.50s
window_size = 144
dot_color = (255, 255, 255)
dot_radius = 4

print("=" * 80)
print("🚀 PRODUCTION RUN: DENSITY CLEAN GRID MOVIES (PILOT-EXCLUDED, FORWARD-ALIGNED)")
print(f"   Grid: {rows}x{cols} ({num_samples} tiles) | Window: {window_size}px | Duration: {pre+post} frames (1.5s)")
print(f"   Destination: {OUT_DIR}")
print("=" * 80)

# 1. Load model, config, results, and sessions
print("[1/6] Loading model checkpoint, config, and session metadata...", flush=True)
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

# Identify and filter out pilot recordings
pilot_keys = set()
for s in sessions:
    is_pilot = "pilot" in s.session_id.lower() or "pilot" in str(s.metadata.get("animal_id", "")).lower() or "pilot" in str(s.metadata.get("trial_id", "")).lower()
    if is_pilot:
        k = getattr(s, "results_dict_key", None)
        if k:
            pilot_keys.add(k)
        for rk in results.keys():
            if s.session_id in rk or rk in s.session_id:
                pilot_keys.add(rk)

for rk in results.keys():
    if "pilot" in rk.lower():
        pilot_keys.add(rk)

valid_keys = [k for k in results.keys() if k not in pilot_keys]
print(f"      Total recordings: {len(results)} | Pilot (EXCLUDED): {len(pilot_keys)} | High-Quality Non-Pilot: {len(valid_keys)}")

# 2. Reconstruct coordinates, forward heading, and balanced centroid
print("[2/6] Calculating anatomical forward heading and torso-balanced centroid...", flush=True)
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts

Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)

nose_idx = bodyparts.index("nose")
tail_idx = bodyparts.index("tail_base")
neck_idx = bodyparts.index("neck")
left_hip_idx = bodyparts.index("left_hip")
right_hip_idx = bodyparts.index("right_hip")

cfg_corr = config.copy()
cfg_corr["anterior_idxs"] = [nose_idx]
cfg_corr["posterior_idxs"] = [tail_idx]

_, headings_corr = kpms.get_centroids_headings(coordinates, **cfg_corr)

centroids_corr = {}
for k, coords in coordinates.items():
    torso_pts = coords[:, [neck_idx, left_hip_idx, right_hip_idx]]
    centroids_corr[k] = torso_pts.mean(axis=1)

# 3. Load VFI and Map Video Paths
print("[3/6] Mapping VFI and video files on disk...", flush=True)
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
with np.load(vfi_path) as z:
    video_frame_indexes = {k: z[k] for k in z.files}

stem_to_path = {}
for s in sessions:
    if hasattr(s, "video_refs_df") and s.video_refs_df is not None and len(s.video_refs_df) > 0:
        for _, row in s.video_refs_df.iterrows():
            p = row["path"]
            stem_to_path[Path(p).stem] = p
            stem_to_path[s.session_id] = p

video_paths = {}
for k in valid_keys:
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

# 4. Extract syllable instances from non-pilot recordings and perform density sampling
print("[4/6] Extracting syllable instances from non-pilot sessions and computing density exemplars...", flush=True)
syllables_clean = {k: results[k]["syllable"] for k in valid_keys if "syllable" in results[k]}
all_instances = get_syllable_instances(
    syllables_clean, pre=pre, post=post, min_duration=0, min_frequency=0, min_instances=1
)

trajectories_dict = {}
all_trajs_list = []
rng = np.random.default_rng(42)

for s in range(100):
    inst_s = all_instances.get(s, [])
    if len(inst_s) == 0:
        continue
    if len(inst_s) > 400:
        sub_idx = rng.choice(len(inst_s), 400, replace=False)
        sub_inst = [inst_s[i] for i in sub_idx]
    else:
        sub_inst = inst_s

    s_trajs = []
    for key, start, end in sub_inst:
        chunk = coordinates[key][start - pre : start + post]
        c = centroids_corr[key][start]
        h = headings_corr[key][start]
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

print(f"      Density-sampled {len(sampled_instances)} syllables using purely non-pilot bouts.", flush=True)

# 5. Render All 100 Grid Movies
print("\n[5/6] Rendering all 100 clean density grid movies (4x6, 144px, H.264)...", flush=True)
render_t0 = time.time()

for syl in tqdm(range(100), desc="Rendering Clean Movies", ncols=80):
    if syl not in sampled_instances:
        continue
    insts = sampled_instances[syl]

    tiles_clean = []
    for key, start, end in insts:
        cs = centroids_corr[key][start - pre : start + post]
        h, c = headings_corr[key][start], cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

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

    tc = np.stack(tiles_clean).reshape(rows, cols, post + pre, window_size, window_size, 3)
    clean_frames = np.concatenate(np.concatenate(tc, axis=2), axis=2)

    out_path = OUT_DIR / f"syllable{syl}.mp4"
    write_video_clip(clean_frames, str(out_path), fps=fps)

print(f"\n[INFO] Rendered {len(sampled_instances)} movies in {time.time()-render_t0:.2f}s.", flush=True)

# 6. Package Clean Zip Deliverable
print("\n[6/6] Packaging clean zip deliverable...", flush=True)
zip_path = OUT_DIR.parent / "karen_density_movies_clean.zip"

all_vids = sorted(list(OUT_DIR.glob("syllable*.mp4")), key=lambda p: int(p.stem.replace("syllable", "")))
print(f"      Zipping {len(all_vids)} clean density movies into {zip_path}...")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for v in tqdm(all_vids, desc="Zipping MP4s", ncols=80):
        zf.write(v, arcname=f"density_movies_clean/{v.name}")

zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"      Zip file created: {zip_path} ({zip_size_mb:.2f} MB)")

# Copy to network lab folder
net_dest_dir = Path("/mnt/neuropharm/common/Lab member folder/Sebastian/karen_analysis")
if net_dest_dir.exists():
    net_zip = net_dest_dir / "karen_density_movies_clean.zip"
    import shutil
    shutil.copy2(zip_path, net_zip)
    print(f"      ✅ Copied clean zip to network folder: {net_zip}")

print("\n" + "=" * 80)
print("🎉 COMPLETE! All 100 density clean movies regenerated and zipped!")
print("=" * 80)
