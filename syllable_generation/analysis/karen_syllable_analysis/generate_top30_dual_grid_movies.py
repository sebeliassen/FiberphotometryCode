#!/usr/bin/env python3
"""
generate_top30_dual_grid_movies.py

High-performance Single-Pass Dual Grid Movie Generator for Top-30 Karen Syllables.

Features:
- Single-pass video I/O: Opens source video files ONCE per instance and generates
  BOTH clean (no overlay) and keypoint/trajectory overlay grid movies simultaneously in memory.
- Skips already generated syllables automatically.
- Fast-seek OpenCVReader monkeypatch for strided/downsampled VFI.
- Writes to:
  - syllable_generation/analysis/karen_syllable_analysis/output/grid_movies_with_trajectories/
  - syllable_generation/analysis/karen_syllable_analysis/output/grid_movies_without_trajectories/
"""

import os
import sys
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import cv2
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates
import keypoint_moseq as kpms
from keypoint_moseq.viz import (
    overlay_keypoints_on_image,
    get_edges,
    write_video_clip,
    get_syllable_instances,
)
from tqdm import tqdm

# ==============================================================================
# 1. FAST-SEEK OPENCVREADER MONKEYPATCH
# ==============================================================================
from vidio.read import OpenCVReader
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
print("[INFO] Applied OpenCVReader fast-seek monkeypatch to vidio.read.", flush=True)

# ==============================================================================
# 2. PATHS & CONFIGURATION
# ==============================================================================
PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"
DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"

OUT_BASE = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output"
OUT_WITH_TRAJ = OUT_BASE / "grid_movies_with_trajectories"
OUT_WITHOUT_TRAJ = OUT_BASE / "grid_movies_without_trajectories"

OUT_WITH_TRAJ.mkdir(parents=True, exist_ok=True)
OUT_WITHOUT_TRAJ.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. LOAD DATA & MODEL METADATA
# ==============================================================================
print(f"[INFO] Loading checkpoint for {MODEL_NAME}...", flush=True)
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

print(f"[INFO] Loading config...", flush=True)
config = kpms.load_config(str(PROJECT_DIR))
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts

print(f"[INFO] Loading results.h5...", flush=True)
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

print(f"[INFO] Estimating coordinates, centroids, and headings...", flush=True)
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
centroids, headings = kpms.get_centroids_headings(coordinates, **config)

# Load VFI
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
print(f"[INFO] Loading VFI from {vfi_path}...", flush=True)
with np.load(vfi_path) as z:
    video_frame_indexes = {k: z[k] for k in z.files}

# Build video path map from sessions
print(f"[INFO] Loading session video paths from {DATA_PKL}...", flush=True)
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

print(f"[INFO] Mapped {len(video_paths)} / {len(results)} recordings to video files on disk.", flush=True)

# ==============================================================================
# 4. COMPUTE TOP 30 SYLLABLES BY N_BOUTS & FILTER OUT EXISTING
# ==============================================================================
from collections import Counter
bout_counts = Counter()
for k, v in results.items():
    if "syllable" in v:
        sylls = np.array(v["syllable"])
        diffs = np.diff(sylls, prepend=-999)
        bout_mask = (diffs != 0) & (sylls >= 0)
        for syl in sylls[bout_mask]:
            bout_counts[int(syl)] += 1

top30_sylls = [s for s, c in bout_counts.most_common(30)]
print(f"[INFO] Top 30 Syllables by n_bouts: {top30_sylls}", flush=True)

existing_traj = {int(p.stem.replace("syllable", "")) for p in OUT_WITH_TRAJ.glob("*.mp4")}
existing_clean = {int(p.stem.replace("syllable", "")) for p in OUT_WITHOUT_TRAJ.glob("*.mp4")}
already_rendered = existing_traj.intersection(existing_clean)

target_sylls = [s for s in top30_sylls if s not in already_rendered]
print(f"[INFO] Already rendered ({len(already_rendered)}): {sorted(list(already_rendered))}", flush=True)
print(f"[INFO] Target syllables to generate ({len(target_sylls)}): {target_sylls}", flush=True)

if not target_sylls:
    print("[INFO] All top 30 syllables are already rendered! Exiting.", flush=True)
    sys.exit(0)

# ==============================================================================
# 5. SINGLE-PASS DUAL TILE & DUAL GRID GENERATOR
# ==============================================================================
rows, cols = 4, 6
num_samples = rows * cols
fps = 40
pre_sec, post_sec = 0.5, 1.0
pre = int(pre_sec * fps)
post = int(post_sec * fps)

edges = get_edges(bodyparts, config.get("skeleton", []))
window_size = 144
print(f"[INFO] Using window size: {window_size} pixels", flush=True)

# Sample instances for target syllables
syllables_dict = {k: v["syllable"] for k, v in results.items() if "syllable" in v}
all_instances = get_syllable_instances(syllables_dict, pre=pre, post=post, min_duration=0, min_frequency=0, min_instances=1)

sampled_instances = {}
for syl in target_sylls:
    insts = all_instances.get(syl, [])
    if len(insts) == 0:
        continue
    if len(insts) < num_samples:
        insts = insts + [insts[0]] * (num_samples - len(insts))
    np.random.seed(0)
    idx = np.random.choice(len(insts), num_samples, replace=False)
    sampled_instances[syl] = [insts[i] for i in idx]

print(f"[INFO] Prepared instances for {len(sampled_instances)} syllables.", flush=True)

def _generate_dual_grid_movie(instances, syl_id):
    """
    Renders both clean and overlay grid movies in a SINGLE I/O PASS.
    """
    dot_color = (255, 255, 255)
    dot_radius = 4
    scaled_window_size = window_size

    tiles_clean = []
    tiles_overlay = []

    for key, start, end in instances:
        cs = centroids[key][start - pre : start + post]
        h, c = headings[key][start], cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
        syllable_coordinates = coordinates[key][start - pre : start + post].copy()

        frame_ixs = video_frame_indexes[key][start - pre : start + post]
        reader = OpenCVReader(video_paths[key])
        frames = [reader[ix] for ix in frame_ixs]
        reader.close()

        c_rot = r @ c - window_size // 2
        M = np.float32([[np.cos(h), np.sin(h), -c_rot[0]], [-np.sin(h), np.cos(h), -c_rot[1]]])

        t_clean = []
        t_overlay = []

        for ii, (raw_f, c_pt) in enumerate(zip(frames, cs)):
            # 1. Clean frame
            f_clean = cv2.warpAffine(raw_f.copy(), M, (window_size, window_size))
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                cv2.circle(f_clean, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            t_clean.append(f_clean)

            # 2. Overlay frame
            coords = syllable_coordinates[ii]
            f_ov = overlay_keypoints_on_image(raw_f.copy(), coords, edges=edges)
            f_ov = cv2.warpAffine(f_ov, M, (window_size, window_size))
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                cv2.circle(f_ov, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            t_overlay.append(f_ov)

        tiles_clean.append(np.stack(t_clean))
        tiles_overlay.append(np.stack(t_overlay))

    # Reshape and concatenate into 4x6 grid
    tc = np.stack(tiles_clean).reshape(rows, cols, post + pre, scaled_window_size, scaled_window_size, 3)
    frames_clean = np.concatenate(np.concatenate(tc, axis=2), axis=2)

    to = np.stack(tiles_overlay).reshape(rows, cols, post + pre, scaled_window_size, scaled_window_size, 3)
    frames_overlay = np.concatenate(np.concatenate(to, axis=2), axis=2)

    return frames_clean, frames_overlay

# ==============================================================================
# 6. RENDER ALL REMAINING TOP-30 SYLLABLES IN SINGLE PASS
# ==============================================================================
print("\n" + "="*80, flush=True)
print(f"🎬 RENDERING {len(sampled_instances)} MISSING TOP-30 SYLLABLES (SINGLE-PASS DUAL ENCODING)", flush=True)
print(f"   Outputs: {OUT_BASE}", flush=True)
print("="*80, flush=True)

for syl in tqdm(target_sylls, desc="Top-30 Syllables"):
    if syl not in sampled_instances:
        continue
    insts = sampled_instances[syl]
    clean_frames, overlay_frames = _generate_dual_grid_movie(insts, syl)

    clean_out = OUT_WITHOUT_TRAJ / f"syllable{syl}.mp4"
    traj_out = OUT_WITH_TRAJ / f"syllable{syl}.mp4"

    write_video_clip(clean_frames, str(clean_out), fps=fps)
    write_video_clip(overlay_frames, str(traj_out), fps=fps)
    print(f" Saved dual videos for Syllable {syl}: {clean_out.name} & {traj_out.name}", flush=True)

print("\n" + "="*80, flush=True)
print(" ALL TOP-30 SYLLABLES SUCCESSFULLY GENERATED IN A SINGLE PASS!", flush=True)
print("="*80, flush=True)
