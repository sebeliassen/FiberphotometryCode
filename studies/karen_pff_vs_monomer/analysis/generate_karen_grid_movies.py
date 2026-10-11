#!/usr/bin/env python3
"""
generate_karen_grid_movies.py

Generates Keypoint-MoSeq Grid Movies for the Karen Project (final_karen_model_k240k)
both WITH keypoint/trajectory overlays and WITHOUT keypoint overlays.

Features:
- Fast-seek OpenCVReader monkeypatch for downsampled VFI (stride ~2)
- Direct results.h5 loading (<1s startup)
- Targets relevant syllables (top nominal candidate hits, class comparisons, and core repertoire)
- Saves outputs to:
  - syllable_generation/analysis/karen_syllable_analysis/output/grid_movies_with_trajectories
  - syllable_generation/analysis/karen_syllable_analysis/output/grid_movies_without_trajectories
"""

import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

# Force CPU/threading safety for video encoding
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

# ==============================================================================
# 1. OPENCVREADER MONKEYPATCH (FAST SEEKING FOR STRIDED VFI)
# ==============================================================================
from vidio.read import OpenCVReader

_original_read = OpenCVReader.read

def _fast_read(self, framenum):
    if isinstance(framenum, slice):
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

# Target Syllables:
# - Primary k240k nominal hits: [6, 93, 36, 41]
# - Comparison candidate hits: [74, 68, 83, 40, 31, 14]
# - Core frequent syllables: [0, 1, 2, 5, 9, 22]
TARGET_SYLLABLES = [6, 93, 36, 41, 74, 68, 83, 40, 31, 14, 0, 1, 2, 5, 9, 22]
TARGET_SYLLABLES = sorted(list(set(TARGET_SYLLABLES)))

print(f"[INFO] Target Syllables to Render ({len(TARGET_SYLLABLES)}): {TARGET_SYLLABLES}", flush=True)

# ==============================================================================
# 3. LOAD MODEL RESULTS & POSE DATA
# ==============================================================================
print(f"[INFO] Loading checkpoint for {MODEL_NAME}...", flush=True)
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

print("[INFO] Loading config...", flush=True)
config = kpms.load_config(str(PROJECT_DIR))
if "use_bodyparts" in config:
    config["bodyparts"] = config["use_bodyparts"]

print("[INFO] Loading results.h5...", flush=True)
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

print("[INFO] Estimating coordinates, centroids, and headings...", flush=True)
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
# 4. FILTER INSTANCES FOR TARGET SYLLABLES
# ==============================================================================
original_sample_instances = kpms.viz.sample_instances
original_get_syllable_instances = kpms.viz.get_syllable_instances
original_util_sample = kpms.util.sample_instances

def _lenient_get_syllable_instances(syllables, pre, post, min_duration=0, min_frequency=0, min_instances=1):
    return original_get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=0, min_instances=1
    )

def _filtering_and_lenient_sample_instances(syllable_instances, num_samples, *args, **kwargs):
    padded_instances = {}
    for k, v in syllable_instances.items():
        if len(v) == 0:
            continue
        if len(v) < num_samples:
            padded_instances[k] = v + [v[0]] * (num_samples - len(v))
        else:
            padded_instances[k] = v

    sampled = original_sample_instances(padded_instances, num_samples, *args, **kwargs)
    filtered = {k: v for k, v in sampled.items() if k in TARGET_SYLLABLES}
    print(f"[INFO] Filtered down to {len(filtered)} target syllables for grid rendering.", flush=True)
    return filtered

kpms.viz.get_syllable_instances = _lenient_get_syllable_instances
kpms.viz.sample_instances = _filtering_and_lenient_sample_instances

# Clean config conflicting keys
cfg_clean = dict(config)
cfg_clean.pop("fps", None)
cfg_clean.pop("overlay_keypoints", None)

# ==============================================================================
# 5. GENERATE GRID MOVIES (WITH & WITHOUT TRAJECTORIES)
# ==============================================================================
OUTPUT_FPS = 40  # 40Hz output

# 1. Grid Movies WITH Keypoint/Trajectory Overlays
print("\n" + "="*80, flush=True)
print("🎬 GENERATING GRID MOVIES WITH TRAJECTORIES (overlay_keypoints=True)", flush=True)
print(f"   Output Directory: {OUT_WITH_TRAJ}", flush=True)
print("="*80, flush=True)

kpms.generate_grid_movies(
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(OUT_WITH_TRAJ),
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=OUTPUT_FPS,
    overlay_keypoints=True,
    rows=4,
    cols=6,
    pre=0.5,
    post=1.0,
    **cfg_clean
)
print(" Grid movies with trajectories completed.", flush=True)

# 2. Grid Movies WITHOUT Keypoint/Trajectory Overlays
print("\n" + "="*80, flush=True)
print("🎬 GENERATING GRID MOVIES WITHOUT TRAJECTORIES (overlay_keypoints=False)", flush=True)
print(f"   Output Directory: {OUT_WITHOUT_TRAJ}", flush=True)
print("="*80, flush=True)

kpms.generate_grid_movies(
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(OUT_WITHOUT_TRAJ),
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=OUTPUT_FPS,
    overlay_keypoints=False,
    rows=4,
    cols=6,
    pre=0.5,
    post=1.0,
    **cfg_clean
)
print(" Grid movies without trajectories completed.", flush=True)

# Restore original functions
kpms.viz.get_syllable_instances = original_get_syllable_instances
kpms.util.sample_instances = original_util_sample
kpms.viz.sample_instances = original_sample_instances

print("\n" + "="*80, flush=True)
print(" ALL GRID MOVIES SUCCESSFULLY GENERATED IN:", flush=True)
print(f"  - {OUT_WITH_TRAJ}", flush=True)
print(f"  - {OUT_WITHOUT_TRAJ}", flush=True)
print("="*80, flush=True)
