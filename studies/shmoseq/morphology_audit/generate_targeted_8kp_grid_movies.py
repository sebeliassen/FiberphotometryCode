#!/usr/bin/env python3
"""
generate_targeted_8kp_grid_movies.py
=============================================================================
Brief, focused visual validation for 8KP Selected Candidate (amph_8kp_k150k_run_03).

Target Syllables for Visual Audit:
- Common Syllables (Top Cohort Usage):
  * Syllable 10 (Usage: 54.9%, Mean Size: 47.5 px)
  * Syllable 6  (Usage: 52.5%, Mean Size: 48.3 px)
  * Syllable 1  (Usage: 51.6%, Mean Size: 50.2 px)
  * Syllable 0  (Usage: 42.8%, Mean Size: 49.9 px)
- Extreme Large-Size Associated Syllables:
  * Syllable 41 (Mean Size: 54.8 px, Spread: 1.6 px)
  * Syllable 33 (Mean Size: 54.3 px, Spread: 2.8 px)
- Extreme Small-Size Associated Syllables:
  * Syllable 49 (Mean Size: 43.7 px, Spread: 2.1 px)
  * Syllable 43 (Mean Size: 44.2 px, Spread: 2.7 px)

Validation Objectives:
1. Inspect tracking quality (no jumping, flipping, or phantom keypoint noise).
2. Ground truth size-associated syllables: do large syllables correspond to
   upright postures / rearing / elongation, and small syllables to hunched / turning /
   grooming postures? Or are they identical behaviors split across large and small mice?
=============================================================================
"""

import os
import sys
from pathlib import Path

# Force CPU execution & headless operation to prevent GPU resource conflict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from scipy import stats
from vidio.read import OpenCVReader

# ==============================================================================
# FAST-SEEK OPENCVREADER MONKEYPATCH
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

# ==============================================================================
# TARGETED SYLLABLE SCOPING FILTER
# ==============================================================================
TARGET_SYLLABLES = [0, 1, 6, 10, 33, 41, 43, 49]
orig_get_syllable_instances = util.get_syllable_instances

def targeted_get_syllable_instances(*args, **kwargs):
    instances = orig_get_syllable_instances(*args, **kwargs)
    filtered = {k: v for k, v in instances.items() if k in TARGET_SYLLABLES}
    return filtered

viz.get_syllable_instances = targeted_get_syllable_instances
util.get_syllable_instances = targeted_get_syllable_instances

_orig_filter = util.filter_centroids_headings
_filtered_cache = {}

def fast_filter_centroids_headings(centroids, headings, filter_size=9):
    key = id(centroids)
    if key not in _filtered_cache:
        _filtered_cache[key] = _orig_filter(centroids, headings, filter_size=filter_size)
    return _filtered_cache[key]

util.filter_centroids_headings = fast_filter_centroids_headings
viz.filter_centroids_headings = fast_filter_centroids_headings

# ==============================================================================
# VIDEO SYNCHRONIZATION AUDIT HELPER
# ==============================================================================
def verify_grid_movie_sync(video_path, num_frames=5, grid_rows=3, grid_cols=4):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or w <= 0 or h <= 0:
        cap.release()
        return False, "Failed to read video dimensions"

    tile_h, tile_w = h // grid_rows, w // grid_cols
    sample_frames = np.linspace(max(5, int(total_frames * 0.1)),
                                min(total_frames - 5, int(total_frames * 0.9)),
                                num_frames, dtype=int)
    center_vals = []
    arena_vals = []

    for f_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for r in range(grid_rows):
            for c in range(grid_cols):
                tile = gray[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
                th, tw = tile.shape
                cy, cx = th // 2, tw // 2
                y_grid, x_grid = np.ogrid[:th, :tw]
                dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
                center_mask = dist_sq <= 15**2
                floor_mask = (dist_sq > 40**2) & (tile > 30)
                if np.sum(center_mask) > 0 and np.sum(floor_mask) > 100:
                    center_vals.append(np.mean(tile[center_mask]))
                    arena_vals.append(np.median(tile[floor_mask]))
    cap.release()

    if len(center_vals) < 5:
        return False, "Insufficient valid tiles"

    center_vals = np.array(center_vals)
    arena_vals = np.array(arena_vals)
    contrast = arena_vals - center_vals
    t_stat, p_val = stats.ttest_rel(arena_vals, center_vals)
    is_valid = (np.mean(contrast) > 20.0) and (p_val < 1e-4)
    report = f"Contrast: {np.mean(contrast):.1f} px (t={t_stat:.2f}, p={p_val:.2e})"
    return is_valid, report


def main():
    project_dir = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "amph_project_8kp"
    model_name = "amph_8kp_k150k_run_03"
    video_dir = REPO_ROOT / "src_oft" / "amph_vids_ssd"
    out_dir = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "morphology_audit" / "grid_movies_targeted_8kp"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 TARGETED VISUAL CHECK: 8KP SELECTED CANDIDATE (RUN 03)")
    print(f"   Project:    {project_dir}")
    print(f"   Model:      {model_name}")
    print(f"   Syllables:  {TARGET_SYLLABLES}")
    print(f"   Output Dir: {out_dir}")
    print("=" * 80, flush=True)

    # 1. Load config
    print("[1/5] Loading configuration...", flush=True)
    config = kpms.load_config(str(project_dir))
    bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
    config["bodyparts"] = bodyparts

    # 2. Load model checkpoint
    print(f"[2/5] Loading checkpoint for {model_name}...", flush=True)
    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)

    # 3. Load results
    print(f"[3/5] Loading results for {model_name}...", flush=True)
    results = kpms.load_results(str(project_dir), model_name)

    # 4. Estimate coordinates, centroids, headings
    print("[4/5] Estimating coordinates and centroids...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    # 5. Load VFI & verify scale
    vfi_path = project_dir / model_name / "video_frame_indexes.npz"
    with np.load(vfi_path) as z:
        raw_vfi = {k: z[k] for k in z.files}

    first_key = next(iter(raw_vfi.keys()))
    max_idx = raw_vfi[first_key].max()
    print(f"[INFO] First VFI sequence max index: {max_idx}")

    target_fps = int(config.get("fps", 40))
    if max_idx <= 230000:
        print("[INFO] 40Hz model units detected. Applying 2x upsampling for 80Hz camera videos...", flush=True)
        upsample_rate = 2
        upsampled_vfi = {}
        for k, v in raw_vfi.items():
            v_rep = np.repeat(v * upsample_rate, upsample_rate)
            offsets = np.tile(np.arange(upsample_rate), len(v))
            upsampled_vfi[k] = (v_rep + offsets).astype(int)
        video_frame_indexes = upsampled_vfi
        results = {k: {**v, "syllable": np.repeat(v["syllable"], upsample_rate)} for k, v in results.items()}
        coordinates = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in coordinates.items()}
        centroids = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in centroids.items()}
        headings = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in headings.items()}
        target_fps = target_fps * upsample_rate
    else:
        print("[INFO] VFI already contains native 80Hz camera frame indices.", flush=True)
        video_frame_indexes = raw_vfi

    # Map video paths
    ssd_vids = list(video_dir.glob("*.avi"))
    video_paths = {}
    for k in results.keys():
        stem = k.split("DLC_")[0]
        cand = video_dir / f"{stem}.avi"
        if cand.exists():
            video_paths[k] = str(cand)
        else:
            for v in ssd_vids:
                if v.stem == stem or v.stem in k:
                    video_paths[k] = str(v)
                    break

    common_keys = [k for k in results.keys() if k in video_paths and k in video_frame_indexes]
    print(f"[INFO] Mapped {len(common_keys)} / {len(results)} sessions to video files.", flush=True)
    results = {k: results[k] for k in common_keys}
    coordinates = {k: coordinates[k] for k in common_keys}
    centroids = {k: centroids[k] for k in common_keys}
    headings = {k: headings[k] for k in common_keys}
    video_frame_indexes = {k: video_frame_indexes[k] for k in common_keys}

    config_clean = config.copy()
    config_clean.pop("fps", None)
    config_clean.pop("overlay_keypoints", None)

    print("\n[5/5] Rendering targeted syllable grid movies (3x4 = 12 instances per movie)...", flush=True)
    kpms.generate_grid_movies(
        results=results,
        project_dir=str(project_dir),
        model_name=model_name,
        output_dir=str(out_dir),
        video_paths=video_paths,
        coordinates=coordinates,
        centroids=centroids,
        headings=headings,
        video_frame_indexes=video_frame_indexes,
        fps=target_fps,
        rows=3,
        cols=4,
        min_frequency=0.001,
        min_duration=3,
        overlay_keypoints=False,
        sampling_options={"mode": "random"},
        **config_clean
    )

    print("\n✓ Rendering completed. Running synchronization contrast audit...", flush=True)
    vids = sorted(list(out_dir.glob("syllable*.mp4")))
    for v in vids:
        is_val, rep = verify_grid_movie_sync(v, grid_rows=3, grid_cols=4)
        status = "PASS ✓" if is_val else "FAIL ✗"
        print(f"  [{status}] {v.name:20s} | {rep}")

    print("\n🎉 Targeted visual check movies ready in:")
    print(f"   {out_dir}")

if __name__ == "__main__":
    main()
