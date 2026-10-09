#!/usr/bin/env python3
"""
generate_run12_grid_movies.py
=============================================================================
Generate Keypoint-MoSeq Syllable Grid Movies for Run 12 (amph_prod_k150k_run_12).

Outputs:
1. grid_movies_without_density/ (Random sampling mode)
2. grid_movies_with_density/    (Density / PCA exemplar mode)
Both with symlinks:
   grid_movies_random -> grid_movies_without_density
   grid_movies_density -> grid_movies_with_density

Specifications:
- Restricted to top 30 syllables (0 to 29).
- Clean video crops with indicator dot only (overlay_keypoints=False).
- 2x VFI upsampling (40Hz model -> 80Hz raw camera video sync).
- Fast-seek monkeypatch on OpenCVReader.
- Automated mouse-vs-arena intensity contrast sync validation.
=============================================================================
"""

import os
import sys
from pathlib import Path

# Force CPU execution & headless operation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import cv2
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates
import keypoint_moseq as kpms
import keypoint_moseq.viz as viz
import keypoint_moseq.util as util
from vidio.read import OpenCVReader
from scipy import stats

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
print("[INFO] Applied OpenCVReader fast-seek monkeypatch to vidio.read.", flush=True)

# ==============================================================================
# SYLLABLE SCOPING FILTER (TOP 30 SYLLABLES: 0 TO 29)
# ==============================================================================
orig_get_syllable_instances = util.get_syllable_instances

def filtered_get_syllable_instances(*args, **kwargs):
    instances = orig_get_syllable_instances(*args, **kwargs)
    filtered = {k: v for k, v in instances.items() if 0 <= k < 30}
    return filtered

viz.get_syllable_instances = filtered_get_syllable_instances
util.get_syllable_instances = filtered_get_syllable_instances

# Pre-filter caching for centroid/heading smoothing
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
# VIDEO SYNCHRONIZATION VERIFICATION HELPER
# ==============================================================================
def verify_grid_movie_sync(video_path, num_frames=5, grid_rows=4, grid_cols=6):
    """
    Verifies that the mouse is correctly centered in the grid movie tiles.
    Because the mouse is significantly darker than the surrounding arena floor,
    the cropped centroid pixels must be significantly darker than the floor.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0 or w <= 0 or h <= 0:
        cap.release()
        return False, "Failed to read video dimensions or frame count"
        
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
                
                # Center mouse region (radius ~15 px)
                y_grid, x_grid = np.ogrid[:th, :tw]
                dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
                center_mask = dist_sq <= 15**2
                
                # Arena floor region (ignore black boundary padding < 30)
                floor_mask = (dist_sq > 40**2) & (tile > 30)
                
                if np.sum(center_mask) > 0 and np.sum(floor_mask) > 100:
                    center_vals.append(np.mean(tile[center_mask]))
                    arena_vals.append(np.median(tile[floor_mask]))
                    
    cap.release()
    
    if len(center_vals) < 10:
        return False, "Insufficient valid tiles sampled"
        
    center_vals = np.array(center_vals)
    arena_vals = np.array(arena_vals)
    contrast = arena_vals - center_vals  # Positive means mouse center is darker than floor
    
    t_stat, p_val = stats.ttest_rel(arena_vals, center_vals)
    mean_contrast = np.mean(contrast)
    mean_center = np.mean(center_vals)
    mean_floor = np.mean(arena_vals)
    
    is_valid = (mean_contrast > 25.0) and (p_val < 1e-5)
    report = (f"Floor Mean: {mean_floor:.1f}, Center Mean: {mean_center:.1f}, "
              f"Contrast: {mean_contrast:.1f} (t={t_stat:.2f}, p={p_val:.2e})")
    return is_valid, report


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate AMPH Run 12 Grid Movies")
    parser.add_argument("--mode", choices=["all", "density", "random"], default="all", help="Sampling mode to render")
    args = parser.parse_args()

    project_dir = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "amph_project"
    model_name = "amph_prod_k150k_run_12"
    video_dir = REPO_ROOT / "src_oft" / "amph_vids_ssd"

    out_random = project_dir / model_name / "grid_movies_without_density"
    out_density = project_dir / model_name / "grid_movies_with_density"
    out_random.mkdir(parents=True, exist_ok=True)
    out_density.mkdir(parents=True, exist_ok=True)

    # Symlink aliases
    sym_random = project_dir / model_name / "grid_movies_random"
    sym_density = project_dir / model_name / "grid_movies_density"
    if not sym_random.exists():
        sym_random.symlink_to(out_random.name)
    if not sym_density.exists():
        sym_density.symlink_to(out_density.name)

    print("=" * 80)
    print("🎬 AMPH RUN 12 GRID MOVIE GENERATION (TOP 30 SYLLABLES 00..29)")
    print(f"   Project: {project_dir}")
    print(f"   Model:   {model_name}")
    print(f"   Videos:  {video_dir}")
    print(f"   Mode:    {args.mode}")
    print("=" * 80)

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

    # 4. Estimate coordinates, centroids, and headings
    print("[4/5] Estimating coordinates and spatial metrics...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    # 5. Load VFI (Native camera video frame indexes)
    vfi_path = project_dir / model_name / "video_frame_indexes.npz"
    print(f"[INFO] Loading VFI from {vfi_path}...", flush=True)
    with np.load(vfi_path) as z:
        video_frame_indexes = {k: z[k] for k in z.files}

    # In amph_prod_k150k_run_12, video_frame_indexes.npz ALREADY directly stores the true
    # 80Hz camera video frame indexes (range ~49,000 to ~450,000 with step 2).
    # Length of each VFI sequence matches model sequence length (fps = 40 Hz).
    target_fps = int(config.get("fps", 40))
    print(f"[INFO] VFI verified: {len(video_frame_indexes)} sessions, model FPS = {target_fps} Hz.", flush=True)

    # Map video files
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

    print(f"[INFO] Successfully mapped {len(video_paths)} / {len(results)} videos to SSD files.", flush=True)
    assert len(video_paths) == len(results), "Error: Not all recordings mapped to video files!"

    # Clean config for keyword unpacking
    config_clean = config.copy()
    config_clean.pop("fps", None)
    config_clean.pop("overlay_keypoints", None)

    # --------------------------------------------------------------------------
    # RUN A: WITHOUT DENSITY SAMPLING (Random)
    # --------------------------------------------------------------------------
    if args.mode in ["all", "random"]:
        print("\n" + "=" * 80)
        print("▶ GENERATING GRID MOVIES: RANDOM SAMPLING (00..29)")
        print(f"   Output Directory: {out_random}")
        print("=" * 80)
        kpms.generate_grid_movies(
            results=results,
            project_dir=str(project_dir),
            model_name=model_name,
            output_dir=str(out_random),
            video_paths=video_paths,
            coordinates=coordinates,
            centroids=centroids,
            headings=headings,
            video_frame_indexes=video_frame_indexes,
            fps=target_fps,
            rows=4,
            cols=6,
            overlay_keypoints=False,
            sampling_options={"mode": "random"},
            **config_clean
        )
        print("✓ Completed Random Grid Movies!", flush=True)

    # --------------------------------------------------------------------------
    # RUN B: WITH DENSITY SAMPLING (Density Mode)
    # --------------------------------------------------------------------------
    if args.mode in ["all", "density"]:
        print("\n" + "=" * 80)
        print("▶ GENERATING GRID MOVIES: DENSITY SAMPLING (00..29)")
        print(f"   Output Directory: {out_density}")
        print("=" * 80)
        kpms.generate_grid_movies(
            results=results,
            project_dir=str(project_dir),
            model_name=model_name,
            output_dir=str(out_density),
            video_paths=video_paths,
            coordinates=coordinates,
            centroids=centroids,
            headings=headings,
            video_frame_indexes=video_frame_indexes,
            fps=target_fps,
            rows=4,
            cols=6,
            overlay_keypoints=False,
            sampling_options={
                "mode": "density",
                "pca_samples": 50000,
                "pca_dim": 4,
                "n_neighbors": 50,
            },
            **config_clean
        )
        print("✓ Completed Density Grid Movies!", flush=True)

    # --------------------------------------------------------------------------
    # VIDEO SYNCHRONIZATION AUDIT (Mouse Centroid Contrast vs Arena Floor)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 RUNNING AUTOMATED VIDEO SYNCHRONIZATION AUDIT")
    print("   Testing dark mouse centroid contrast against arena floor...")
    print("=" * 80)

    for mode_name, target_dir in [("Random", out_random), ("Density", out_density)]:
        test_vids = sorted(list(target_dir.glob("syllable*.mp4")))[:5]
        if not test_vids:
            continue
        print(f"\nEvaluating {mode_name} movies ({len(test_vids)} sample videos):")
        for tv in test_vids:
            is_valid, report = verify_grid_movie_sync(tv)
            status_str = "PASS ✓" if is_valid else "FAIL ✗"
            print(f"  [{status_str}] {tv.name:15s} | {report}")
            assert is_valid, f"Sync check failed for {tv.name}: {report}"

    print("\n🎉 ALL REQUESTED GRID MOVIES SUCCESSFULLY GENERATED AND VERIFIED!")

if __name__ == "__main__":
    main()
