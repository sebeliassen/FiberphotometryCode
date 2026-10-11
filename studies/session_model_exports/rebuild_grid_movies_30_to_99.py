#!/usr/bin/env python3
"""
rebuild_grid_movies_30_to_99.py
=============================================================================
Re-renders Grid Movies for Syllables 30 through 99 (Run 12) with the exact
same 3.00-second, high-resolution format as Syllables 0 through 29:

- pre = 1.0 s (40 frames @ 40 fps)
- post = 2.0 s (80 frames @ 40 fps)
- Total duration: 3.00 s (120 frames)
- Grid layout: 4 rows x 6 cols (24 instances)
- max_video_size: 1920 (matching Syllables 0-29 resolution)
- Overlay: Centroid indicator dot only (overlay_keypoints=False)
- Strict Protection: Syllables 0 to 29 are NEVER touched or overwritten!

Outputs:
1. grid_movies_without_density/ (Random sampling mode)
2. grid_movies_with_density/    (Density exemplar mode)
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

import cv2
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
import tqdm
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
print("[INFO] Applied OpenCVReader fast-seek monkeypatch to vidio.read.", flush=True)

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
# ROBUST SAMPLER (Supports tail syllables with < 24 instances)
# ==============================================================================
def robust_sample_instances(
    syllable_instances,
    num_samples=24,
    mode="random",
    pca_samples=50000,
    pca_dim=4,
    n_neighbors=50,
    coordinates=None,
    pre=40,
    post=80,
    centroids=None,
    headings=None,
    filter_size=9,
):
    assert mode in ["random", "density"]

    if mode == "random":
        sampled = {}
        for s, inst in syllable_instances.items():
            if len(inst) == 0:
                continue
            if len(inst) >= num_samples:
                idx = np.random.choice(len(inst), num_samples, replace=False)
            else:
                idx = np.random.choice(len(inst), num_samples, replace=True)
            sampled[s] = [inst[i] for i in idx]
        return sampled

    elif mode == "density":
        assert not (coordinates is None or headings is None or centroids is None)

        clean_coords = {}
        for key in coordinates.keys():
            arr = coordinates[key].copy()
            outliers = np.isnan(arr).any(-1)
            clean_coords[key] = util.interpolate_keypoints(arr, outliers)

        trajectories = {}
        for s, inst in syllable_instances.items():
            if len(inst) == 0:
                continue
            trajectories[s] = util.get_instance_trajectories(
                inst,
                clean_coords,
                pre=pre,
                post=post,
                centroids=centroids,
                headings=headings,
                filter_size=filter_size,
            )

        if not trajectories:
            return {}

        X_all = np.vstack(list(trajectories.values()))
        if X_all.shape[0] > pca_samples:
            X_all = X_all[np.random.choice(X_all.shape[0], pca_samples, replace=False)]

        pca = PCA(n_components=min(pca_dim, X_all.shape[0], X_all.shape[1])).fit(X_all.reshape(X_all.shape[0], -1))
        Xpca_all = pca.transform(X_all.reshape(X_all.shape[0], -1))

        eff_all_k = min(n_neighbors, len(Xpca_all) - 1) if len(Xpca_all) > 1 else 1
        all_nbrs = NearestNeighbors(n_neighbors=eff_all_k, algorithm="kd_tree").fit(Xpca_all)

        sampled = {}
        for s, X_syll in trajectories.items():
            inst = syllable_instances[s]
            n_inst = len(inst)
            if n_inst == 0:
                continue

            if n_inst < num_samples:
                # Rare syllable: tile with replacement
                idx = np.random.choice(n_inst, num_samples, replace=True)
                sampled[s] = [inst[i] for i in idx]
                continue

            Xpca = pca.transform(X_syll.reshape(X_syll.shape[0], -1))
            eff_k = min(n_neighbors, len(Xpca) - 1)
            if eff_k < 1:
                idx = np.random.choice(n_inst, num_samples, replace=True)
                sampled[s] = [inst[i] for i in idx]
                continue

            nbrs = NearestNeighbors(n_neighbors=eff_k, algorithm="kd_tree").fit(Xpca)
            distances, indices = nbrs.kneighbors(Xpca)
            local_density = 1.0 / (distances.mean(1) + 1e-6)

            distances_all, _ = all_nbrs.kneighbors(Xpca)
            global_density = 1.0 / (distances_all.mean(1) + 1e-6)
            exemplar = np.argmax(local_density / (global_density + 1e-6))

            nbr_pool = indices[exemplar]
            if len(nbr_pool) >= num_samples:
                sample_idx = np.random.choice(nbr_pool, num_samples, replace=False)
            else:
                sample_idx = np.random.choice(nbr_pool, num_samples, replace=True)

            sampled[s] = [inst[i] for i in sample_idx]

        return sampled


def main():
    project_dir = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    model_name = "amph_prod_k150k_run_12"
    video_dir = REPO_ROOT / "src_oft/amph_vids_ssd"

    out_random = project_dir / model_name / "grid_movies_without_density"
    out_density = project_dir / model_name / "grid_movies_with_density"

    print("=" * 80)
    print("🎬 REBUILDING GRID MOVIES FOR SYLLABLES 30 TO 99 (MATCHING 0..29 SPECS)")
    print(f"   Project: {project_dir}")
    print(f"   Model:   {model_name}")
    print("   Protected: Syllables 0 to 29 (will NOT be touched)")
    print("   Target:    Syllables 30 to 99")
    print("   Duration:  3.00 seconds (1.0s pre, 2.0s post = 120 frames)")
    print("   Resolution: max_video_size = 1920 (matching Syllables 0..29)")
    print("=" * 80, flush=True)

    # 1. Load config & model
    print("[1/4] Loading configuration and model checkpoint...", flush=True)
    config = kpms.load_config(str(project_dir))
    bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
    config["bodyparts"] = bodyparts

    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.load_results(str(project_dir), model_name)

    # 2. Coordinates & Spatial Metrics
    print("[2/4] Estimating coordinates, centroids, and headings...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"]),
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    # 3. VFI and Videos
    vfi_path = project_dir / model_name / "video_frame_indexes.npz"
    with np.load(vfi_path) as z:
        video_frame_indexes = {k: z[k] for k in z.files}

    target_fps = int(config.get("fps", 40))
    video_paths = {}
    ssd_vids = list(video_dir.glob("*.avi"))
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

    common_keys = sorted(list(set(results.keys()) & set(video_frame_indexes.keys()) & set(video_paths.keys())))
    results = {k: results[k] for k in common_keys}
    coordinates = {k: coordinates[k] for k in common_keys}
    centroids = {k: centroids[k] for k in common_keys}
    headings = {k: headings[k] for k in common_keys}
    video_frame_indexes = {k: video_frame_indexes[k] for k in common_keys}

    # Filter centroids and headings once
    centroids, headings = util.filter_centroids_headings(centroids, headings, filter_size=9)

    # Extract syllable instances with pre=40, post=80 (3.0s window)
    pre = round(1.0 * target_fps)   # 40 frames
    post = round(2.0 * target_fps)  # 80 frames
    syllables = {k: v["syllable"] for k, v in results.items()}
    all_instances = util.get_syllable_instances(
        syllables,
        pre=pre,
        post=post,
        min_duration=2,
        min_frequency=0.0,
        min_instances=1,
    )

    # Scope strictly to syllables 30 through 99
    target_sylls = [s for s in range(30, 100) if s in all_instances]
    print(f"\n[3/4] Scoped to {len(target_sylls)} active syllables in range [30, 99].", flush=True)

    rows, cols = 4, 6
    num_samples = rows * cols  # 24
    max_video_size = 1920

    # Exact resolutions matching Syllables 0..29:
    # Random:  window_size=272, scaled_window_size=272 -> 1632x1088
    # Density: window_size=320, scaled_window_size=320 -> 1920x1280
    random_window_size = 272
    density_window_size = 320

    # -----------------------------------------------------------------------
    # RUN A: RANDOM GRID MOVIES (Syllables 30 to 99)
    # -----------------------------------------------------------------------
    # Check which random movies need rendering (size < 2 MB means old 1.0s low-res)
    missing_random = [s for s in target_sylls if not ((out_random / f"syllable{s}.mp4").exists() and (out_random / f"syllable{s}.mp4").stat().st_size > 2000000)]
    print(f"Random Grid Movies to render: {len(missing_random)} / {len(target_sylls)}")

    if missing_random:
        print("\n" + "=" * 80)
        print(f"▶ RENDERING {len(missing_random)} RANDOM GRID MOVIES (Syllables 30..99)")
        print(f"   Target Dir: {out_random} (Format: 1632x1088, 3.0s, 120 frames)")
        print("=" * 80, flush=True)

        subset_instances = {s: all_instances[s] for s in missing_random}
        sampled_random = robust_sample_instances(
            subset_instances,
            num_samples=num_samples,
            mode="random",
        )

        for s in tqdm.tqdm(sorted(sampled_random.keys()), desc="Random Grid", ncols=75):
            assert s >= 30, f"Error: Syllable {s} is < 30! Syllables 0..29 are protected!"
            instances = sampled_random[s]
            frames = viz.grid_movie(
                instances,
                rows,
                cols,
                video_paths,
                centroids,
                headings,
                random_window_size,
                video_frame_indexes,
                scaled_window_size=random_window_size,
                pre=pre,
                post=post,
                overlay_keypoints=False,
                coordinates=coordinates,
            )
            out_path = out_random / f"syllable{s}.mp4"
            viz.write_video_clip(frames, str(out_path), fps=target_fps, quality=8)
        print("✓ Completed Random Grid Movies!", flush=True)
    else:
        print("✓ All Random Grid Movies are already up to date (3.0s, 1632x1088)!", flush=True)

    # -----------------------------------------------------------------------
    # RUN B: DENSITY GRID MOVIES (Syllables 30 to 99)
    # -----------------------------------------------------------------------
    missing_density = [s for s in target_sylls if not ((out_density / f"syllable{s}.mp4").exists() and (out_density / f"syllable{s}.mp4").stat().st_size > 2000000)]
    print(f"\nDensity Grid Movies to render: {len(missing_density)} / {len(target_sylls)}")

    if missing_density:
        print("\n" + "=" * 80)
        print(f"▶ RENDERING {len(missing_density)} DENSITY GRID MOVIES (Syllables 30..99)")
        print(f"   Target Dir: {out_density} (Format: 1920x1280, 3.0s, 120 frames)")
        print("=" * 80, flush=True)

        subset_instances_density = {s: all_instances[s] for s in missing_density}
        sampled_density = robust_sample_instances(
            subset_instances_density,
            num_samples=num_samples,
            mode="density",
            pca_samples=50000,
            pca_dim=4,
            n_neighbors=50,
            coordinates=coordinates,
            pre=pre,
            post=post,
            centroids=centroids,
            headings=headings,
        )

        for s in tqdm.tqdm(sorted(sampled_density.keys()), desc="Density Grid", ncols=75):
            assert s >= 30, f"Error: Syllable {s} is < 30! Syllables 0..29 are protected!"
            instances = sampled_density[s]
            frames = viz.grid_movie(
                instances,
                rows,
                cols,
                video_paths,
                centroids,
                headings,
                density_window_size,
                video_frame_indexes,
                scaled_window_size=density_window_size,
                pre=pre,
                post=post,
                overlay_keypoints=False,
                coordinates=coordinates,
            )
            out_path = out_density / f"syllable{s}.mp4"
            viz.write_video_clip(frames, str(out_path), fps=target_fps, quality=8)
        print("✓ Completed Density Grid Movies!", flush=True)
    else:
        print("✓ All Density Grid Movies are already up to date (3.0s, 1920x1280)!", flush=True)

    # -----------------------------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 VERIFYING REBUILT GRID MOVIES (SAMPLES)")
    print("=" * 80, flush=True)
    for folder in [out_density, out_random]:
        print(f"\nChecking: {folder.name}")
        for test_s in [0, 29, 30, 50, 70, 90]:
            p = folder / f"syllable{test_s}.mp4"
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"  syllable{test_s:2d}.mp4: {w}x{h} | {nf} frames ({nf/40.0:.2f}s)")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
