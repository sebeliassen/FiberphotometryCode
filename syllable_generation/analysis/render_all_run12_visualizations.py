#!/usr/bin/env python3
"""
render_all_run12_visualizations.py
=============================================================================
Generates Keypoint-MoSeq Visualizations for ALL 100 Syllables in Run 12
(amph_prod_k150k_run_12):
1. Grid movies WITH density sampling (Canonical Exemplar / PCA mode)
2. Grid movies WITHOUT density sampling (Random uniform mode)
3. Trajectory plots (GIF animations and PDF static vector plots)

Features:
- Incremental execution: skips any syllable that already has a valid output file.
- Universal coverage: handles all 100 syllables (ranks 0-99). For rare syllables
  with < 24 bouts, samples with replacement so all tiles in the 4x6 grid are filled.
- Forces CPU execution to avoid ptxas / GPU conflicts.
- Applied OpenCVReader fast-seek monkeypatch for fast video reading.
- Applied Matplotlib buffer patch for modern matplotlib compatibility.
=============================================================================
"""

import os
import sys
from pathlib import Path
import warnings

# Force CPU execution & headless operation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import tqdm

# ---------------------------------------------------------------------------
# Patches & Optimizations
# ---------------------------------------------------------------------------
# Fast-seek monkeypatch for OpenCVReader
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

# Matplotlib buffer patch for modern matplotlib
def modern_rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        return np.asarray(canvas.buffer_rgba())[:, :, :3]
    elif hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3))
    return np.asarray(canvas.print_to_buffer()[0])[:, :, :3]
viz.rasterize_figure = modern_rasterize_figure

# ---------------------------------------------------------------------------
# Robust Instance Sampler (Supports rare syllables with < 24 instances)
# ---------------------------------------------------------------------------
def robust_sample_instances(
    syllable_instances,
    num_samples,
    mode="random",
    pca_samples=50000,
    pca_dim=4,
    n_neighbors=50,
    coordinates=None,
    pre=5,
    post=15,
    centroids=None,
    headings=None,
    filter_size=9,
):
    """
    Sample num_samples instances for each syllable.
    If a syllable has fewer than num_samples, sample with replacement (tile).
    """
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
        
        # Clean coordinates
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
            eff_k = min(n_neighbors, n_inst - 1) if n_inst > 1 else 1
            nbrs = NearestNeighbors(n_neighbors=eff_k, algorithm="kd_tree").fit(Xpca)
            distances, indices = nbrs.kneighbors(Xpca)
            local_density = 1.0 / (distances.mean(1) + 1e-6)

            all_dists, _ = all_nbrs.kneighbors(Xpca)
            global_density = 1.0 / (all_dists.mean(1) + 1e-6)
            exemplar = np.argmax(local_density / global_density)

            cand_indices = indices[exemplar]
            if len(cand_indices) >= num_samples:
                idx = np.random.choice(cand_indices, num_samples, replace=False)
            else:
                idx = np.random.choice(n_inst, num_samples, replace=False)
            sampled[s] = [inst[i] for i in idx]

        return sampled


def main():
    project_dir = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    model_name = "amph_prod_k150k_run_12"
    video_dir = REPO_ROOT / "src_oft/amph_vids_ssd"

    out_random = project_dir / model_name / "grid_movies_without_density"
    out_density = project_dir / model_name / "grid_movies_with_density"
    out_traj = project_dir / model_name / "trajectory_plots_density"

    out_random.mkdir(parents=True, exist_ok=True)
    out_density.mkdir(parents=True, exist_ok=True)
    out_traj.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 COMPLETE RUN 12 VISUALIZATION PIPELINE (ALL 100 SYLLABLES)")
    print(f"   Project: {project_dir}")
    print(f"   Model:   {model_name}")
    print("=" * 80)

    # 1. Load config & checkpoint
    print("\n[1/5] Loading configuration and model...", flush=True)
    config = kpms.load_config(str(project_dir))
    bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
    config["bodyparts"] = bodyparts
    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.load_results(str(project_dir), model_name)

    # 2. Coordinates, centroids, headings
    print("[2/5] Estimating coordinates, centroids, headings...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    # 3. Load VFI & map videos
    print("[3/5] Mapping videos and loading VFI...", flush=True)
    vfi_path = project_dir / model_name / "video_frame_indexes.npz"
    with np.load(vfi_path) as z:
        video_frame_indexes = {k: z[k] for k in z.files}

    target_fps = int(config.get("fps", 40))
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

    config_clean = config.copy()
    config_clean.pop("fps", None)
    config_clean.pop("overlay_keypoints", None)

    # Filter keys to common intersection
    common_keys = sorted(list(set(results.keys()) & set(video_frame_indexes.keys()) & set(video_paths.keys())))
    results = {k: results[k] for k in common_keys}
    coordinates = {k: coordinates[k] for k in common_keys}
    centroids = {k: centroids[k] for k in common_keys}
    headings = {k: headings[k] for k in common_keys}
    video_frame_indexes = {k: video_frame_indexes[k] for k in common_keys}

    # Extract instances for all syllables with min_instances=1
    print("\n[4/5] Extracting syllable instances across cohort...", flush=True)
    syllables = {k: v["syllable"] for k, v in results.items()}
    all_instances = util.get_syllable_instances(
        syllables,
        pre=round(0.5 * target_fps),
        post=round(0.5 * target_fps),
        min_duration=2,
        min_frequency=0.0,
        min_instances=1,
    )
    print(f"Total active syllables found in cohort: {len(all_instances)} / 100")

    # Determine which syllables need generation
    def get_missing_grid(target_dir):
        missing = []
        for s in range(100):
            p = target_dir / f"syllable{s}.mp4"
            if not (p.exists() and p.stat().st_size > 10000):
                missing.append(s)
        return missing

    missing_random = get_missing_grid(out_random)
    missing_density = get_missing_grid(out_density)

    print(f"Missing random grid movies:  {len(missing_random)}")
    print(f"Missing density grid movies: {len(missing_density)}")

    # Window size determination
    window_size = config.get("window_size", 192)
    rows, cols = 4, 6
    num_samples = rows * cols
    max_video_size = 720
    scaled_window_size = max_video_size / max(rows, cols)
    scaled_window_size = int(np.floor(scaled_window_size / 16) * 16)
    scaled_window_size = min(scaled_window_size, window_size)

    # Render missing RANDOM grid movies
    if missing_random:
        print(f"\n▶ Rendering {len(missing_random)} missing syllables for Random Grid Movies...")
        subset_instances = {s: all_instances[s] for s in missing_random if s in all_instances}
        sampled_random = robust_sample_instances(
            subset_instances,
            num_samples=num_samples,
            mode="random",
        )
        for s, instances in tqdm.tqdm(sampled_random.items(), desc="Random Grid Movies", ncols=75):
            frames = viz.grid_movie(
                instances,
                rows,
                cols,
                video_paths,
                centroids,
                headings,
                window_size,
                video_frame_indexes,
                scaled_window_size=scaled_window_size,
                pre=round(0.5 * target_fps),
                post=round(0.5 * target_fps),
                overlay_keypoints=False,
                coordinates=coordinates,
            )
            out_path = out_random / f"syllable{s}.mp4"
            viz.write_video_clip(frames, str(out_path), fps=target_fps, quality=8)

    # Render missing DENSITY grid movies
    if missing_density:
        print(f"\n▶ Rendering {len(missing_density)} missing syllables for Density Grid Movies...")
        subset_instances = {s: all_instances[s] for s in missing_density if s in all_instances}
        sampled_density = robust_sample_instances(
            subset_instances,
            num_samples=num_samples,
            mode="density",
            pca_samples=50000,
            pca_dim=4,
            n_neighbors=50,
            coordinates=coordinates,
            pre=round(0.5 * target_fps),
            post=round(0.5 * target_fps),
            centroids=centroids,
            headings=headings,
        )
        for s, instances in tqdm.tqdm(sampled_density.items(), desc="Density Grid Movies", ncols=75):
            frames = viz.grid_movie(
                instances,
                rows,
                cols,
                video_paths,
                centroids,
                headings,
                window_size,
                video_frame_indexes,
                scaled_window_size=scaled_window_size,
                pre=round(0.5 * target_fps),
                post=round(0.5 * target_fps),
                overlay_keypoints=False,
                coordinates=coordinates,
            )
            out_path = out_density / f"syllable{s}.mp4"
            viz.write_video_clip(frames, str(out_path), fps=target_fps, quality=8)

    # -----------------------------------------------------------------------
    # [5/5] Trajectory Plots for all missing syllables
    # -----------------------------------------------------------------------
    print("\n[5/5] Checking and rendering Trajectory Plots (GIF & PDF)...", flush=True)
    missing_traj = []
    for s in range(100):
        p_gif = out_traj / f"Syllable{s}.gif"
        p_pdf = out_traj / f"Syllable{s}.pdf"
        if not (p_gif.exists() and p_gif.stat().st_size > 1000 and p_pdf.exists() and p_pdf.stat().st_size > 1000):
            missing_traj.append(s)

    print(f"Missing trajectory plots: {len(missing_traj)} / 100 syllables")
    if missing_traj:
        traj_pre = 5
        traj_post = 15
        
        # Sample exemplar instances for trajectory
        subset_instances = {s: all_instances[s] for s in missing_traj if s in all_instances}
        sampled_traj = robust_sample_instances(
            subset_instances,
            num_samples=min(24, max(1, min([len(v) for v in subset_instances.values()] or [1]))),
            mode="density",
            pca_samples=50000,
            pca_dim=4,
            n_neighbors=50,
            coordinates=coordinates,
            pre=traj_pre,
            post=traj_post,
            centroids=centroids,
            headings=headings,
        )

        clean_coords = {}
        for key in coordinates.keys():
            arr = coordinates[key].copy()
            outliers = np.isnan(arr).any(-1)
            clean_coords[key] = util.interpolate_keypoints(arr, outliers)

        traj_data = {}
        for s, inst in sampled_traj.items():
            ts = util.get_instance_trajectories(
                inst,
                clean_coords,
                pre=traj_pre,
                post=traj_post,
                centroids=centroids,
                headings=headings,
            )
            traj_data[s] = np.nanmedian(ts, axis=0)

        # Bodyparts & skeleton
        use_bps = config.get("use_bodyparts", config.get("bodyparts"))
        skeleton = config.get("skeleton", [])
        edges = util.get_edges(use_bps, skeleton) if len(skeleton) > 0 else []

        # Coordinate flip for display
        Xs_dict = {s: arr * np.array([1, -1]) for s, arr in traj_data.items()}
        
        # Compute global limits
        all_stacked = np.stack(list(Xs_dict.values()))
        lims = viz.get_limits(all_stacked, pctl=0, left=0.1, right=0.1, top=0.2, bottom=0.2)

        for s, X_2D in tqdm.tqdm(Xs_dict.items(), desc="Generating Trajectories", ncols=75):
            title = f"Syllable{s}"
            fig, ax, rasters = viz.plot_trajectories(
                [title],
                X_2D[None],
                lims,
                edges=edges,
                return_rasters=True,
                keypoint_colormap=config.get("keypoint_colormap", "autumn"),
            )
            pdf_path = out_traj / f"{title}.pdf"
            plt.savefig(str(pdf_path))
            plt.close(fig)

            gif_path = out_traj / f"{title}.gif"
            frame_duration = (traj_pre + traj_post) / len(rasters) / target_fps
            viz.save_gif(rasters, str(gif_path), duration=frame_duration)

    # -----------------------------------------------------------------------
    # Final Inventory & Integrity Check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📋 FINAL VISUALIZATION ASSETS INVENTORY (RUN 12)")
    print("=" * 80)
    final_rand = [s for s in range(100) if (out_random / f"syllable{s}.mp4").exists()]
    final_dens = [s for s in range(100) if (out_density / f"syllable{s}.mp4").exists()]
    final_traj_gif = [s for s in range(100) if (out_traj / f"Syllable{s}.gif").exists()]
    final_traj_pdf = [s for s in range(100) if (out_traj / f"Syllable{s}.pdf").exists()]

    print(f"Random Grid Movies:    {len(final_rand)} / 100 syllables")
    print(f"Density Grid Movies:   {len(final_dens)} / 100 syllables")
    print(f"Trajectory GIFs:       {len(final_traj_gif)} / 100 syllables")
    print(f"Trajectory PDFs:       {len(final_traj_pdf)} / 100 syllables")

    if len(final_rand) == 100 and len(final_dens) == 100 and len(final_traj_gif) == 100:
        print("\n🎉 ALL 100 SYLLABLES COMPLETE AND VERIFIED!")
    else:
        print("\nNote: Some rare syllables may have had 0 instances matching filters.")
    print("=" * 80)

if __name__ == "__main__":
    main()
