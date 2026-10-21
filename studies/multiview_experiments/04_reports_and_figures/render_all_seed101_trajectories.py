#!/usr/bin/env python3
"""
render_all_seed101_trajectories.py
=============================================================================
Publication-grade Trajectory Visualizations for ALL Syllables in Champion Seed 101:
- DENSITY SAMPLING ENABLED: Samples exemplar instances maximizing PCA trajectory space density.
- FULL COHORT COVERAGE: Gracefully handles frequent, moderate, and rare syllables
  (with adaptive neighbor counts and all-instance fallback for rare syllables).
1. Superimposed Plots (TopCam Dorsal + BotCam Ventral fused in shared arena frame)
   - Vector PDF (publication ready)
   - 15-frame looping animated GIF
2. Tandem Side-by-Side Plots (Dorsal Spine on Left | Ventral Paws on Right)
   - Decoupled view-specific axis limits (lims_dorsal vs lims_ventral)
   - Auto-zoomed on paw stance/swing phases without dorsal spine clipping
   - Vector PDF + 15-frame looping animated GIF

Environment: kpms_gpu_cuml
=============================================================================
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib

matplotlib.use("Agg")
import pickle

import h5py
import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
import matplotlib.pyplot as plt
from jax_moseq.models.keypoint_slds import estimate_coordinates
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Modern Matplotlib Rasterize Patch
def modern_rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    elif hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3)).copy()
    return np.asarray(canvas.print_to_buffer()[0])[:, :, :3].copy()
viz.rasterize_figure = modern_rasterize_figure

# Paths & Directories
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed101"
MODEL_DIR = PROJECT_DIR / MODEL_NAME
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.h5"
RESULTS_PATH = MODEL_DIR / "results.h5"

OUT_DIR = SCRIPT_DIR / "visualizations_seed101" / "trajectories"
SUPER_DIR = OUT_DIR / "superimposed"
SIDE_DIR = OUT_DIR / "side_by_side"
SUPER_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80, flush=True)
    print("🎬 RENDERING ALL SEED 101 TRAJECTORY PLOTS (BUILT-IN DENSITY SAMPLING)", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Config & Anatomy
    print("[1/4] Loading project configuration and anatomical skeleton...", flush=True)
    config = kpms.load_config(str(PROJECT_DIR))
    use_bps = config["use_bodyparts"]
    skeleton = config.get("skeleton", [])
    edges_all = util.get_edges(use_bps, skeleton) if len(skeleton) > 0 else []
    edges_all_tuples = [tuple(e) for e in edges_all]

    extra_ventral_global = [
        (17, 18), # Left Front Paw - Right Front Paw
        (19, 20), # Left Hind Paw - Right Hind Paw
        (17, 19), # Left Front Paw - Left Hind Paw
        (18, 20), # Right Front Paw - Right Hind Paw
        (21, 17), # Bot Nose - Left Front Paw
        (21, 18), # Bot Nose - Right Front Paw
        (22, 19), # Bot Tail - Left Hind Paw
        (22, 20), # Bot Tail - Right Hind Paw
        (21, 22), # Bot Nose - Bot Tail midline axis
    ]
    edges_super = list(set(edges_all_tuples + extra_ventral_global))

    dorsal_idxs = list(range(17))
    ventral_idxs = list(range(17, 23))

    edges_dorsal = [(i, j) for (i, j) in edges_all_tuples if i < 17 and j < 17]
    edges_ventral = [(i - 17, j - 17) for (i, j) in extra_ventral_global]

    colors_super = np.zeros((23, 3), dtype=float)
    colors_super[:17] = plt.cm.Blues(np.linspace(0.45, 0.95, 17))[:, :3]
    colors_super[17] = [0.90, 0.15, 0.15] # LF
    colors_super[18] = [0.90, 0.15, 0.15] # RF
    colors_super[19] = [0.95, 0.50, 0.10] # LH
    colors_super[20] = [0.95, 0.50, 0.10] # RH
    colors_super[21] = [0.85, 0.70, 0.10] # Bot Nose
    colors_super[22] = [0.85, 0.70, 0.10] # Bot Tail

    colors_dorsal = colors_super[:17]
    colors_ventral = colors_super[17:]

    # 2. Fast Parameter & Data Loading via HDF5 across Cohort
    print("[2/4] Reading observation matrix Cd and loading session results...", flush=True)
    with h5py.File(CHECKPOINT_PATH, "r") as f:
        Cd = f["model_snapshots/250/params/Cd"][:]
        Cd_jax = jnp.array(Cd)
        print(f"  Loaded Cd matrix with shape: {Cd.shape}", flush=True)

    traj_pre = 5
    traj_post = 20
    num_timesteps = 15

    # 2A. Instantaneous Cohort Syllable Scan to Index All Syllables
    print("  Scanning cohort results.h5 for all syllable occurrences...", flush=True)
    all_bouts_by_syll = {}
    with h5py.File(RESULTS_PATH, "r") as f:
        all_keys = list(f.keys())
        for k in all_keys:
            s_arr = f[k]["syllable"][:]
            diffs = np.diff(s_arr)
            bounds = np.where(diffs != 0)[0] + 1
            starts = np.r_[0, bounds]
            ends = np.r_[bounds, len(s_arr)]
            for start, end in zip(starts, ends):
                dur = end - start
                if dur >= 3 and start >= traj_pre and (start + traj_post) < len(s_arr):
                    s_id = int(s_arr[start])
                    if s_id not in all_bouts_by_syll:
                        all_bouts_by_syll[s_id] = []
                    all_bouts_by_syll[s_id].append((k, int(start), int(end)))

    all_sylls_found = sorted(all_bouts_by_syll.keys())
    print(f"  Found {len(all_sylls_found)} distinct syllables across cohort: {all_sylls_found}", flush=True)

    # 2B. Select up to 60 candidate bouts per syllable and identify needed sessions
    selected_bouts_by_syll = {}
    needed_sessions = set()
    for s, bouts in all_bouts_by_syll.items():
        if len(bouts) <= 60:
            chosen = bouts
        else:
            step = len(bouts) / 60.0
            chosen = [bouts[int(i * step)] for i in range(60)]
        selected_bouts_by_syll[s] = chosen
        for sid, _, _ in chosen:
            needed_sessions.add(sid)

    all_instances = selected_bouts_by_syll

    pkl_density_path = OUT_DIR.parent / "density_sampled_instances.pkl"
    pkl_traj_path = OUT_DIR.parent / "sampled_median_trajectories.pkl"

    # Check if we can skip session loading and extraction completely
    if pkl_density_path.exists() and pkl_traj_path.exists():
        print("  ⚡ Found cached trajectories and density instances! Loading directly...", flush=True)
        with open(pkl_density_path, "rb") as pf:
            sampled_instances = pickle.load(pf)
        with open(pkl_traj_path, "rb") as pf:
            traj_dict = pickle.load(pf)
        print(f"  Loaded {len(sampled_instances)} sampled instances and {len(traj_dict)} trajectories from cache.", flush=True)
    else:
        sample_sessions = sorted(list(needed_sessions))
        print(f"  Loading & reconstructing coordinates for {len(sample_sessions)} targeted sessions...", flush=True)

        centroids = {}
        headings = {}
        coordinates = {}

        with h5py.File(RESULTS_PATH, "r") as f:
            for idx, k in enumerate(sample_sessions):
                grp = f[k]
                x_arr = grp["latent_state"][:]
                v_arr = grp["centroid"][:]
                h_arr = grp["heading"][:]

                Y_est = estimate_coordinates(
                    jnp.array(x_arr),
                    jnp.array(v_arr),
                    jnp.array(h_arr),
                    Cd_jax
                )
                arr = np.array(Y_est)
                outliers = np.isnan(arr).any(-1)
                clean_coords = util.interpolate_keypoints(arr, outliers)

                coordinates[k] = clean_coords
                centroids[k] = v_arr
                headings[k] = h_arr

                if (idx + 1) % 5 == 0 or (idx + 1) == len(sample_sessions):
                    print(f"    [{idx+1}/{len(sample_sessions)}] Loaded {k}", flush=True)

        all_instances = selected_bouts_by_syll

        # 3. Extract All Syllable Instances with Graceful Rare Fallback
        print("[3/4] Extracting raw syllable trajectories & running density sampling...", flush=True)

        if pkl_density_path.exists():
            print(f"  ⚡ Found cached density sampled instances: {pkl_density_path}. Loading directly...", flush=True)
            with open(pkl_density_path, "rb") as pf:
                sampled_instances = pickle.load(pf)
            print(f"  Loaded density sampled instances for {len(sampled_instances)} syllables.", flush=True)
        else:
            # Robust Density Sampling with Rare Syllable Fallback (Never Crashes on Low-Frequency Syllables)
            print("  Extracting raw syllable trajectories for density modeling...", flush=True)
            raw_trajs_by_syll = {}
            for s in all_sylls_found:
                raw_trajs_by_syll[s] = util.get_instance_trajectories(
                    all_instances[s],
                    coordinates,
                    pre=traj_pre,
                    post=traj_post,
                    centroids=centroids,
                    headings=headings,
                    filter_size=9,
                )

            # Background PCA space (fit across sample of all cohort instances)
            all_traj_list = [t for t in raw_trajs_by_syll.values() if len(t) > 0]
            X_all = np.vstack(all_traj_list)
            pca_samples = min(20000, len(X_all))
            if len(X_all) > pca_samples:
                sub_idx = np.random.choice(len(X_all), pca_samples, replace=False)
                X_pca_train = X_all[sub_idx]
            else:
                X_pca_train = X_all

            pca = PCA(n_components=min(4, X_pca_train.shape[1])).fit(X_pca_train.reshape(X_pca_train.shape[0], -1))
            X_all_pca = pca.transform(X_all.reshape(X_all.shape[0], -1))

            global_k = min(50, len(X_all_pca))
            all_nbrs = NearestNeighbors(n_neighbors=global_k).fit(X_all_pca)

            sampled_instances = {}
            for s in all_sylls_found:
                inst = all_instances[s]
                N = len(inst)
                if N == 0:
                    continue
                if N < 10:
                    # Rare syllable fallback: keep all available instances
                    sampled_instances[s] = inst
                    continue

                # Contrastive density sampling (prototypical exemplar in PCA trajectory space)
                X_s = raw_trajs_by_syll[s]
                X_s_pca = pca.transform(X_s.reshape(X_s.shape[0], -1))

                k_local = min(50, N)
                nbrs = NearestNeighbors(n_neighbors=k_local).fit(X_s_pca)
                distances, indices = nbrs.kneighbors(X_s_pca)
                local_density = 1.0 / (distances.mean(axis=1) + 1e-8)

                distances_global, _ = all_nbrs.kneighbors(X_s_pca)
                global_density = 1.0 / (distances_global.mean(axis=1) + 1e-8)

                density_ratio = local_density / global_density
                exemplar_idx = int(np.argmax(density_ratio))

                num_to_sample = min(50, N)
                chosen_sub_indices = np.random.choice(indices[exemplar_idx], num_to_sample, replace=False)
                sampled_instances[s] = [inst[i] for i in chosen_sub_indices]

            print(f"  Density sampling complete across all {len(sampled_instances)} syllables.")

            # Save sampled density instances for grid movie generator
            with open(pkl_density_path, "wb") as pf:
                pickle.dump(sampled_instances, pf)
            print(f"  Saved density sampled instances to: {pkl_density_path}")

        # Compute median trajectories
        if pkl_traj_path.exists():
            print(f"  ⚡ Found cached median trajectories: {pkl_traj_path}. Loading directly...", flush=True)
            with open(pkl_traj_path, "rb") as pf:
                traj_dict = pickle.load(pf)
            print(f"  Loaded median trajectories for {len(traj_dict)} syllables.", flush=True)
        else:
            traj_dict = {}
            for s in all_sylls_found:
                inst = sampled_instances.get(s, [])
                if len(inst) == 0:
                    continue
                ts = util.get_instance_trajectories(
                    inst,
                    coordinates,
                    pre=traj_pre,
                    post=traj_post,
                    centroids=centroids,
                    headings=headings,
                )
                traj_dict[s] = np.nanmedian(ts, axis=0) * np.array([1, -1])

            with open(pkl_traj_path, "wb") as pf:
                pickle.dump(traj_dict, pf)
            print(f"  Saved median trajectories to: {pkl_traj_path}")

        print(f"  Successfully extracted trajectories for {len(traj_dict)} syllables.")

    # Global axis limits across all valid syllables
    stacked_all = np.stack(list(traj_dict.values()))
    lims_super = viz.get_limits(stacked_all, pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_dorsal = viz.get_limits(stacked_all[:, :, dorsal_idxs], pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_ventral = viz.get_limits(stacked_all[:, :, ventral_idxs], pctl=0, left=0.25, right=0.25, top=0.25, bottom=0.25)

    # 4. Render Visualizations
    print(f"\n[4/4] Rendering Superimposed & Side-by-Side PDFs and GIFs for {len(traj_dict)} syllables...", flush=True)

    for idx, s in enumerate(all_sylls_found):
        if s not in traj_dict:
            continue
        traj = traj_dict[s]
        n_inst = len(sampled_instances.get(s, []))
        sampling_tag = "Density Sampled" if len(all_instances.get(s, [])) >= 10 else f"Rare ({n_inst} bouts)"
        desc = f"Syllable {s:02d}"

        pdf_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.pdf"
        gif_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.gif"
        pdf_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.pdf"
        gif_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.gif"

        if pdf_super.exists() and gif_super.exists() and pdf_side.exists() and gif_side.exists():
            if (idx + 1) % 10 == 0 or (idx + 1) == len(all_sylls_found):
                print(f"  [{idx+1}/{len(all_sylls_found)}] Syllable {s:02d} already rendered (resumed).", flush=True)
            continue

        # A. Superimposed
        fig_s, ax_s, rasters_s = viz.plot_trajectories(
            [f"{desc} — Superimposed ({sampling_tag})"],
            traj[None],
            lims_super,
            edges=edges_super,
            keypoint_colors=colors_super,
            return_rasters=True,
            num_timesteps=num_timesteps,
            node_size=42.0,
            line_width=2.6,
            alpha=0.15,
            plot_width=5.2,
        )
        pdf_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.pdf"
        fig_s.savefig(str(pdf_super), bbox_inches="tight", dpi=300)
        plt.close(fig_s)

        gif_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.gif"
        images_s = [Image.fromarray(r) for r in rasters_s]
        images_s[0].save(
            str(gif_super),
            save_all=True,
            append_images=images_s[1:],
            duration=120,
            loop=0
        )

        # B. Side-by-Side (Dorsal left | Ventral right)
        traj_dorsal = traj[:, dorsal_idxs]
        traj_ventral = traj[:, ventral_idxs]

        fig_d, _, rasters_d = viz.plot_trajectories(
            [f"{desc} — Dorsal Spine ({sampling_tag})"],
            traj_dorsal[None],
            lims_dorsal,
            edges=edges_dorsal,
            keypoint_colors=colors_dorsal,
            return_rasters=True,
            num_timesteps=num_timesteps,
            node_size=42.0,
            line_width=2.6,
            alpha=0.15,
            plot_width=4.8,
        )
        plt.close(fig_d)

        fig_v, _, rasters_v = viz.plot_trajectories(
            [f"{desc} — Ventral Paws ({sampling_tag})"],
            traj_ventral[None],
            lims_ventral,
            edges=edges_ventral,
            keypoint_colors=colors_ventral,
            return_rasters=True,
            num_timesteps=num_timesteps,
            node_size=55.0,
            line_width=3.2,
            alpha=0.15,
            plot_width=4.8,
        )
        plt.close(fig_v)

        # Height normalization to ensure rasters_d and rasters_v concatenate seamlessly
        side_rasters = []
        for r_d, r_v in zip(rasters_d, rasters_v):
            h = min(r_d.shape[0], r_v.shape[0])
            w_d = int(round(r_d.shape[1] * (h / r_d.shape[0])))
            w_v = int(round(r_v.shape[1] * (h / r_v.shape[0])))
            img_d = np.array(Image.fromarray(np.uint8(r_d)).resize((w_d, h), Image.Resampling.LANCZOS))
            img_v = np.array(Image.fromarray(np.uint8(r_v)).resize((w_v, h), Image.Resampling.LANCZOS))
            divider = np.full((h, 4, 3), 255, dtype=np.uint8)
            side_rasters.append(np.hstack([img_d, divider, img_v]))

        gif_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.gif"
        images_side = [Image.fromarray(r) for r in side_rasters]
        images_side[0].save(
            str(gif_side),
            save_all=True,
            append_images=images_side[1:],
            duration=120,
            loop=0
        )

        # PDF side-by-side
        fig_tandem, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        ax_left.imshow(rasters_d[-1])
        ax_left.axis("off")

        ax_right.imshow(rasters_v[-1])
        ax_right.axis("off")

        plt.subplots_adjust(wspace=0.04)
        pdf_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.pdf"
        fig_tandem.savefig(str(pdf_side), bbox_inches="tight", dpi=300)
        plt.close(fig_tandem)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_sylls_found):
            print(f"  Rendered [{idx+1}/{len(all_sylls_found)}] Syllable {s:02d} ({sampling_tag})...", flush=True)

    print("=" * 80)
    print("🎉 ALL SYLLABLE TRAJECTORY PLOTS & GIFS GENERATED SUCCESSFULLY!")
    print(f"   Superimposed: {SUPER_DIR}")
    print(f"   Side-by-Side: {SIDE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
