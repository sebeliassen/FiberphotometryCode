#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_seed101_trajectories.py
=============================================================================
Publication-grade Trajectory Visualizations for Cohort Static 21kp Seed 101:
- PURE NUMPY / SCIPY / MATPLOTLIB (Zero JAX compiler overhead, zero hangs)
- DENSITY SAMPLING: Samples exemplar instances maximizing PCA trajectory space density.
- FULL COHORT COVERAGE: Gracefully handles frequent, moderate, and rare syllables.
- 21 KEYPOINTS ARCHITECTURE:
  * 17 Dorsal TopCam keypoints (0..16)
  * 4 Ventral Paws (17..20: LF, RF, LH, RH)
- DUAL VISUAL MODES:
  1. Superimposed Plots (TopCam Dorsal + BotCam Paws fused in shared arena frame)
     - Vector PDF (publication ready, 300 DPI)
     - 15-frame looping animated GIF
  2. Tandem Side-by-Side Plots (Dorsal Spine on Left | Ventral Paws on Right)
     - Decoupled view-specific axis limits (lims_dorsal vs lims_ventral)
     - Auto-zoomed on paw stance/swing phases without dorsal spine clipping
     - Vector PDF (300 DPI) + 15-frame looping animated GIF

Environment: keypoint_moseq or kpms_gpu_cuml (pure CPU)
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import h5py
import pickle
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Paths & Directories
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXP_DIR / "cohort_static_21kp"
MODEL_NAME = "model_21kp_static_50ar_200slds_seed101"
MODEL_DIR = PROJECT_DIR / MODEL_NAME
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.h5"
RESULTS_PATH = MODEL_DIR / "results.h5"
CONFIG_PATH = PROJECT_DIR / "config.yml"

OUT_DIR = SCRIPT_DIR / "visualizations_cohort_static_21kp_seed101" / "trajectories"
SUPER_DIR = OUT_DIR / "superimposed"
SIDE_DIR = OUT_DIR / "side_by_side"
SUPER_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pure NumPy Keypoint SLDS Coordinate Reconstruction
# =============================================================================

def center_embedding(k):
    """Gamma basis matrix for centered keypoints."""
    A = np.zeros((k, k - 1))
    A[0, 0] = -1.0 / np.sqrt(2.0)
    A[1, 0] = 1.0 / np.sqrt(2.0)
    for i in range(2, k):
        A[:i, i - 1] = -1.0 / np.sqrt(i * (i + 1.0))
        A[i, i - 1] = i / np.sqrt(i * (i + 1.0))
    return A


def estimate_coordinates_numpy(x, v, h, Cd):
    """
    Reconstructs keypoint coordinates from latent state x, centroid v, heading h,
    and observation matrix Cd in pure NumPy.
    """
    C = Cd[:, :-1]
    d_bias = Cd[:, -1]
    y = x @ C.T + d_bias
    k = Cd.shape[0] // 2 + 1
    y = y.reshape(-1, k - 1, 2)
    Gamma = center_embedding(k)
    Y_bar = Gamma @ y  # shape: (T, k, 2)

    # Rigid transform: rotate by h and translate by v
    cos_h = np.cos(h)[:, None, None]
    sin_h = np.sin(h)[:, None, None]
    x_rot = Y_bar[:, :, 0:1] * cos_h - Y_bar[:, :, 1:2] * sin_h
    y_rot = Y_bar[:, :, 0:1] * sin_h + Y_bar[:, :, 1:2] * cos_h
    Y_rot = np.concatenate([x_rot, y_rot], axis=-1)
    Y_est = Y_rot + v[:, None, :]
    return Y_est


def inverse_rigid_transform_numpy(Y, v, h):
    """
    Transforms coordinates into the egocentric reference frame defined by v and h.
    Y: (..., 2), v: (..., 2), h: (...)
    """
    Y_centered = Y - v
    cos_h = np.cos(h)[:, None, None]
    sin_h = np.sin(h)[:, None, None]
    x_rot = Y_centered[:, :, 0:1] * cos_h + Y_centered[:, :, 1:2] * sin_h
    y_rot = -Y_centered[:, :, 0:1] * sin_h + Y_centered[:, :, 1:2] * cos_h
    return np.concatenate([x_rot, y_rot], axis=-1)


def interpolate_keypoints(arr, outliers):
    """Linearly interpolates outlier coordinates."""
    clean = arr.copy()
    clean[outliers] = np.nan
    T, K, D = clean.shape
    for k in range(K):
        for d in range(D):
            s = pd.Series(clean[:, k, d])
            clean[:, k, d] = s.interpolate(method="linear", limit_direction="both").fillna(0.0).values
    return clean


def get_instance_trajectories_numpy(instances, coordinates, pre=5, post=20, centroids=None, headings=None, filter_size=9):
    """Extracts egocentrically centered trajectories for a list of bouts."""
    if centroids is not None and headings is not None and filter_size > 1:
        c_filt = {k: median_filter(v, size=(filter_size, 1)) for k, v in centroids.items()}
        h_filt = {k: median_filter(v, size=filter_size) for k, v in headings.items()}
    else:
        c_filt = centroids
        h_filt = headings

    trajs = []
    for sid, s, e in instances:
        seg = coordinates[sid][s - pre : s + post]
        if c_filt is not None and h_filt is not None:
            c_onset = c_filt[sid][s:s+1]
            h_onset = h_filt[sid][s:s+1]
            seg = inverse_rigid_transform_numpy(seg, c_onset, h_onset)
        trajs.append(seg)
    return np.array(trajs)


def get_limits(coordinates, pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20):
    """Computes robust bounding box limits across trajectory coordinates."""
    coords = coordinates.reshape(-1, 2)
    valid = coords[~np.isnan(coords).any(axis=1)]
    if len(valid) == 0:
        return np.array([[-50, -50], [50, 50]])
    xmin, ymin = np.percentile(valid, pctl, axis=0)
    xmax, ymax = np.percentile(valid, 100 - pctl, axis=0)
    w = max(10.0, xmax - xmin)
    h = max(10.0, ymax - ymin)
    return np.array([
        [xmin - w * left, ymin - h * bottom],
        [xmax + w * right, ymax + h * top]
    ])


def rasterize_figure(fig):
    """Canvas rasterizer compatible with Matplotlib 3.8+."""
    canvas = fig.canvas
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    elif hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3)).copy()
    return np.asarray(canvas.print_to_buffer()[0])[:, :, :3].copy()


def plot_trajectories_clean(
    title,
    X,
    lims,
    edges=[],
    keypoint_colors=None,
    node_size=42.0,
    line_width=2.6,
    num_timesteps=15,
    plot_width=5.0,
):
    """
    Renders high-contrast publication pose trajectories with time interpolation.
    X: shape (n_frames, n_keypoints, 2)
    Returns: (fig, rasters)
    """
    n_frames, n_kps, _ = X.shape
    if keypoint_colors is None:
        keypoint_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_kps))[:, :3]

    # Interpolate along time
    f_interp = interp1d(np.arange(n_frames), X, axis=0, kind="linear")
    t_steps = np.linspace(0, n_frames - 1, num_timesteps)
    X_steps = f_interp(t_steps)

    xmin, ymin = lims[0]
    xmax, ymax = lims[1]
    aspect = (ymax - ymin) / max(1e-5, (xmax - xmin))
    fig_h = max(3.5, plot_width * aspect)

    fig, ax = plt.subplots(figsize=(plot_width, fig_h), dpi=300)
    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    rasters = []
    for step_idx in range(num_timesteps):
        pt = X_steps[step_idx]
        alpha = 0.25 + 0.75 * (step_idx / float(num_timesteps - 1))

        # Plot bones
        for i, j in edges:
            if i < n_kps and j < n_kps:
                ax.plot([pt[i, 0], pt[j, 0]], [pt[i, 1], pt[j, 1]],
                        color="black", linewidth=line_width * 1.1, alpha=min(1.0, alpha * 0.8), zorder=step_idx * 2)
                ax.plot([pt[i, 0], pt[j, 0]], [pt[i, 1], pt[j, 1]],
                        color=keypoint_colors[i], linewidth=line_width, alpha=alpha, zorder=step_idx * 2 + 1)

        # Plot keypoint nodes
        for k_idx in range(n_kps):
            ax.scatter(pt[k_idx, 0], pt[k_idx, 1],
                       s=node_size, color=keypoint_colors[k_idx],
                       edgecolors="black", linewidths=0.75, alpha=alpha, zorder=step_idx * 2 + 2)

        rasters.append(rasterize_figure(fig))

    return fig, rasters


# =============================================================================
# Main Trajectory Rendering Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Render Trajectories for Cohort Static 21kp Seed 101")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render (default: render all active >=0.1%)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing PDF and GIF files")
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("🎬 RENDERING COHORT STATIC 21kp SEED 101 TRAJECTORY PLOTS (PURE NUMPY)", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Config & Anatomy
    print("[1/4] Loading project configuration and anatomical skeleton...", flush=True)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    use_bps = config["use_bodyparts"]
    skeleton = config.get("skeleton", [])
    edges_all_tuples = [
        (use_bps.index(e[0]), use_bps.index(e[1]))
        for e in skeleton if e[0] in use_bps and e[1] in use_bps
    ]

    extra_ventral_paw_edges = [
        (17, 18),  # LF - RF
        (19, 20),  # LH - RH
        (17, 19),  # LF - LH
        (18, 20),  # RF - RH
    ]
    edges_super = list(set(edges_all_tuples + extra_ventral_paw_edges))

    dorsal_idxs = list(range(17))
    ventral_idxs = list(range(17, 21))

    edges_dorsal = [(i, j) for (i, j) in edges_all_tuples if i < 17 and j < 17]
    edges_ventral = [(i - 17, j - 17) for (i, j) in extra_ventral_paw_edges]

    colors_super = np.zeros((21, 3), dtype=float)
    colors_super[:17] = plt.cm.Blues(np.linspace(0.45, 0.95, 17))[:, :3]
    colors_super[17] = [0.90, 0.15, 0.15]  # Left Front
    colors_super[18] = [0.90, 0.15, 0.15]  # Right Front
    colors_super[19] = [0.95, 0.50, 0.10]  # Left Hind
    colors_super[20] = [0.95, 0.50, 0.10]  # Right Hind

    colors_dorsal = colors_super[:17]
    colors_ventral = colors_super[17:21]

    # 2. Fast Parameter & Data Loading via HDF5
    print("[2/4] Reading observation matrix Cd and scanning syllable occurrences...", flush=True)
    with h5py.File(CHECKPOINT_PATH, "r") as f:
        it = "250" if "250" in f["model_snapshots"] else sorted(f["model_snapshots"].keys(), key=int)[-1]
        Cd = f[f"model_snapshots/{it}/params/Cd"][:]
        print(f"  Loaded Cd matrix with shape: {Cd.shape} from snapshot {it}", flush=True)

    traj_pre = 5
    traj_post = 20
    num_timesteps = 15

    all_bouts_by_syll = {}
    syll_total_frames = {}
    total_frames_cohort = 0

    with h5py.File(RESULTS_PATH, "r") as f:
        all_keys = list(f.keys())
        for k in all_keys:
            s_arr = f[k]["syllable"][:]
            total_frames_cohort += len(s_arr)
            u, c = np.unique(s_arr, return_counts=True)
            for val, cnt in zip(u, c):
                val = int(val)
                syll_total_frames[val] = syll_total_frames.get(val, 0) + cnt

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

    active_sylls = [
        s for s, cnt in sorted(syll_total_frames.items(), key=lambda x: x[1], reverse=True)
        if (cnt / total_frames_cohort) >= 0.001 and s in all_bouts_by_syll
    ]
    print(f"  Found {len(active_sylls)} active syllables (>=0.1% usage): {active_sylls}", flush=True)

    target_sylls = sorted(args.syllables) if args.syllables is not None else active_sylls
    target_sylls = [s for s in target_sylls if s in all_bouts_by_syll]
    print(f"  Target syllables to render ({len(target_sylls)}): {target_sylls}", flush=True)

    selected_bouts_by_syll = {}
    needed_sessions = set()
    for s in target_sylls:
        bouts = all_bouts_by_syll[s]
        if len(bouts) <= 60:
            chosen = bouts
        else:
            step = len(bouts) / 60.0
            chosen = [bouts[int(i * step)] for i in range(60)]
        selected_bouts_by_syll[s] = chosen
        for sid, _, _ in chosen:
            needed_sessions.add(sid)

    pkl_density_path = OUT_DIR.parent / "density_sampled_instances.pkl"
    pkl_traj_path = OUT_DIR.parent / "sampled_median_trajectories.pkl"

    if pkl_density_path.exists() and pkl_traj_path.exists() and not args.overwrite:
        print(f"  ⚡ Found cached trajectories and density instances! Loading directly...", flush=True)
        with open(pkl_density_path, "rb") as pf:
            sampled_instances = pickle.load(pf)
        with open(pkl_traj_path, "rb") as pf:
            traj_dict = pickle.load(pf)
    else:
        sample_sessions = sorted(list(needed_sessions))
        print(f"  Loading & reconstructing 21kp coordinates for {len(sample_sessions)} targeted sessions...", flush=True)

        centroids = {}
        headings = {}
        coordinates = {}

        with h5py.File(RESULTS_PATH, "r") as f:
            for idx, k in enumerate(sample_sessions):
                grp = f[k]
                x_arr = grp["latent_state"][:]
                v_arr = grp["centroid"][:]
                h_arr = grp["heading"][:]

                arr = estimate_coordinates_numpy(x_arr, v_arr, h_arr, Cd)
                outliers = np.isnan(arr).any(-1)
                clean_coords = interpolate_keypoints(arr, outliers)

                coordinates[k] = clean_coords
                centroids[k] = v_arr
                headings[k] = h_arr

                if (idx + 1) % 25 == 0 or (idx + 1) == len(sample_sessions):
                    print(f"    [{idx+1}/{len(sample_sessions)}] Reconstructed {k}", flush=True)

        print("[3/4] Extracting raw syllable trajectories & running density sampling...", flush=True)
        raw_trajs_by_syll = {}
        for s in target_sylls:
            raw_trajs_by_syll[s] = get_instance_trajectories_numpy(
                selected_bouts_by_syll[s],
                coordinates,
                pre=traj_pre,
                post=traj_post,
                centroids=centroids,
                headings=headings,
                filter_size=9,
            )

        # PCA space fit across instances
        all_traj_list = [t for t in raw_trajs_by_syll.values() if len(t) > 0]
        X_all = np.vstack(all_traj_list)
        pca_samples = min(20000, len(X_all))
        if len(X_all) > pca_samples:
            np.random.seed(101)
            sub_idx = np.random.choice(len(X_all), pca_samples, replace=False)
            X_pca_train = X_all[sub_idx]
        else:
            X_pca_train = X_all

        pca = PCA(n_components=min(4, X_pca_train.shape[1])).fit(X_pca_train.reshape(X_pca_train.shape[0], -1))
        X_all_pca = pca.transform(X_all.reshape(X_all.shape[0], -1))

        global_k = min(50, len(X_all_pca))
        all_nbrs = NearestNeighbors(n_neighbors=global_k).fit(X_all_pca)

        sampled_instances = {}
        for s in target_sylls:
            inst = selected_bouts_by_syll[s]
            N = len(inst)
            if N < 10:
                sampled_instances[s] = inst
                continue

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
            np.random.seed(101 + s)
            chosen_sub_indices = np.random.choice(indices[exemplar_idx], num_to_sample, replace=False)
            sampled_instances[s] = [inst[i] for i in chosen_sub_indices]

        with open(pkl_density_path, "wb") as pf:
            pickle.dump(sampled_instances, pf)

        traj_dict = {}
        for s in target_sylls:
            inst = sampled_instances.get(s, [])
            if len(inst) == 0:
                continue
            ts = get_instance_trajectories_numpy(
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

    # Global axis limits
    stacked_all = np.stack(list(traj_dict.values()))
    lims_super = get_limits(stacked_all, pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_dorsal = get_limits(stacked_all[:, :, dorsal_idxs], pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_ventral = get_limits(stacked_all[:, :, ventral_idxs], pctl=0, left=0.25, right=0.25, top=0.25, bottom=0.25)

    # 4. Render Visualizations
    print(f"\n[4/4] Rendering Superimposed & Side-by-Side PDFs and GIFs for {len(target_sylls)} syllables...", flush=True)

    for idx, s in enumerate(target_sylls):
        if s not in traj_dict:
            continue
        traj = traj_dict[s]
        n_inst = len(sampled_instances.get(s, []))
        sampling_tag = "Density Sampled" if len(all_bouts_by_syll.get(s, [])) >= 10 else f"Rare ({n_inst} bouts)"
        desc = f"Syllable {s:02d}"

        pdf_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.pdf"
        gif_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.gif"
        pdf_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.pdf"
        gif_side = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.gif"

        if not args.overwrite and pdf_super.exists() and gif_super.exists() and pdf_side.exists() and gif_side.exists():
            print(f"  [{idx+1}/{len(target_sylls)}] {desc} already rendered, skipping.", flush=True)
            continue

        # A. Superimposed
        fig_s, rasters_s = plot_trajectories_clean(
            f"{desc} — Superimposed ({sampling_tag})",
            traj,
            lims_super,
            edges=edges_super,
            keypoint_colors=colors_super,
            node_size=42.0,
            line_width=2.6,
            num_timesteps=num_timesteps,
            plot_width=5.2,
        )
        fig_s.savefig(str(pdf_super), bbox_inches="tight", dpi=300)
        plt.close(fig_s)

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

        fig_d, rasters_d = plot_trajectories_clean(
            f"{desc} — Dorsal Spine ({sampling_tag})",
            traj_dorsal,
            lims_dorsal,
            edges=edges_dorsal,
            keypoint_colors=colors_dorsal,
            node_size=42.0,
            line_width=2.6,
            num_timesteps=num_timesteps,
            plot_width=4.8,
        )
        plt.close(fig_d)

        fig_v, rasters_v = plot_trajectories_clean(
            f"{desc} — Ventral Paws ({sampling_tag})",
            traj_ventral,
            lims_ventral,
            edges=edges_ventral,
            keypoint_colors=colors_ventral,
            node_size=55.0,
            line_width=3.2,
            num_timesteps=num_timesteps,
            plot_width=4.8,
        )
        plt.close(fig_v)

        side_rasters = []
        for r_d, r_v in zip(rasters_d, rasters_v):
            h = min(r_d.shape[0], r_v.shape[0])
            w_d = int(round(r_d.shape[1] * (h / r_d.shape[0])))
            w_v = int(round(r_v.shape[1] * (h / r_v.shape[0])))
            img_d = np.array(Image.fromarray(np.uint8(r_d)).resize((w_d, h), Image.Resampling.LANCZOS))
            img_v = np.array(Image.fromarray(np.uint8(r_v)).resize((w_v, h), Image.Resampling.LANCZOS))
            divider = np.full((h, 4, 3), 255, dtype=np.uint8)
            side_rasters.append(np.hstack([img_d, divider, img_v]))

        images_side = [Image.fromarray(r) for r in side_rasters]
        images_side[0].save(
            str(gif_side),
            save_all=True,
            append_images=images_side[1:],
            duration=120,
            loop=0
        )

        fig_tandem, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        ax_left.imshow(rasters_d[-1])
        ax_left.axis("off")
        ax_right.imshow(rasters_v[-1])
        ax_right.axis("off")
        plt.subplots_adjust(wspace=0.04)
        fig_tandem.savefig(str(pdf_side), bbox_inches="tight", dpi=300)
        plt.close(fig_tandem)

        print(f"  [{idx+1}/{len(target_sylls)}] Rendered {desc} (Side-by-Side & Superimposed PDFs/GIFs)", flush=True)

    print("\n✅ All trajectory visualizations completed successfully!", flush=True)


if __name__ == "__main__":
    main()
