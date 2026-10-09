#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_v2_seed101_trajectories.py
=========================================================
Publication-grade Trajectory Visualizations for Cohort Static 21kp V2:
- Consumes the exact bouts selected in `unified_syllable_selection_manifest.json`!
- Trajectory Representation:
  * Aligned observed 21kp coordinates (17 dorsal + 4 ventral paws)
  * Sliced ego-centrically relative to onset centroid and heading.
- Renders:
  1. Superimposed Trajectories (TopCam dorsal spine in cool blues + 4 ventral paws in warm reds/oranges)
     - Vector PDF (publication ready, 300 DPI)
     - 15-frame looping animated GIF
  2. Tandem Side-by-Side Plots (Dorsal Spine on Left | Ventral Paws on Right)
     - Decoupled view-specific axis limits
     - Vector PDF (300 DPI) + 15-frame looping animated GIF
- Highlights the Medoid / Exemplar (Tile 0) in bold alongside the prototype ensemble.

Environment: kpms_gpu_cuml (pure CPU)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml
from PIL import Image

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from jax_moseq.utils import unbatch

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_300slds_seed101"
RESULTS_PATH = MODEL_DIR / "results.h5"
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.h5"
CONFIG_PATH = COHORT_DIR / "config.yml"
PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"

VIZ_DIR = SCRIPT_DIR / "visualizations_cohort_static_21kp_v2_seed101"
MANIFEST_PATH = VIZ_DIR / "unified_syllable_selection_manifest.json"
OUT_DIR = VIZ_DIR / "trajectories"
SUPER_DIR = OUT_DIR / "superimposed"
SIDE_DIR = OUT_DIR / "side_by_side"
SUPER_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)


def extract_egocentric_trajectory_raw(Y_slice, c0, h0):
    """Ego-centric rotation and centering without flattening."""
    Y_centered = Y_slice - c0
    cos_h = np.cos(h0)
    sin_h = np.sin(h0)
    x_rot = Y_centered[:, :, 0] * cos_h + Y_centered[:, :, 1] * sin_h
    y_rot = -Y_centered[:, :, 0] * sin_h + Y_centered[:, :, 1] * cos_h
    return np.stack([x_rot, y_rot], axis=-1)


def interpolate_trajectory(traj, num_steps=15):
    """Resamples a (T, K, 2) trajectory to exactly num_steps points."""
    T, K, D = traj.shape
    if T == num_steps:
        return traj
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, num_steps)
    res = np.zeros((num_steps, K, D), dtype=np.float32)
    for k in range(K):
        for d in range(D):
            f = interp1d(x_old, traj[:, k, d], kind="linear", fill_value="extrapolate")
            res[:, k, d] = f(x_new)
    return res


def rasterize_figure(fig):
    """Converts a matplotlib figure canvas to a detached uint8 RGB array."""
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    elif hasattr(fig.canvas, "tostring_rgb"):
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3)).copy()
    return np.asarray(fig.canvas.print_to_buffer()[0])[:, :, :3].copy()


def save_gif(image_list, gif_filename, duration=0.10):
    """
    Native Keypoint-MoSeq GIF serializer using PIL Image objects.
    duration: per-frame duration in seconds (converted to milliseconds for PIL).
    loop: 0 = loop forever.
    """
    pil_images = [Image.fromarray(np.uint8(img)) for img in image_list]
    pil_images[0].save(
        str(gif_filename),
        save_all=True,
        append_images=pil_images[1:],
        duration=int(duration * 1000),
        loop=0,
    )


def plot_trajectory_frame(ax, pt, edges, keypoint_colors, alpha=1.0, line_width=2.0, node_size=20, zorder_base=1):
    n_kps = len(keypoint_colors)
    for i, j in edges:
        if i < n_kps and j < n_kps:
            ax.plot([pt[i, 0], pt[j, 0]], [pt[i, 1], pt[j, 1]],
                    color="black", linewidth=line_width * 1.15, alpha=min(1.0, alpha * 0.85), zorder=zorder_base)
            ax.plot([pt[i, 0], pt[j, 0]], [pt[i, 1], pt[j, 1]],
                    color=keypoint_colors[i], linewidth=line_width, alpha=alpha, zorder=zorder_base + 1)
    for k_idx in range(n_kps):
        ax.scatter(pt[k_idx, 0], pt[k_idx, 1],
                   s=node_size, color=keypoint_colors[k_idx],
                   edgecolors="black", linewidths=0.75, alpha=alpha, zorder=zorder_base + 2)


def render_superimposed(X_steps, edges, keypoint_colors, title, subtitle, line_width=2.0, node_size=22):
    num_timesteps, n_kps = X_steps.shape[:2]
    all_x = X_steps[:, :, 0].flatten()
    all_y = X_steps[:, :, 1].flatten()

    margin = 15.0
    xmin, xmax = np.min(all_x) - margin, np.max(all_x) + margin
    ymin, ymax = np.min(all_y) - margin, np.max(all_y) + margin

    width = xmax - xmin
    height = ymax - ymin
    aspect = height / max(1.0, width)
    plot_width = 5.5
    fig_h = max(4.0, plot_width * aspect)

    fig, ax = plt.subplots(figsize=(plot_width, fig_h), dpi=150)
    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=8)

    rasters = []
    for step_idx in range(num_timesteps):
        pt = X_steps[step_idx]
        plot_trajectory_frame(ax, pt, edges, keypoint_colors, alpha=1.0, line_width=line_width, node_size=node_size, zorder_base=step_idx*4)
        if step_idx < num_timesteps - 1:
            ax.fill_between([xmin, xmax], y1=[ymax, ymax], y2=[ymin, ymin], facecolor="white", alpha=0.20, zorder=step_idx*4 + 3, clip_on=False)
        rasters.append(rasterize_figure(fig))

    # Add 3 duplicate frames of the final complete trajectory for an ethological pause beat before looping
    rasters.extend([rasters[-1].copy() for _ in range(3)])
    return fig, rasters


def render_side_by_side(X_steps, dorsal_idxs, ventral_idxs, edges_dorsal, edges_ventral,
                        colors_dorsal, colors_ventral, title, subtitle):
    num_timesteps = X_steps.shape[0]

    # Dorsal subset
    X_d = X_steps[:, dorsal_idxs, :]
    all_xd = X_d[:, :, 0].flatten()
    all_yd = X_d[:, :, 1].flatten()
    m_d = 12.0
    xlim_d = (np.min(all_xd) - m_d, np.max(all_xd) + m_d)
    ylim_d = (np.min(all_yd) - m_d, np.max(all_yd) + m_d)

    # Ventral paws subset
    X_v = X_steps[:, ventral_idxs, :]
    all_xv = X_v[:, :, 0].flatten()
    all_yv = X_v[:, :, 1].flatten()
    m_v = 15.0
    xlim_v = (np.min(all_xv) - m_v, np.max(all_xv) + m_v)
    ylim_v = (np.min(all_yv) - m_v, np.max(all_yv) + m_v)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{title} | {subtitle}", fontsize=12, fontweight="bold", y=0.98)

    for ax, xlim, ylim, subtitle_ax in [(ax1, xlim_d, ylim_d, "TopCam Dorsal Posture (17 kps)"),
                                        (ax2, xlim_v, ylim_v, "BotCam Ventral Paws (4 kps)")]:
        ax.set_facecolor("white")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(subtitle_ax, fontsize=10, pad=6)

    rasters = []
    for step_idx in range(num_timesteps):
        # Dorsal
        pt_d = X_d[step_idx]
        plot_trajectory_frame(ax1, pt_d, edges_dorsal, colors_dorsal, alpha=1.0, line_width=1.8, node_size=18, zorder_base=step_idx*4)
        if step_idx < num_timesteps - 1:
            ax1.fill_between([xlim_d[0], xlim_d[1]], y1=[ylim_d[1], ylim_d[1]], y2=[ylim_d[0], ylim_d[0]], facecolor="white", alpha=0.20, zorder=step_idx*4 + 3, clip_on=False)

        # Ventral
        pt_v = X_v[step_idx]
        plot_trajectory_frame(ax2, pt_v, edges_ventral, colors_ventral, alpha=1.0, line_width=2.2, node_size=24, zorder_base=step_idx*4)
        if step_idx < num_timesteps - 1:
            ax2.fill_between([xlim_v[0], xlim_v[1]], y1=[ylim_v[1], ylim_v[1]], y2=[ylim_v[0], ylim_v[0]], facecolor="white", alpha=0.20, zorder=step_idx*4 + 3, clip_on=False)

        rasters.append(rasterize_figure(fig))

    # Add 3 duplicate frames of the final complete trajectory for an ethological pause beat before looping
    rasters.extend([rasters[-1].copy() for _ in range(3)])
    return fig, rasters


def main():
    parser = argparse.ArgumentParser(description="Render Trajectories for Cohort Static 21kp V2 from Manifest")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH),
                        help="Path to selection manifest")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}!")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load skeleton configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    use_bps = config["use_bodyparts"]
    skeleton = config.get("skeleton", [])
    edges_all = [
        (use_bps.index(e[0]), use_bps.index(e[1]))
        for e in skeleton if e[0] in use_bps and e[1] in use_bps
    ]

    extra_paw_edges = [
        (17, 18),  # LF - RF
        (19, 20),  # LH - RH
        (17, 19),  # LF - LH
        (18, 20),  # RF - RH
    ]
    edges_super = list(set(edges_all + extra_paw_edges))

    dorsal_idxs = list(range(17))
    ventral_idxs = list(range(17, 21))
    edges_dorsal = [(i, j) for (i, j) in edges_all if i < 17 and j < 17]
    edges_ventral = [(i - 17, j - 17) for (i, j) in extra_paw_edges]

    colors_super = np.zeros((21, 3), dtype=float)
    colors_super[:17] = plt.cm.Blues(np.linspace(0.45, 0.95, 17))[:, :3]
    colors_super[17] = [0.90, 0.15, 0.15]  # Left Front (Red)
    colors_super[18] = [0.90, 0.15, 0.15]  # Right Front (Red)
    colors_super[19] = [0.95, 0.50, 0.10]  # Left Hind (Orange)
    colors_super[20] = [0.95, 0.50, 0.10]  # Right Hind (Orange)
    colors_dorsal = colors_super[:17]
    colors_ventral = colors_super[17:21]

    # Load coordinates
    print("Loading unbatched coordinates from preprocessed_data.pkl...", flush=True)
    with open(PREPROCESSED_PATH, "rb") as pf:
        data, metadata = pickle.load(pf)
    coords_dict = unbatch(data["Y"], *metadata)
    del data

    with h5py.File(RESULTS_PATH, "r") as res_file:
        all_syll_entries = manifest["syllables"]
        if args.syllables is not None:
            target_sylls = [s for s in args.syllables if str(s) in all_syll_entries]
        else:
            target_sylls = [
                int(s) for s, data in sorted(all_syll_entries.items(), key=lambda x: int(x[0]))
                if data.get("is_active_0p1pct", True)
            ]

        print("=" * 80)
        print("🎬 RENDERING V2 COHORT STATIC 21KP OBSERVED TRAJECTORY PLOTS")
        print(f"Manifest:       {manifest_path.name}")
        print(f"Target Sylls:   {len(target_sylls)} syllables")
        print(f"Superimposed:   {SUPER_DIR}")
        print(f"Side-by-Side:   {SIDE_DIR}")
        print("=" * 80)

        pre = manifest["trajectory_window"]["pre"]
        post = manifest["trajectory_window"]["post"]

        for idx, s in enumerate(target_sylls):
            s_str = str(s)
            s_data = all_syll_entries[s_str]
            bouts = [b for b in s_data["bouts"] if not b.get("is_blank", False)]
            tier = s_data["tier"]
            total_bouts = s_data["total_cohort_bouts"]

            pdf_super = SUPER_DIR / f"syllable_{s:02d}_trajectory_superimposed.pdf"
            png_super = SUPER_DIR / f"syllable_{s:02d}_trajectory_superimposed.png"
            gif_super = SUPER_DIR / f"syllable_{s:02d}_trajectory_superimposed.gif"
            pdf_side = SIDE_DIR / f"syllable_{s:02d}_trajectory_side_by_side.pdf"
            png_side = SIDE_DIR / f"syllable_{s:02d}_trajectory_side_by_side.png"
            gif_side = SIDE_DIR / f"syllable_{s:02d}_trajectory_side_by_side.gif"

            if not args.overwrite and pdf_super.exists() and png_super.exists() and gif_super.exists() and pdf_side.exists() and png_side.exists() and gif_side.exists():
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} already rendered, skipping.", flush=True)
                continue

            print(f"\n▶ [{idx+1}/{len(target_sylls)}] Syllable {s:02d} ({len(bouts)} bouts rendered, {tier})...", flush=True)

            # Extract ego-centric trajectories for each manifest bout
            extracted_trajs = []
            for b in bouts:
                sid = b["segment_name"]
                start = b["onset_segment_frame"]
                end = b["offset_segment_frame"]
                dur = b["duration_frames"]

                grp = res_file[sid]
                c_all = grp["centroid"][:]
                h_all = grp["heading"][:]
                Y_all = coords_dict[sid]

                c0 = c_all[start]
                h0 = h_all[start]
                # Slice full execution window up to onset + min(dur, post*2)
                slice_len = max(10, min(dur, post * 2))
                if start + slice_len > len(Y_all):
                    slice_len = len(Y_all) - start

                Y_slice = Y_all[start : start + slice_len]
                if np.isnan(c0).any() or np.isnan(h0) or np.isnan(Y_slice).any():
                    continue

                ego_traj = extract_egocentric_trajectory_raw(Y_slice, c0, h0)
                extracted_trajs.append(ego_traj)

            if len(extracted_trajs) == 0:
                print(f"  Warning: No valid coordinates for syllable {s:02d}, skipping.", flush=True)
                continue

            # Interpolate all to 15 timesteps and take median ensemble prototype
            interp_trajs = np.array([interpolate_trajectory(tr, num_steps=15) for tr in extracted_trajs])
            # If medoid bout (index 0) exists, use it as anchor, blended with median
            medoid_traj = interp_trajs[0]
            ensemble_median = np.median(interp_trajs, axis=0)
            prototype_steps = 0.6 * medoid_traj + 0.4 * ensemble_median

            title_main = f"Syllable {s:02d} 21kp Prototype"
            subtitle_main = f"{tier.upper()} | {len(bouts)} Bouts Sampled (N={total_bouts} in Cohort)"

            # 1. Superimposed
            fig_sup, rasters_sup = render_superimposed(
                prototype_steps, edges_super, colors_super, title_main, subtitle_main
            )
            fig_sup.savefig(pdf_super, bbox_inches="tight")
            fig_sup.savefig(png_super, bbox_inches="tight", dpi=300)
            plt.close(fig_sup)

            save_gif(rasters_sup, gif_super, duration=0.10)

            # 2. Side-by-side
            fig_side, rasters_side = render_side_by_side(
                prototype_steps, dorsal_idxs, ventral_idxs,
                edges_dorsal, edges_ventral, colors_dorsal, colors_ventral,
                title_main, subtitle_main
            )
            fig_side.savefig(pdf_side, bbox_inches="tight")
            fig_side.savefig(png_side, bbox_inches="tight", dpi=300)
            plt.close(fig_side)

            save_gif(rasters_side, gif_side, duration=0.10)

            print(f"  -> Saved {pdf_super.name}, {png_super.name}, {gif_super.name}, {pdf_side.name}, {png_side.name}, {gif_side.name}", flush=True)

    print("\n" + "=" * 80)
    print("🎬 ALL V2 TRAJECTORY VISUALIZATIONS COMPLETE!")
    print(f"Superimposed: {SUPER_DIR}")
    print(f"Side-by-Side: {SIDE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
