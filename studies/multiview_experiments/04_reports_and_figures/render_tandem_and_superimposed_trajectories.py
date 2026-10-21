#!/usr/bin/env python3
"""
render_tandem_and_superimposed_trajectories.py
=============================================================================
Generates publication-quality Trajectory Visualizations for Multi-View KPMS:
1. Superimposed Plots (TopCam Dorsal + BotCam Ventral fused in shared frame)
   - Publication-quality vector PDF
   - True looping animated GIF (15 timesteps)
2. Side-by-Side Plots (Dorsal on Left with lims_dorsal | Ventral on Right with lims_ventral)
   - Publication-quality vector PDF
   - True looping animated GIF (15 timesteps)

Key Features:
- Preserves anatomical left/right chirality (no lateral flip needed).
- Zoomed ventral axis limits so paw kinematics and stride phases are not squished.
- Distinct color coding: cool blue/slate dorsal spine vs. vivid coral/orange paws.
- Uses kpms.viz built-in routines with modern Matplotlib buffer-copy patch.
=============================================================================
"""
import os

from fpcode.common.paths import get_repo_root

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib

matplotlib.use("Agg")
import imageio
import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
import matplotlib.pyplot as plt
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from PIL import Image


# --- 1. Modern Matplotlib Rasterize Patch (ensuring deep buffer copy for animations) ---
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

# --- 2. Paths & Model Definition ---
REPO_ROOT = Path(str(get_repo_root()))
PROJECT_DIR = REPO_ROOT / 'studies/multiview_experiments/multiview_kappa_search/project'
MODEL_NAME = "fullmodel_multiview_23kp_k7.5e13"

OUT_DIR = REPO_ROOT / 'studies/multiview_experiments/trajectory_visualizations'
SUPER_DIR = OUT_DIR / "superimposed"
SIDE_DIR = OUT_DIR / "side_by_side"
SUPER_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)

# Key behavioral syllables to render (locomotion, scanning, turning, rearing, grooming, pivot)
TARGET_SYLLABLES = [79, 52, 15, 4, 55, 98, 84, 48, 59, 26, 6]

SYLLABLE_LABELS = {
    79: "Syllable 79: Low-Velocity Head Scanning / Pause",
    52: "Syllable 52: Baseline Forward Locomotion / Walk",
    15: "Syllable 15: Sustained Exploratory Locomotion",
    4:  "Syllable 04: Active Sniffing & Turning",
    55: "Syllable 55: Rear-Supported Wall Rearing",
    98: "Syllable 98: Postural Reorientation / Pause",
    84: "Syllable 84: Locomotion Initiation / Stride Acceleration",
    48: "Syllable 48: Behavioral Pause & Micro-Grooming",
    59: "Syllable 59: Intermediate Stride Adaptation",
    26: "Syllable 26: Rhythmic Bilateral Facial/Paw Grooming",
    6:  "Syllable 06: Sharp Lateral Pivot / Heading Shift",
}

def main():
    print("=" * 80)
    print("🎬 RENDERING TANDEM MULTI-VIEW TRAJECTORY PLOTS & ANIMATED GIFS")
    print(f"Model: {MODEL_NAME}")
    print(f"Target Syllables ({len(TARGET_SYLLABLES)}): {TARGET_SYLLABLES}")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 80)

    # 1. Config & Anatomy
    print("[1/4] Loading project configuration and anatomical skeleton...", flush=True)
    config = kpms.load_config(str(PROJECT_DIR))
    use_bps = config["use_bodyparts"]
    skeleton = config.get("skeleton", [])
    edges_all = util.get_edges(use_bps, skeleton) if len(skeleton) > 0 else []
    edges_all_tuples = [tuple(e) for e in edges_all]

    # Additional ventral connections to form an anatomical ventral paw girdle
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

    # Curated Anatomical Color Palette
    # Dorsal (0-16): Gradient of rich oceanic blues and teals
    # Ventral Front Paws (17, 18): High-visibility crimson red (#E53E3E)
    # Ventral Hind Paws (19, 20): High-visibility warm amber (#DD6B20)
    # Ventral Body Anchors (21, 22): Muted gold (#D69E2E)
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

    # 2. Load Checkpoint and Reconstruct Latent Coordinates
    print("[2/4] Loading model checkpoint & unbatching coordinates...", flush=True)
    model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
    results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}
    syllables = {k: v["syllable"] for k, v in results.items()}

    clean_coords = {}
    for key in coordinates.keys():
        arr = coordinates[key].copy()
        outliers = np.isnan(arr).any(-1)
        clean_coords[key] = util.interpolate_keypoints(arr, outliers)

    # 3. Compute Representative Trajectories per Syllable
    print("[3/4] Extracting representative trajectories for target syllables...", flush=True)
    traj_pre = 5
    traj_post = 20
    num_timesteps = 15 # 15-frame smooth animation

    all_instances = util.get_syllable_instances(
        syllables,
        pre=traj_pre,
        post=traj_post,
        min_duration=2,
        min_frequency=0.0,
        min_instances=1,
    )

    traj_dict = {}
    for s in TARGET_SYLLABLES:
        inst = all_instances.get(s, [])
        if len(inst) == 0:
            print(f"  [WARN] No instances found for Syllable {s}, skipping.")
            continue
        sample_inst = inst[:50] # median over up to 50 representative bouts
        ts = util.get_instance_trajectories(
            sample_inst,
            clean_coords,
            pre=traj_pre,
            post=traj_post,
            centroids=centroids,
            headings=headings,
        )
        # Median across instances, invert y for standard display coordinates
        traj_dict[s] = np.nanmedian(ts, axis=0) * np.array([1, -1])

    # Global axis limits across all target syllables
    stacked_all = np.stack(list(traj_dict.values()))
    lims_super = viz.get_limits(stacked_all, pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_dorsal = viz.get_limits(stacked_all[:, :, dorsal_idxs], pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
    lims_ventral = viz.get_limits(stacked_all[:, :, ventral_idxs], pctl=0, left=0.25, right=0.25, top=0.25, bottom=0.25)

    # 4. Render Visualizations
    print(f"[4/4] Rendering Superimposed & Side-by-Side PDFs and GIFs for {len(traj_dict)} syllables...", flush=True)

    for s in TARGET_SYLLABLES:
        if s not in traj_dict:
            continue
        traj = traj_dict[s]
        desc = SYLLABLE_LABELS.get(s, f"Syllable {s}")

        # =========================================================================
        # A. SUPERIMPOSED TRAJECTORY PLOT (TOP + BOT FUSED)
        # =========================================================================
        fig_s, ax_s, rasters_s = viz.plot_trajectories(
            [desc],
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

        # Save vector PDF
        pdf_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.pdf"
        fig_s.savefig(str(pdf_super), bbox_inches="tight", dpi=300)
        plt.close(fig_s)

        # Save animated looping GIF
        gif_super = SUPER_DIR / f"syllable_{s:02d}_superimposed.gif"
        frames_super = [np.uint8(r) for r in rasters_s]
        imageio.mimsave(str(gif_super), frames_super, duration=0.08, loop=0)

        # =========================================================================
        # B. SIDE-BY-SIDE TANDEM TRAJECTORY PLOT (DORSAL | VENTRAL)
        # =========================================================================
        # Render Dorsal TopCam Panel
        fig_d, ax_d, rasters_d = viz.plot_trajectories(
            ["Dorsal View (TopCam 17 KPs)"],
            traj[:, dorsal_idxs][None],
            lims_dorsal,
            edges=edges_dorsal,
            keypoint_colors=colors_dorsal,
            return_rasters=True,
            num_timesteps=num_timesteps,
            node_size=40.0,
            line_width=2.5,
            alpha=0.15,
            plot_width=4.5,
        )

        # Render Ventral BotCam Panel (with zoomed lims_ventral)
        fig_v, ax_v, rasters_v = viz.plot_trajectories(
            ["Ventral Paws (BotCam 6 KPs)"],
            traj[:, ventral_idxs][None],
            lims_ventral,
            edges=edges_ventral,
            keypoint_colors=colors_ventral,
            return_rasters=True,
            num_timesteps=num_timesteps,
            node_size=48.0,
            line_width=2.8,
            alpha=0.15,
            plot_width=4.5,
        )

        # Create Unified Vector PDF
        fig_combo, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
        ax_left.axis("off")
        ax_right.axis("off")
        ax_left.imshow(rasters_d[-1])
        ax_right.imshow(rasters_v[-1])
        fig_combo.suptitle(desc, fontsize=13, fontweight="bold", y=0.98)

        pdf_tandem = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.pdf"
        fig_combo.savefig(str(pdf_tandem), bbox_inches="tight", dpi=300)
        plt.close(fig_combo)
        plt.close(fig_d)
        plt.close(fig_v)

        # Create Unified Looping Animated GIF
        h_d, w_d = rasters_d[0].shape[:2]
        h_v, w_v = rasters_v[0].shape[:2]
        target_h = min(h_d, h_v)

        combo_rasters = []
        for rd, rv in zip(rasters_d, rasters_v):
            im_d = Image.fromarray(np.uint8(rd))
            im_v = Image.fromarray(np.uint8(rv))
            im_d = im_d.resize((int(w_d * target_h / h_d), target_h), Image.Resampling.BILINEAR)
            im_v = im_v.resize((int(w_v * target_h / h_v), target_h), Image.Resampling.BILINEAR)
            combo = np.hstack([np.asarray(im_d), np.asarray(im_v)])
            combo_rasters.append(combo)

        gif_tandem = SIDE_DIR / f"syllable_{s:02d}_tandem_side_by_side.gif"
        imageio.mimsave(str(gif_tandem), combo_rasters, duration=0.08, loop=0)

        # Verify animation
        im_check_s = Image.open(str(gif_super))
        im_check_t = Image.open(str(gif_tandem))
        print(f"  [DONE] Syllable {s:02d} | Superimposed: {im_check_s.n_frames}f GIF + PDF | Side-by-Side: {im_check_t.n_frames}f GIF + PDF")

    print("\n" + "=" * 80)
    print("🎉 ALL MULTI-VIEW TRAJECTORY PLOTS & GIFS GENERATED SUCCESSFULLY!")
    print(f"   Superimposed Outputs: {SUPER_DIR}")
    print(f"   Side-by-Side Outputs: {SIDE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
