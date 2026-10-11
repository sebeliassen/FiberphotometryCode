#!/usr/bin/env python3
"""
render_seed303_trajectories.py
=============================================================================
Publication-grade Trajectory Visualizations for Champion Multi-View KPMS Seed 303:
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
import h5py
import imageio
import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
import matplotlib.pyplot as plt
from jax_moseq.models.keypoint_slds import estimate_coordinates
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

# --- 2. Paths & Directories ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed303"
MODEL_DIR = PROJECT_DIR / MODEL_NAME
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.h5"
RESULTS_PATH = MODEL_DIR / "results.h5"

OUT_DIR = SCRIPT_DIR / "visualizations_seed303" / "trajectories"
SUPER_DIR = OUT_DIR / "superimposed"
SIDE_DIR = OUT_DIR / "side_by_side"
SUPER_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)

# Top 10 syllables for Seed 303 (highest usage in cohort)
TARGET_SYLLABLES = [2, 10, 6, 0, 8, 5, 4, 3, 1, 13]

def main():
    print("=" * 80)
    print("🎬 RENDERING TANDEM MULTI-VIEW TRAJECTORY PLOTS & ANIMATED GIFS (SEED 303)")
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

    # 2. Fast Parameter & Data Loading via HDF5
    print("[2/4] Reading observation matrix Cd and session results...", flush=True)
    with h5py.File(CHECKPOINT_PATH, "r") as f:
        # Load Cd from iteration 250 snapshot
        Cd = f["model_snapshots/250/params/Cd"][:]
        Cd_jax = jnp.array(Cd)
        print(f"  Loaded Cd matrix with shape: {Cd.shape}")

    # To be extremely memory efficient, select a representative subset of sessions
    sample_sessions = []
    with h5py.File(RESULTS_PATH, "r") as f:
        all_keys = list(f.keys())
        sample_sessions = all_keys[:25] # 25 diverse sessions provide ample instances for all syllables
        print(f"  Sampling bouts from {len(sample_sessions)} sessions...")

        centroids = {}
        headings = {}
        syllables = {}
        coordinates = {}

        for k in sample_sessions:
            grp = f[k]
            x_arr = grp["latent_state"][:]
            v_arr = grp["centroid"][:]
            h_arr = grp["heading"][:]
            s_arr = grp["syllable"][:]

            # Reconstruct 23-keypoint coordinates on the fly
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
            syllables[k] = s_arr

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
            coordinates,
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
        desc = f"Syllable {s:02d}"

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
        print(f"  ✅ Syllable {s:02d} | Superimposed: {im_check_s.n_frames}f GIF + PDF | Side-by-Side: {im_check_t.n_frames}f GIF + PDF")

    print("\n" + "=" * 80)
    print("🎉 ALL MULTI-VIEW TRAJECTORY PLOTS & GIFS GENERATED SUCCESSFULLY!")
    print(f"   Superimposed: {SUPER_DIR}")
    print(f"   Side-by-Side: {SIDE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
