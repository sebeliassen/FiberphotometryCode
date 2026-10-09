#!/usr/bin/env python3
"""
Test script to generate 1 superimposed and 1 side-by-side trajectory plot (GIF + PDF)
and verify layout, colors, and rasterization.
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
import matplotlib.pyplot as plt
import imageio
from PIL import Image

import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates

import keypoint_moseq as kpms
import keypoint_moseq.viz as viz
import keypoint_moseq.util as util

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

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project")
MODEL_NAME = "fullmodel_multiview_23kp_k7.5e13"
TEST_OUT = PROJECT_DIR / MODEL_NAME / "test_visualizations"
TEST_OUT.mkdir(parents=True, exist_ok=True)

# 1. Config & Anatomy
config = kpms.load_config(str(PROJECT_DIR))
use_bps = config["use_bodyparts"]
skeleton = config.get("skeleton", [])
edges_all = util.get_edges(use_bps, skeleton) if len(skeleton) > 0 else []

# Additional ventral edges for anatomical clarity
extra_ventral_global = [
    (17, 18), # LF-RF
    (19, 20), # LH-RH
    (17, 19), # LF-LH
    (18, 20), # RF-RH
    (21, 17), # nose-LF
    (21, 18), # nose-RF
    (22, 19), # tail-LH
    (22, 20), # tail-RH
    (21, 22), # nose-tail
]
edges_all_tuples = [tuple(e) for e in edges_all]
edges_super = list(set(edges_all_tuples + extra_ventral_global))

dorsal_idxs = list(range(17))
ventral_idxs = list(range(17, 23))

edges_dorsal = [(i, j) for (i, j) in edges_all if i < 17 and j < 17]
edges_ventral = [(i - 17, j - 17) for (i, j) in edges_super if i >= 17 and j >= 17]

# Color styling:
# Dorsal: cool cyan/blue gradient
# Ventral paws (17-20): vivid red/crimson & orange
# Ventral anchors (21-22): gold
colors_super = np.zeros((23, 3), dtype=float)
cmap_cool = plt.cm.Blues(np.linspace(0.4, 0.9, 17))[:, :3]
colors_super[:17] = cmap_cool
colors_super[17] = [0.90, 0.15, 0.15] # Left front paw: bright red
colors_super[18] = [0.90, 0.15, 0.15] # Right front paw: bright red
colors_super[19] = [0.95, 0.50, 0.10] # Left hind paw: orange
colors_super[20] = [0.95, 0.50, 0.10] # Right hind paw: orange
colors_super[21] = [0.85, 0.70, 0.10] # Bot nose: gold
colors_super[22] = [0.85, 0.70, 0.10] # Bot tail: gold

colors_dorsal = colors_super[:17]
colors_ventral = colors_super[17:]

# 2. Load model & estimate coordinates
print("Loading model...")
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

# 3. Extract trajectories for test syllables (52: locomotion, 79: scanning)
test_sylls = [52, 79]
traj_pre = 5
traj_post = 20
num_timesteps = 15

all_instances = util.get_syllable_instances(
    syllables,
    pre=traj_pre,
    post=traj_post,
    min_duration=2,
    min_frequency=0.0,
    min_instances=1,
)

traj_dict = {}
for s in test_sylls:
    inst = all_instances.get(s, [])
    sample_inst = inst[:50]
    ts = util.get_instance_trajectories(
        sample_inst,
        clean_coords,
        pre=traj_pre,
        post=traj_post,
        centroids=centroids,
        headings=headings,
    )
    # Median across instances, invert y for display
    traj_dict[s] = np.nanmedian(ts, axis=0) * np.array([1, -1])

# Global limits
stacked_all = np.stack(list(traj_dict.values()))
lims_super = viz.get_limits(stacked_all, pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
lims_dorsal = viz.get_limits(stacked_all[:, :, dorsal_idxs], pctl=0, left=0.15, right=0.15, top=0.20, bottom=0.20)
lims_ventral = viz.get_limits(stacked_all[:, :, ventral_idxs], pctl=0, left=0.25, right=0.25, top=0.25, bottom=0.25)

print("Rendering test visualizations...")
for s in test_sylls:
    traj = traj_dict[s] # shape (T, 23, 2)
    
    # -------------------------------------------------------------
    # A. SUPERIMPOSED PLOT (Top + Bot fused)
    # -------------------------------------------------------------
    title_super = f"Syllable {s} (Superimposed Multi-View)"
    fig_s, ax_s, rasters_s = viz.plot_trajectories(
        [title_super],
        traj[None],
        lims_super,
        edges=edges_super,
        keypoint_colors=colors_super,
        return_rasters=True,
        num_timesteps=num_timesteps,
        node_size=40.0,
        line_width=2.5,
        alpha=0.15,
        plot_width=5.0,
    )
    pdf_super = TEST_OUT / f"syllable_{s:02d}_superimposed.pdf"
    fig_s.savefig(str(pdf_super), bbox_inches="tight")
    plt.close(fig_s)

    gif_super = TEST_OUT / f"syllable_{s:02d}_superimposed.gif"
    imageio.mimsave(str(gif_super), [np.uint8(r) for r in rasters_s], duration=0.08, loop=0)
    print(f"Saved superimposed: {pdf_super.name} & {gif_super.name}")

    # -------------------------------------------------------------
    # B. SIDE-BY-SIDE PLOT (Dorsal on left, Ventral on right)
    # -------------------------------------------------------------
    title_d = f"Syllable {s} (Dorsal TopCam)"
    traj_d = traj[:, dorsal_idxs]
    fig_d, ax_d, rasters_d = viz.plot_trajectories(
        [title_d],
        traj_d[None],
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

    title_v = f"Syllable {s} (Ventral BotCam Paws)"
    traj_v = traj[:, ventral_idxs]
    fig_v, ax_v, rasters_v = viz.plot_trajectories(
        [title_v],
        traj_v[None],
        lims_ventral,
        edges=edges_ventral,
        keypoint_colors=colors_ventral,
        return_rasters=True,
        num_timesteps=num_timesteps,
        node_size=45.0,
        line_width=2.5,
        alpha=0.15,
        plot_width=4.5,
    )

    # Combined PDF
    fig_combo, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
    ax_left.axis("off")
    ax_right.axis("off")
    # Draw last rasters
    ax_left.imshow(rasters_d[-1])
    ax_right.imshow(rasters_v[-1])
    fig_combo.suptitle(f"Syllable {s}: Multi-View Kinematic Decomposition", fontsize=14, fontweight="bold", y=0.98)
    pdf_tandem = TEST_OUT / f"syllable_{s:02d}_tandem_side_by_side.pdf"
    fig_combo.savefig(str(pdf_tandem), bbox_inches="tight", dpi=300)
    plt.close(fig_combo)
    plt.close(fig_d)
    plt.close(fig_v)

    # Combined animated GIF (side-by-side horizontal stack)
    # Match heights if needed
    h_d, w_d = rasters_d[0].shape[:2]
    h_v, w_v = rasters_v[0].shape[:2]
    target_h = min(h_d, h_v)
    
    combo_rasters = []
    for rd, rv in zip(rasters_d, rasters_v):
        # Resize to matching height
        im_d = Image.fromarray(np.uint8(rd))
        im_v = Image.fromarray(np.uint8(rv))
        im_d = im_d.resize((int(w_d * target_h / h_d), target_h), Image.Resampling.BILINEAR)
        im_v = im_v.resize((int(w_v * target_h / h_v), target_h), Image.Resampling.BILINEAR)
        combo = np.hstack([np.asarray(im_d), np.asarray(im_v)])
        combo_rasters.append(combo)

    gif_tandem = TEST_OUT / f"syllable_{s:02d}_tandem_side_by_side.gif"
    imageio.mimsave(str(gif_tandem), combo_rasters, duration=0.08, loop=0)
    print(f"Saved side-by-side: {pdf_tandem.name} & {gif_tandem.name}")

print("Verification complete.")
