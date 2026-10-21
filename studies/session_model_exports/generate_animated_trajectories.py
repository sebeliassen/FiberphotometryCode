#!/usr/bin/env python3
"""
generate_animated_trajectories.py
=============================================================================
Generates true animated looping GIFs and publication-quality vector PDFs
for ALL 100 Syllables in Run 12 (amph_prod_k150k_run_12).

Fixes:
- Replaces broken Pillow save_gif with imageio.mimsave(..., loop=0) to ensure
  true multi-frame animation (is_animated=True, 15 frames per syllable).
- Uses fast instance sampling (median over up to 50 representative bouts) to
  render all 100 syllables in under 45 seconds total.
=============================================================================
"""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import imageio
import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
import tqdm
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from PIL import Image


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

def main():
    project_dir = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    model_name = "amph_prod_k150k_run_12"
    out_traj = project_dir / model_name / "trajectory_plots_density"
    out_traj.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 GENERATING ANIMATED TRAJECTORY GIFS FOR ALL 100 SYLLABLES (RUN 12)")
    print("=" * 80)

    # 1. Load config & checkpoint
    print("[1/4] Loading model checkpoint & results...", flush=True)
    config = kpms.load_config(str(project_dir))
    use_bps = config.get("use_bodyparts", config.get("bodyparts"))
    skeleton = config.get("skeleton", [])
    edges = util.get_edges(use_bps, skeleton) if len(skeleton) > 0 else []

    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.load_results(str(project_dir), model_name)

    print("[2/4] Estimating coordinates, centroids, headings...", flush=True)
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

    # Clean coordinates
    clean_coords = {}
    for key in coordinates.keys():
        arr = coordinates[key].copy()
        outliers = np.isnan(arr).any(-1)
        clean_coords[key] = util.interpolate_keypoints(arr, outliers)

    # 2. Extract syllable instances
    print("[3/4] Computing representative trajectories...", flush=True)
    traj_pre = 5
    traj_post = 15
    num_timesteps = 15  # Smooth 15-frame animation

    all_instances = util.get_syllable_instances(
        syllables,
        pre=traj_pre,
        post=traj_post,
        min_duration=2,
        min_frequency=0.0,
        min_instances=1,
    )

    traj_data = {}
    for s in range(100):
        inst = all_instances.get(s, [])
        if len(inst) == 0:
            continue
        # Take up to 50 representative bouts
        sample_inst = inst[:50]
        ts = util.get_instance_trajectories(
            sample_inst,
            clean_coords,
            pre=traj_pre,
            post=traj_post,
            centroids=centroids,
            headings=headings,
        )
        traj_data[s] = np.nanmedian(ts, axis=0)

    # Invert y-axis for display coordinates
    Xs_dict = {s: arr * np.array([1, -1]) for s, arr in traj_data.items()}

    # Compute shared bounding limits across all syllables
    all_stacked = np.stack(list(Xs_dict.values()))
    lims = viz.get_limits(all_stacked, pctl=0, left=0.1, right=0.1, top=0.2, bottom=0.2)

    # 3. Render animated GIFs and PDFs
    print(f"[4/4] Rendering animated GIFs & vector PDFs for {len(Xs_dict)} syllables...", flush=True)
    verified_animated = 0

    for s, X_2D in tqdm.tqdm(Xs_dict.items(), desc="Animated Trajectories", ncols=75):
        title = f"Syllable{s}"
        fig, ax, rasters = viz.plot_trajectories(
            [title],
            X_2D[None],
            lims,
            edges=edges,
            return_rasters=True,
            num_timesteps=num_timesteps,
            keypoint_colormap=config.get("keypoint_colormap", "autumn"),
        )

        # Save vector PDF
        pdf_path = out_traj / f"{title}.pdf"
        plt.savefig(str(pdf_path))
        plt.close(fig)

        # Save animated GIF using imageio.mimsave with loop=0
        gif_path = out_traj / f"{title}.gif"
        frames_uint8 = [np.uint8(r) for r in rasters]
        imageio.mimsave(str(gif_path), frames_uint8, duration=0.08, loop=0)

        # Quick verify
        im_check = Image.open(str(gif_path))
        if getattr(im_check, "is_animated", False) and getattr(im_check, "n_frames", 1) > 1:
            verified_animated += 1

    print("\n" + "=" * 80)
    print(f"🎉 COMPLETED: {verified_animated} / {len(Xs_dict)} ANIMATED TRAJECTORY GIFS VERIFIED!")
    print(f"   Output Directory: {out_traj}")
    print("=" * 80)

if __name__ == "__main__":
    main()
