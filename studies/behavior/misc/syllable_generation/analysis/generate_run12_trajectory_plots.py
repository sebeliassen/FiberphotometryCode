#!/usr/bin/env python3
"""
generate_run12_trajectory_plots.py
=============================================================================
Generate Keypoint-MoSeq Trajectory Plots (Density Sampling) for Run 12
(amph_prod_k150k_run_12).

Outputs:
- Combined overview: all_trajectories.pdf and all_trajectories.gif
- Individual syllable plots: Syllable0.pdf .. Syllable29.pdf
- Individual syllable animations: Syllable0.gif .. Syllable29.gif
in amph_project/amph_prod_k150k_run_12/trajectory_plots_density/

Restricted to top 30 syllables (00 to 29) using built-in KPMS API.
=============================================================================
"""

import os
import sys
from pathlib import Path

# Force CPU execution & headless matplotlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

# Syllable filter for top 30 syllables (0 to 29)
orig_get_syllable_instances = util.get_syllable_instances

def filtered_get_syllable_instances(*args, **kwargs):
    instances = orig_get_syllable_instances(*args, **kwargs)
    filtered = {k: v for k, v in instances.items() if 0 <= k < 30}
    return filtered

viz.get_syllable_instances = filtered_get_syllable_instances
util.get_syllable_instances = filtered_get_syllable_instances

# Optimization: filter centroids & headings once globally to avoid redundant filtering per syllable
_orig_filter = util.filter_centroids_headings
_filtered_cache = {}

def fast_filter_centroids_headings(centroids, headings, filter_size=9):
    key = id(centroids)
    if key not in _filtered_cache:
        _filtered_cache[key] = _orig_filter(centroids, headings, filter_size=filter_size)
    return _filtered_cache[key]

util.filter_centroids_headings = fast_filter_centroids_headings
viz.filter_centroids_headings = fast_filter_centroids_headings

# Modern Matplotlib 3.8+ fix: tostring_rgb -> buffer_rgba
def modern_rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[:, :, :3]
    elif hasattr(canvas, "tostring_rgb"):
        width, height = canvas.get_width_height()
        raster_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        return raster_flat.reshape((height, width, 3))
    else:
        rgba = np.asarray(canvas.print_to_buffer()[0])
        return rgba[:, :, :3]

viz.rasterize_figure = modern_rasterize_figure


def main():
    project_dir = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "amph_project"
    model_name = "amph_prod_k150k_run_12"
    output_dir = project_dir / model_name / "trajectory_plots_density"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎨 GENERATING KPMS TRAJECTORY PLOTS (DENSITY SAMPLING, TOP 30 SYLLABLES)")
    print(f"   Project:    {project_dir}")
    print(f"   Model:      {model_name}")
    print(f"   Output:     {output_dir}")
    print("=" * 80)

    # 1. Load config
    print("[1/4] Loading project configuration...", flush=True)
    config = kpms.load_config(str(project_dir))
    bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
    config["bodyparts"] = bodyparts
    skeleton = config.get("skeleton", [])
    use_bodyparts = config.get("use_bodyparts", bodyparts)
    fps = config.get("fps", 40)

    # 2. Load model checkpoint and results
    print(f"[2/4] Loading checkpoint and results for {model_name}...", flush=True)
    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.load_results(str(project_dir), model_name)

    # 3. Estimate latent coordinates
    print("[3/4] Estimating coordinates from latent SLDS states...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    print(f"[INFO] Successfully unbatched {len(coordinates)} session coordinate sets.", flush=True)

    # 4. Generate trajectory plots using built-in KPMS API
    print("[4/4] Generating trajectory plots via kpms.generate_trajectory_plots...", flush=True)
    kpms.generate_trajectory_plots(
        coordinates=coordinates,
        results=results,
        project_dir=str(project_dir),
        model_name=model_name,
        output_dir=str(output_dir),
        pre=0.167,
        post=0.5,
        fps=fps,
        skeleton=skeleton,
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        density_sample=True,
        sampling_options={"n_neighbors": 50},
        save_individually=True,
        save_gifs=True,
        save_mp4s=False,
        interactive=False,
    )

    print("\n" + "=" * 80)
    print("✓ TRAJECTORY PLOTS SUCCESSFULLY GENERATED!")
    print(f"   Output Directory: {output_dir}")

    # Verify outputs
    pdf_files = sorted(list(output_dir.glob("*.pdf")))
    gif_files = sorted(list(output_dir.glob("*.gif")))
    print(f"   PDF files generated: {len(pdf_files)} (e.g. {[p.name for p in pdf_files[:5]]} ...)")
    print(f"   GIF files generated: {len(gif_files)} (e.g. {[g.name for g in gif_files[:5]]} ...)")
    print("=" * 80)

if __name__ == "__main__":
    main()
