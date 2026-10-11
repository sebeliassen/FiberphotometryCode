#!/usr/bin/env python3
"""
rebuild_all_trajectory_gifs.py
=============================================================================
Re-renders pure, unpatched Keypoint-MoSeq trajectory plots (both GIFs and PDFs)
for ALL syllables in Run 12 using native kpms.generate_trajectory_plots.

Guarantees:
- NO MONKEYPATCHING on rasterize_figure.
- Native KPMS rasterization generates true animated multi-frame looping GIFs.
- Generates Syllable0.gif through Syllable99.gif and all_trajectories.gif.
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
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from PIL import Image


def main():
    project_dir = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    model_name = "amph_prod_k150k_run_12"
    out_traj = project_dir / model_name / "trajectory_plots_density"
    out_traj.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 NATIVE KPMS TRAJECTORY ANIMATION GENERATOR (ALL SYLLABLES)")
    print(f"   Project: {project_dir}")
    print(f"   Model:   {model_name}")
    print(f"   Target:  {out_traj}")
    print("=" * 80)

    # 1. Load config & checkpoint
    print("[1/3] Loading configuration and model...", flush=True)
    config = kpms.load_config(str(project_dir))
    bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
    config["bodyparts"] = bodyparts
    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.load_results(str(project_dir), model_name)

    # 2. Coordinates
    print("[2/3] Estimating coordinates...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)

    target_fps = int(config.get("fps", 40))
    config_clean = config.copy()
    config_clean.pop("fps", None)
    config_clean.pop("overlay_keypoints", None)

    # 3. Generate native trajectory plots using built-in KPMS
    print("[3/3] Generating native KPMS trajectory plots...", flush=True)
    kpms.generate_trajectory_plots(
        coordinates=coordinates,
        results=results,
        project_dir=str(project_dir),
        model_name=model_name,
        output_dir=str(out_traj),
        fps=target_fps,
        pre=0.167,
        post=0.5,
        min_frequency=0.0,
        min_duration=2,
        density_sample=True,
        sampling_options={"n_neighbors": 20},
        save_individually=True,
        save_gifs=True,
        save_mp4s=False,
        **config_clean
    )

    # 4. Verify all generated GIFs
    print("\n" + "=" * 80)
    print("🔍 VERIFYING ANIMATED TRAJECTORY GIFS:")
    animated_count = 0
    gifs = sorted(list(out_traj.glob("Syllable*.gif")))
    for g in gifs:
        im = Image.open(g)
        is_anim = getattr(im, "is_animated", False)
        n_fr = getattr(im, "n_frames", 1)
        if is_anim and n_fr > 1:
            animated_count += 1

    print(f"Verified Animated GIFs: {animated_count} / {len(gifs)} (e.g. Syllable0: {Image.open(out_traj / 'Syllable0.gif').n_frames} frames)")
    print("=" * 80)

if __name__ == "__main__":
    main()
