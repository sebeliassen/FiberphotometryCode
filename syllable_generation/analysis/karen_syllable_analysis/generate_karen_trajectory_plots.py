#!/usr/bin/env python3
"""
generate_karen_trajectory_plots.py

Generates Keypoint-MoSeq Trajectory Plots (both PDF and animated GIF) for all
syllables corresponding to the generated grid movies in the Karen Project (final_karen_model_k240k).

Uses built-in Keypoint-MoSeq functionality:
- kpms.generate_trajectory_plots()
- kpms.viz.plot_trajectories()
- kpms.viz.save_gif()
- kpms.viz.get_edges()

Outputs:
- syllable_generation/analysis/karen_syllable_analysis/output/trajectory_plots/
  * all_trajectories.pdf (8x8 grid of all 64 syllables)
  * all_trajectories.gif (animated 8x8 grid loop of all 64 syllables)
  * Syllable{s}.pdf (individual static trajectory vector plot for each syllable)
  * Syllable{s}.gif (individual animated trajectory loop for each syllable)
"""

import os
import sys
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

# Force headless Agg backend & disable GPU preallocation for plotting safety
os.environ["MPLBACKEND"] = "Agg"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.show = lambda: None  # Ensure headless non-blocking execution

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re
import time
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates
import keypoint_moseq as kpms
from tqdm import tqdm

# ==============================================================================
# 1. MATPLOTLIB 3.8+ COMPATIBILITY PATCH FOR RASTERIZE_FIGURE
# ==============================================================================
def _patched_rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        return np.array(canvas.buffer_rgba())[:, :, :3].copy()
    elif hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3)).copy()
    else:
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_argb(), dtype="uint8").reshape((h, w, 4))[:, :, 1:4].copy()

kpms.viz.rasterize_figure = _patched_rasterize_figure

# ==============================================================================
# 2. PATHS & TARGET CONFIGURATION
# ==============================================================================
PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"

ANALYSIS_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis"
MOVIES_DIR = ANALYSIS_DIR / "output" / "grid_movies_with_trajectories"
OUT_DIR = ANALYSIS_DIR / "output" / "trajectory_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target all 100 syllables in the model (0 to 99)
TARGET_SYLLABLES = list(range(100))

print("=" * 80)
print(f"🎨 KEYPOINT-MOSEQ TRAJECTORY & DENDROGRAM GENERATOR: KAREN DATASET")
print(f"   Model:            {MODEL_NAME}")
print(f"   Target Syllables: {len(TARGET_SYLLABLES)} syllables (Full Repertoire 0 to 99)")
print(f"   Output Directory: {OUT_DIR}")
print("=" * 80)

# ==============================================================================
# 3. LOAD CHECKPOINT, RESULTS & CONFIG
# ==============================================================================
print(f"\n[1/5] Loading model checkpoint and results for {MODEL_NAME}...", flush=True)
t_start = time.time()
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
if "use_bodyparts" in config:
    config["bodyparts"] = config["use_bodyparts"]

fps = config.get("fps", 40)
pre_frames = 5
post_frames = 15
pre_sec = pre_frames / fps   # 0.125s (-5 frames before)
post_sec = post_frames / fps # 0.375s (15 frames after)

print(f"      Model iteration: {current_iter}")
print(f"      FPS: {fps} (Window: {pre_frames} frames pre [{pre_sec}s], {post_frames} frames post [{post_sec}s])")

# ==============================================================================
# 4. ESTIMATE RIGID COORDINATES & PRE-FILTER CENTROIDS/HEADINGS
# ==============================================================================
print(f"\n[2/5] Estimating continuous pose coordinates from latent states...", flush=True)
t0 = time.time()
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
if "bodyparts" in config and "use_bodyparts" in config:
    coordinates = kpms.util.reindex_by_bodyparts(coordinates, config["bodyparts"], config["use_bodyparts"])
print(f"      Estimated coordinates across {len(coordinates)} recordings in {time.time()-t0:.2f}s", flush=True)

print(f"\n[3/5] Pre-filtering session centroids and headings (size=9)...", flush=True)
t0 = time.time()
centroids = {k: v["centroid"] for k, v in results.items()}
headings = {k: v["heading"] for k, v in results.items()}
filt_centroids, filt_headings = kpms.viz.filter_centroids_headings(centroids, headings, filter_size=9)
print(f"      Pre-filtered {len(filt_centroids)} recordings in {time.time()-t0:.2f}s", flush=True)

# ==============================================================================
# 5. COMPUTE HIGH-FIDELITY TYPICAL TRAJECTORIES FOR TARGET SYLLABLES
# ==============================================================================
print(f"\n[4/5] Extracting canonical egocentric trajectories for {len(TARGET_SYLLABLES)} syllables...", flush=True)
t0 = time.time()
syllables = {k: v["syllable"] for k, v in results.items()}
instances = kpms.util.get_syllable_instances(
    syllables, pre=pre_frames, post=post_frames, min_duration=0, min_frequency=0, min_instances=1
)

rng = np.random.default_rng(42)
typical_trajectories = {}

for s in tqdm(TARGET_SYLLABLES, desc="Averaging Syllables", ncols=80):
    inst = instances.get(s, [])
    if len(inst) == 0:
        continue
    # Sample up to 200 representative instances for smooth, noise-free trajectories
    if len(inst) > 200:
        idx = rng.choice(len(inst), 200, replace=False)
        selected_inst = [inst[i] for i in idx]
    else:
        selected_inst = inst

    trajs = []
    for key, start, end in selected_inst:
        chunk = coordinates[key][start - pre_frames : start + post_frames]
        c = filt_centroids[key][start]
        h = filt_headings[key][start]
        trajs.append(kpms.util.np_io(kpms.util.inverse_rigid_transform)(chunk, c, h))
    typical_trajectories[s] = np.mean(trajs, axis=0)

print(f"      Extracted {len(typical_trajectories)} typical trajectories in {time.time()-t0:.2f}s", flush=True)

# ==============================================================================
# 6. RUN BUILT-IN KPMS GENERATE_TRAJECTORY_PLOTS
# ==============================================================================
print(f"\n[5/5] Calling kpms.generate_trajectory_plots to render PDFs & animated GIFs...", flush=True)

# Inject the target trajectories directly into get_typical_trajectories
kpms.viz.get_typical_trajectories = lambda *args, **kwargs: typical_trajectories
kpms.util.get_typical_trajectories = lambda *args, **kwargs: typical_trajectories

cfg_clean = dict(config)
cfg_clean.pop("fps", None)
cfg_clean.pop("pre", None)
cfg_clean.pop("post", None)
cfg_clean.pop("output_dir", None)
cfg_clean.pop("save_individually", None)
cfg_clean.pop("save_gifs", None)
cfg_clean.pop("save_mp4s", None)

kpms.generate_trajectory_plots(
    coordinates=coordinates,
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(OUT_DIR),
    fps=fps,
    pre=pre_sec,
    post=post_sec,
    save_individually=True,
    save_gifs=True,
    save_mp4s=False,
    plot_options={"n_cols": 10, "keypoint_colormap": config.get("keypoint_colormap", "autumn")},
    skeleton=config.get("skeleton", []),
    bodyparts=config.get("bodyparts"),
    use_bodyparts=config.get("use_bodyparts"),
    keypoint_colormap=config.get("keypoint_colormap", "autumn"),
)

# ==============================================================================
# 7. GENERATE SYLLABLE SIMILARITY DENDROGRAM (ALL 100 SYLLABLES)
# ==============================================================================
print(f"\n[6/6] Generating syllable similarity dendrogram (Cosine Distance)...", flush=True)
import shutil
dendro_path = OUT_DIR / "similarity_dendrogram"
kpms.plot_similarity_dendrogram(
    coordinates=coordinates,
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    save_path=str(dendro_path),
    figsize=(22, 7),
    fps=fps,
    pre=pre_sec,
    post=post_sec,
    min_frequency=0,
    min_duration=0,
    metric="cosine",
)

# Copy to output/ root for convenient top-level access
if (OUT_DIR / "similarity_dendrogram.pdf").exists():
    shutil.copy(str(OUT_DIR / "similarity_dendrogram.pdf"), str(ANALYSIS_DIR / "output" / "similarity_dendrogram.pdf"))
if (OUT_DIR / "similarity_dendrogram.png").exists():
    shutil.copy(str(OUT_DIR / "similarity_dendrogram.png"), str(ANALYSIS_DIR / "output" / "similarity_dendrogram.png"))

print(f"      Dendrogram successfully saved:")
print(f"       - {dendro_path}.pdf")
print(f"       - {dendro_path}.png")
print(f"       - {ANALYSIS_DIR / 'output' / 'similarity_dendrogram.pdf'}")

print("\n" + "=" * 80)
print(f"✅ ALL 100 TRAJECTORY PLOTS (PDF & GIF) AND DENDROGRAM GENERATED!")
print(f"   Directory: {OUT_DIR}")
print(f"   Total elapsed time: {time.time() - t_start:.2f}s")
print("=" * 80)
