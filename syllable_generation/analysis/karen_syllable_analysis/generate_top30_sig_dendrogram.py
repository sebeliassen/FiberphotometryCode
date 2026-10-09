#!/usr/bin/env python3
"""
generate_top30_sig_dendrogram.py

Generates a syllable similarity dendrogram for the Top-30 syllables (0 to 29)
plus the significant regulated syllables (6 and 72), yielding exactly 31 syllables.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"

ANALYSIS_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis"
OUT_DIR = ANALYSIS_DIR / "output"
PLOTS_DIR = OUT_DIR / "trajectory_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Define the 31 syllables: Top 30 (0 to 29) + Sig Syllable 72 (Syllable 6 is already in 0..29)
TARGET_SYLLS = sorted(list(set(range(30)).union({6, 72})))
assert len(TARGET_SYLLS) == 31, f"Expected 31 syllables, got {len(TARGET_SYLLS)}"

print("=" * 80)
print("🌳 GENERATING 31-SYLLABLE SIMILARITY DENDROGRAM (Top-30 + Syllables 6 & 72)")
print(f"   Target Syllables ({len(TARGET_SYLLS)}): {TARGET_SYLLS}")
print("   Significant Phenotype Syllables: 6, 72")
print("=" * 80)

# Load checkpoint & config
print("\n[1/3] Loading model & estimating coordinates...", flush=True)
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

fps = config.get("fps", 40)
pre_frames = 5
post_frames = 15

# Estimate continuous coordinates
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)

# Pre-filter centroids and headings
print("[2/3] Pre-filtering centroids & headings...", flush=True)
centroids, headings = kpms.filter_centroids_headings(
    *kpms.get_centroids_headings(coordinates, **config), filter_size=9
)

# Extract typical trajectories for the 31 target syllables
print(f"[3/3] Extracting canonical trajectories for {len(TARGET_SYLLS)} syllables...", flush=True)
syllables_dict = {k: v["syllable"] for k, v in results.items() if "syllable" in v}
all_instances = kpms.util.get_syllable_instances(
    syllables_dict, pre=pre_frames, post=post_frames, min_duration=0, min_frequency=0, min_instances=1
)

typical_trajectories = {}
for s in TARGET_SYLLS:
    insts = all_instances.get(s, [])
    if len(insts) == 0:
        continue
    if len(insts) > 200:
        np.random.seed(0)
        idx = np.random.choice(len(insts), 200, replace=False)
        selected_inst = [insts[i] for i in idx]
    else:
        selected_inst = insts

    trajs = []
    for key, start, end in selected_inst:
        chunk = coordinates[key][start - pre_frames : start + post_frames]
        c = centroids[key][start]
        h = headings[key][start]
        trajs.append(kpms.util.np_io(kpms.util.inverse_rigid_transform)(chunk, c, h))
    typical_trajectories[s] = np.mean(trajs, axis=0)

# Compute pairwise cosine distance matrix
syllable_ixs = sorted(typical_trajectories.keys())
Xs = np.stack([typical_trajectories[s] for s in syllable_ixs])
flat_Xs = Xs.reshape(Xs.shape[0], -1)
distances = squareform(pdist(flat_Xs, metric="cosine"))
Z = linkage(squareform(distances), method="complete")

# 1. Generate Standard and Highlighted Dendrograms
def render_dendrogram(highlight=False, filename_suffix=""):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    labels = []
    for s in syllable_ixs:
        if highlight and s in {6, 72}:
            labels.append(f"★ Syllable {s} (sig)")
        else:
            labels.append(f"Syllable {s}")

    ddata = dendrogram(
        Z,
        labels=labels,
        leaf_font_size=11,
        ax=ax,
        leaf_rotation=90,
        color_threshold=0.7 * max(Z[:, 2]),
    )

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("lightgray")

    title = "Syllable Similarity Dendrogram (Top 30 + Sig Syllables 6, 72 [N=31])"
    if highlight:
        title += " [★ Highlighted]"
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")

    if highlight:
        for tick in ax.get_xticklabels():
            text = tick.get_text()
            if "Syllable 6" in text or "Syllable 72" in text:
                tick.set_color("#d90429")
                tick.set_fontweight("bold")
                tick.set_fontsize(12)
            else:
                tick.set_color("#2b2d42")

    plt.tight_layout()

    pdf_path_1 = OUT_DIR / f"similarity_dendrogram_top30_and_sig{filename_suffix}.pdf"
    png_path_1 = OUT_DIR / f"similarity_dendrogram_top30_and_sig{filename_suffix}.png"
    pdf_path_2 = PLOTS_DIR / f"similarity_dendrogram_top30_and_sig{filename_suffix}.pdf"
    png_path_2 = PLOTS_DIR / f"similarity_dendrogram_top30_and_sig{filename_suffix}.png"

    plt.savefig(pdf_path_1, dpi=300)
    plt.savefig(png_path_1, dpi=300)
    plt.savefig(pdf_path_2, dpi=300)
    plt.savefig(png_path_2, dpi=300)
    plt.close(fig)
    print(f"      Saved: {pdf_path_1.name} & {png_path_1.name}", flush=True)

print("\nRendering dendrograms...", flush=True)
render_dendrogram(highlight=True)
render_dendrogram(highlight=False, filename_suffix="_clean")

print("\n" + "=" * 80)
print("✅ 31-SYLLABLE DENDROGRAM GENERATION COMPLETE!")
print("   Outputs located in:")
print(f"   - {OUT_DIR / 'similarity_dendrogram_top30_and_sig.pdf'}")
print(f"   - {OUT_DIR / 'similarity_dendrogram_top30_and_sig.png'}")
print("=" * 80)
