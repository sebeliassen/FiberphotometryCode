#!/usr/bin/env python3
"""
generate_dorsal_grid_movies.py
==============================
Generates behavioral syllable grid movies from Keypoint-MoSeq model checkpoints
focusing on top dorsal camera videos and top-down posture instances.

Follows all rules and best practices from kpms-visualizations skill:
1. Forces CPU execution (no GPU CUDA/ptxas conflicts).
2. Uses verified 1:1 segment-to-video mapping for karen_project/video_files/*.avi.
3. Native 60 fps timing alignment.
4. Patches get_syllable_instances to focus on top N most frequent syllables.
5. Generates both raw video crops and keypoint skeleton overlay grid movies.
"""

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Force CPU execution before importing JAX / KPMS
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import keypoint_moseq as kpms
import keypoint_moseq.util as util
import keypoint_moseq.viz as viz
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
ARM_DIR = EXP_DIR / "pilot_comparison_21kp_static"
VIDEO_DIR = REPO_ROOT / "studies/karen_pff_vs_monomer/video_files"


def modern_rasterize_figure(fig):
    """Patch for matplotlib rasterize in KPMS trajectory/overlay figures."""
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    elif hasattr(fig.canvas, "tostring_rgb"):
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3))
    return np.asarray(fig.canvas.print_to_buffer()[0])[:, :, :3]

viz.rasterize_figure = modern_rasterize_figure


def main():
    parser = argparse.ArgumentParser(description="Generate dorsal syllable grid movies.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--top-n-syllables", type=int, default=15, help="Number of top syllables to generate")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows (default: 4)")
    parser.add_argument("--cols", type=int, default=6, help="Grid cols (default: 6, total 24 instances)")
    args = parser.parse_args()

    model_path = Path(args.model_dir).resolve()
    project_dir = model_path.parent
    model_name = model_path.name

    print("=" * 80)
    print("🎬 DORSAL SYLLABLE GRID MOVIE GENERATOR")
    print(f"  Project Dir: {project_dir}")
    print(f"  Model Name:  {model_name}")
    print(f"  Top Sylls:   {args.top_n_syllables}")
    print(f"  Grid Size:   {args.rows}x{args.cols} ({args.rows * args.cols} instances per movie)")
    print("=" * 80, flush=True)

    # 1. Load config
    config = kpms.load_config(str(project_dir))
    if "use_bodyparts" in config:
        config["bodyparts"] = config["use_bodyparts"]
    elif "bodyparts" in config:
        config["use_bodyparts"] = config["bodyparts"]

    # 2. Load model checkpoint
    print(f"Loading checkpoint for {model_name}...", flush=True)
    model, data, metadata, current_iter = kpms.load_checkpoint(str(project_dir), model_name)
    print(f"Checkpoint loaded successfully (Iteration {current_iter}).", flush=True)

    # 3. Extract results
    print("Extracting model results...", flush=True)
    results = kpms.extract_results(model, metadata, str(project_dir), model_name, save_results=False)

    # 4. Estimate coordinates, centroids, headings
    print("Estimating posture coordinates, centroids, and headings...", flush=True)
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"])
    )
    coordinates = unbatch(Y_est, *metadata)
    centroids, headings = kpms.get_centroids_headings(coordinates, **config)

    # 5. Load video frame indexes
    vfi_path = project_dir / "video_frame_indexes.npz"
    if not vfi_path.exists():
        raise FileNotFoundError(f"video_frame_indexes.npz not found at {vfi_path}")
    with np.load(vfi_path) as z:
        video_frame_indexes = {k: z[k] for k in z.files}

    # 6. Map segments to video files
    all_vids = list(VIDEO_DIR.glob("*.avi"))
    video_paths = {}
    for k in video_frame_indexes.keys():
        base = re.sub(r"_seg\d+$", "", k)
        matched = [v for v in all_vids if base in v.name]
        if matched:
            video_paths[k] = str(matched[0])
        else:
            print(f"⚠️ Warning: Missing video file for segment {k} (base: {base})")

    # 7. Intersect common keys across all dicts
    common_keys = sorted(list(set(results.keys()) & set(video_frame_indexes.keys()) & set(video_paths.keys()) & set(coordinates.keys())))
    print(f"Matched {len(common_keys)} continuous segments for movie rendering.")

    results_common = {k: results[k] for k in common_keys}
    coordinates_common = {k: coordinates[k] for k in common_keys}
    centroids_common = {k: centroids[k] for k in common_keys}
    headings_common = {k: headings[k] for k in common_keys}
    vfi_common = {k: video_frame_indexes[k] for k in common_keys}
    vid_paths_common = {k: video_paths[k] for k in common_keys}

    # 8. Identify top N syllables by total frame frequency
    all_sylls = np.concatenate([v["syllable"][v["syllable"] >= 0] for v in results_common.values()])
    counts = Counter(all_sylls)
    top_syllables_list = [s for s, _ in counts.most_common(args.top_n_syllables)]
    top_syllables_set = set(top_syllables_list)
    print(f"\nTop {args.top_n_syllables} most frequent syllables to render:")
    for rank, (s, cnt) in enumerate(counts.most_common(args.top_n_syllables), 1):
        pct = (cnt / len(all_sylls)) * 100
        print(f"  Rank {rank:2d}: Syllable {s:2d} -> {cnt:7d} frames ({pct:.2f}%)")

    # Patch get_syllable_instances to focus only on top N syllables
    orig_gsi = util.get_syllable_instances
    def filtered_gsi(*gsi_args, **gsi_kwargs):
        insts = orig_gsi(*gsi_args, **gsi_kwargs)
        return {k: v for k, v in insts.items() if k in top_syllables_set}
    viz.get_syllable_instances = filtered_gsi
    util.get_syllable_instances = filtered_gsi

    # Clean config
    config_clean = config.copy()
    target_fps = int(config_clean.pop("fps", 60))
    config_clean.pop("overlay_keypoints", None)

    # Output directory for raw video grid movies
    out_raw = model_path / "grid_movies_dorsal_raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    print(f"\n🎥 Generating RAW dorsal grid movies -> {out_raw}...", flush=True)

    kpms.generate_grid_movies(
        results=results_common,
        project_dir=str(project_dir),
        model_name=model_name,
        output_dir=str(out_raw),
        video_paths=vid_paths_common,
        coordinates=coordinates_common,
        centroids=centroids_common,
        headings=headings_common,
        video_frame_indexes=vfi_common,
        fps=target_fps,
        rows=args.rows,
        cols=args.cols,
        min_frequency=0.001,
        min_duration=4,
        overlay_keypoints=False,
        sampling_options={"mode": "random"},
        **config_clean
    )
    print("✅ Raw dorsal grid movies completed!", flush=True)

    # Output directory for overlaid keypoints grid movies
    out_overlay = model_path / "grid_movies_dorsal_overlay"
    out_overlay.mkdir(parents=True, exist_ok=True)
    print(f"\n🎥 Generating OVERLAY dorsal grid movies -> {out_overlay}...", flush=True)

    kpms.generate_grid_movies(
        results=results_common,
        project_dir=str(project_dir),
        model_name=model_name,
        output_dir=str(out_overlay),
        video_paths=vid_paths_common,
        coordinates=coordinates_common,
        centroids=centroids_common,
        headings=headings_common,
        video_frame_indexes=vfi_common,
        fps=target_fps,
        rows=args.rows,
        cols=args.cols,
        min_frequency=0.001,
        min_duration=4,
        overlay_keypoints=True,
        sampling_options={"mode": "random"},
        **config_clean
    )
    print("✅ Overlay dorsal grid movies completed!", flush=True)

    raw_files = list(out_raw.glob("*.mp4"))
    overlay_files = list(out_overlay.glob("*.mp4"))
    print("\n" + "=" * 80)
    print("🎉 COMPLETED DORSAL SYLLABLE INSTANCES GENERATION!")
    print(f"  Raw grid movies:     {len(raw_files)} files in {out_raw}")
    print(f"  Overlay grid movies: {len(overlay_files)} files in {out_overlay}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
