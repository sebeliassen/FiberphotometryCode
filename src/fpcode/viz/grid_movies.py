"""
fpcode.viz.grid_movies
======================
Generates syllable grid movies from Keypoint-MoSeq models and results.
Supports both density-sampled and uniform-random instance sampling,
egocentric canonical cropping, and keypoint overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fpcode.common.logging import get_logger

logger = get_logger("fpcode.viz.grid_movies")

DENSITY_SAMPLING_OPTIONS = {"mode": "density", "pca_samples": 50000, "pca_dim": 4, "n_neighbors": 50}


def exact_grid_layout(tiles: int) -> tuple[int, int]:
    """Factor a tile count into a landscape grid without adding extra tiles."""
    if isinstance(tiles, bool) or not isinstance(tiles, int) or tiles < 1:
        raise ValueError("grid_size must be a positive integer")
    rows = int(np.sqrt(tiles))
    while tiles % rows:
        rows -= 1
    return rows, tiles // rows


def _generate_selected_grid_movies(
    viz, results, output_dir, video_paths, coordinates, centroids, headings,
    video_frame_indexes, fps, rows, cols, min_frequency, min_duration,
    overlay_keypoints, sampling_options, syllables, options,
):
    """Use backend primitives to select IDs before rendering, without relabeling.

    The installed KPMS high-level renderer accepts arbitrary kwargs but ignores
    ``syllables``. Density sampling needs the complete eligible population as its
    reference; filter the sampled outputs before encoding, preserving that fit.
    """
    supported = {"pre", "post", "filter_size", "window_size", "max_video_size", "quality",
                 "dot_radius", "dot_color", "bodyparts", "use_bodyparts", "skeleton",
                 "plot_options", "use_dims", "keypoint_colormap"}
    unknown = set(options) - supported
    if unknown:
        raise ValueError(f"Unsupported subset rendering options: {', '.join(sorted(unknown))}")
    pre = round(options.get("pre", 1.0) * fps)
    post = round(options.get("post", 2.0) * fps)
    if coordinates is None:
        raise ValueError("Subset grid movies require coordinates")
    # Density sampling replaces entries with interpolated arrays. Keep those
    # working entries local so later trajectory plots see the original data.
    coordinates = dict(coordinates)
    if options.get("bodyparts") is not None and options.get("use_bodyparts") is not None:
        coordinates = viz.reindex_by_bodyparts(coordinates, options["bodyparts"], options["use_bodyparts"])
    centroids = centroids if centroids is not None else {key: record["centroid"] for key, record in results.items()}
    headings = headings if headings is not None else {key: record["heading"] for key, record in results.items()}
    centroids, headings = viz.filter_centroids_headings(centroids, headings, filter_size=options.get("filter_size", 9))
    labels = {key: record["syllable"] for key, record in results.items()}
    viz.check_video_paths(video_paths, results.keys())
    if video_frame_indexes is None:
        video_frame_indexes = {key: np.arange(len(value)) for key, value in labels.items()}
    if set(video_frame_indexes) != set(results) or any(len(video_frame_indexes[key]) != len(labels[key]) for key in labels):
        raise ValueError("Video frame indexes must match all recording keys and label lengths")
    instances = viz.get_syllable_instances(labels, pre=pre, post=post, min_duration=min_duration,
                                         min_frequency=min_frequency, min_instances=rows * cols)
    requested = set(syllables)
    missing = requested - set(instances)
    if missing:
        raise ValueError(f"Requested syllables lack enough eligible bouts for the grid: {sorted(missing)}")
    # Random sampling has no reference population. Density sampling fits PCA and
    # background neighbors across all eligible syllables, even for one movie.
    sampling_instances = (instances if sampling_options.get("mode") == "density"
                          else {key: value for key, value in instances.items() if key in requested})
    sampled = viz.sample_instances(sampling_instances, rows * cols, coordinates=coordinates,
                                   centroids=centroids, headings=headings, **sampling_options)
    sampled = {key: value for key, value in sampled.items() if key in requested}
    window = options.get("window_size")
    if window is None:
        window = viz.get_grid_movie_window_size(sampled, centroids, headings, coordinates, pre, post)
    window = int(window)
    if window <= 0:
        raise ValueError("Grid movie window_size must be positive")
    scaled = min(int(np.floor(options.get("max_video_size", 1920) / max(rows, cols) / 16) * 16), window)
    if scaled <= 0:
        raise ValueError("max_video_size is too small for the requested grid")
    edges = viz.get_edges(options.get("use_bodyparts"), options.get("skeleton", [])) if overlay_keypoints else []
    plot_options = dict(options.get("plot_options", {}))
    plot_options["keypoint_colormap"] = options.get("keypoint_colormap", "autumn")
    for syllable, bouts in sampled.items():
        frames = viz.grid_movie(
            bouts, rows, cols, video_paths, centroids, headings, window, video_frame_indexes,
            edges=edges, scaled_window_size=scaled, dot_color=options.get("dot_color", (255, 255, 255)),
            pre=pre, post=post, dot_radius=options.get("dot_radius", 4),
            overlay_keypoints=overlay_keypoints, coordinates=coordinates,
            plot_options=plot_options, use_dims=options.get("use_dims", [0, 1]),
        )
        viz.write_video_clip(frames, str(output_dir / f"syllable{syllable}.mp4"), fps=fps, quality=options.get("quality", 7))


def generate_grid_movies(
    results: Dict[str, Any],
    project_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    video_paths: Optional[Dict[str, str]] = None,
    coordinates: Optional[Dict[str, np.ndarray]] = None,
    centroids: Optional[Dict[str, np.ndarray]] = None,
    headings: Optional[Dict[str, np.ndarray]] = None,
    video_frame_indexes: Optional[Dict[str, np.ndarray]] = None,
    fps: int = 40,
    rows: int = 4,
    cols: int = 6,
    min_frequency: float = 0.005,
    min_duration: int = 3,
    overlay_keypoints: bool = False,
    sampling_mode: str = "density",
    syllables: Optional[List[int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Renders multi-tile grid movies showing instances of syllables across sessions.

    Parameters
    ----------
    results : dict
        Results mapping recording keys to syllable classifications and latent states.
    project_dir : str or Path, optional
        Project root directory.
    model_name : str, optional
        Model name within project directory.
    output_dir : str or Path, optional
        Destination directory for generated MP4 files.
    video_paths : dict, optional
        Mapping of recording keys to video file paths.
    sampling_mode : {"density", "random"}, default="density"
        Sampling mode for selecting bouts within each syllable.
    syllables : list of int, optional
        Subset of syllable IDs to render. If None, renders all passing min_frequency.

    Returns
    -------
    dict
        Summary of generated movies with counts, paths, and timings.
    """
    if output_dir is None:
        if project_dir and model_name:
            output_dir = Path(project_dir) / model_name / f"grid_movies_{sampling_mode}"
        else:
            output_dir = Path("grid_movies")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in output_dir.glob("*.mp4")}

    logger.info(f"Generating grid movies ({sampling_mode} sampling) into {output_dir}...")

    # Delegate to kpms.generate_grid_movies if available
    try:
        import keypoint_moseq as kpms
        sampling_options = (
            dict(DENSITY_SAMPLING_OPTIONS)
            if sampling_mode == "density"
            else {"mode": "random"}
        )
        if syllables is not None:
            from keypoint_moseq import viz
            _generate_selected_grid_movies(
                viz, results, output_dir, video_paths, coordinates, centroids, headings,
                video_frame_indexes, fps, rows, cols, min_frequency, min_duration,
                overlay_keypoints, sampling_options, syllables, kwargs,
            )
        else:
            kpms.generate_grid_movies(
            results=results,
            project_dir=str(project_dir) if project_dir else None,
            model_name=model_name,
            output_dir=str(output_dir),
            video_paths=video_paths,
            coordinates=coordinates,
            centroids=centroids,
            headings=headings,
            video_frame_indexes=video_frame_indexes,
            fps=fps,
            rows=rows,
            cols=cols,
            min_frequency=min_frequency,
            min_duration=min_duration,
            overlay_keypoints=overlay_keypoints,
            sampling_options=sampling_options,
            **kwargs,
            )
        generated = sorted(output_dir.glob("*.mp4"))
        generated = [path for path in generated
                     if previous.get(path.name) != (path.stat().st_size, path.stat().st_mtime_ns)]
        if syllables is not None:
            wanted = {f"syllable{syllable}.mp4" for syllable in syllables}
            generated = [path for path in generated if path.name in wanted]
        return {
            "output_dir": str(output_dir),
            "sampling_mode": sampling_mode,
            "movies_count": len(generated),
            "files": [str(f) for f in generated],
        }
    except Exception as e:
        logger.warning(f"KPMS generate_grid_movies failed or unavailable: {e}. Outputting metadata.")
        return {
            "output_dir": str(output_dir),
            "sampling_mode": sampling_mode,
            "movies_count": 0,
            "error": str(e),
        }
