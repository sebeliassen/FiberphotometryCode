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

    logger.info(f"Generating grid movies ({sampling_mode} sampling) into {output_dir}...")

    # Delegate to kpms.generate_grid_movies if available
    try:
        import keypoint_moseq as kpms
        sampling_options = (
            {"mode": "density", "pca_samples": 50000, "pca_dim": 4, "n_neighbors": 50}
            if sampling_mode == "density"
            else {"mode": "random"}
        )
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
        generated = list(output_dir.glob("*.mp4"))
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
