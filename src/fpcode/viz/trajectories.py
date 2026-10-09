"""
fpcode.viz.trajectories
=======================
Trajectory plotting and postural dynamics projections for Keypoint-MoSeq syllables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fpcode.common.logging import get_logger

logger = get_logger("fpcode.viz.trajectories")


def plot_syllable_trajectories(
    coordinates: Dict[str, np.ndarray],
    syllable_labels: Dict[str, np.ndarray],
    syllable_id: int,
    output_path: Optional[Union[str, Path]] = None,
    max_bouts: int = 50,
    centered: bool = True,
) -> Path:
    """
    Plots overlaid 2D centroid or keypoint trajectories for instances of a syllable.

    Parameters
    ----------
    coordinates : dict
        Recording keys to (T, K, 2) keypoint arrays.
    syllable_labels : dict
        Recording keys to (T,) syllable label arrays.
    syllable_id : int
        The target syllable to plot.
    output_path : Path or str, optional
        Destination image path (.png/.pdf).
    max_bouts : int, default=50
        Maximum bouts to plot.
    centered : bool, default=True
        Whether to align bout starts at the origin (0, 0).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    if output_path is None:
        output_path = Path(f"syllable_{syllable_id}_trajectories.png")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    plotted = 0

    for rec_key, sylls in syllable_labels.items():
        if rec_key not in coordinates or plotted >= max_bouts:
            continue
        coords = coordinates[rec_key]
        centroids = np.nanmean(coords, axis=1)

        is_target = sylls == syllable_id
        if not np.any(is_target):
            continue

        # Extract contiguous runs
        diffs = np.diff(is_target.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        if is_target[0]:
            starts = np.r_[0, starts]
        ends = np.where(diffs == -1)[0] + 1
        if is_target[-1]:
            ends = np.r_[ends, len(is_target)]

        for s, e in zip(starts, ends):
            if plotted >= max_bouts:
                break
            traj = centroids[s:e]
            if len(traj) < 3:
                continue
            if centered:
                traj = traj - traj[0]
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1.2)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No bouts of at least 3 frames available for syllable {syllable_id}")
    ax.set_title(f"Syllable {syllable_id} Trajectories ({plotted} bouts)")
    ax.set_xlabel("X (px)" if not centered else "dX (px)")
    ax.set_ylabel("Y (px)" if not centered else "dY (px)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved trajectory plot to {output_path}")
    return output_path
