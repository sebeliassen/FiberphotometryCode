"""
fpcode.viz.figures
==================
Publication-quality figure generation for syllable usage, transition matrices,
and peri-event time histograms (PETH).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fpcode.common.logging import get_logger

logger = get_logger("fpcode.viz.figures")


def plot_syllable_usage(
    frequencies: Union[Dict[int, float], np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Syllable Usage Frequency",
) -> Path:
    """Plots a bar chart of syllable usage frequencies."""
    if output_path is None:
        output_path = Path("syllable_usage.png")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frequencies, dict):
        sylls = sorted(frequencies.keys())
        vals = [frequencies[s] for s in sylls]
    else:
        vals = np.asarray(frequencies)
        sylls = list(range(len(vals)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sylls, vals, color="#3498db", edgecolor="#2980b9", width=0.8)
    ax.set_xlabel("Syllable ID")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved syllable usage figure to {output_path}")
    return output_path


def plot_transition_matrix(
    tm: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Syllable Transition Matrix",
) -> Path:
    """Plots a transition matrix heatmap."""
    if output_path is None:
        output_path = Path("transition_matrix.png")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(tm, cmap="viridis", interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Transition Probability")
    ax.set_xlabel("To Syllable")
    ax.set_ylabel("From Syllable")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved transition matrix figure to {output_path}")
    return output_path
