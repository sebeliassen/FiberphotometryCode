"""
Cohort Strategy Visualization Module

Provides clean functions for generating cohort proportion bar plots, 2D PCA strategy trajectories,
and transition probability matrices across experimental groups.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_cohort_strategy_proportions(
    session_proportions: np.ndarray,
    cohort_labels: List[str],
    output_path: str,
    vibe_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Generates a stacked bar chart comparing mean time spent in each strategy vibe across cohorts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(session_proportions)
    df["cohort"] = cohort_labels

    mean_props = df.groupby("cohort").mean()
    unique_cohorts = mean_props.index.tolist()
    n_topics = session_proportions.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(unique_cohorts))

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_topics)))

    for t in range(n_topics):
        props = mean_props[t].values
        label = vibe_names.get(t, f"Vibe {t}") if vibe_names else f"Vibe {t}"
        ax.bar(unique_cohorts, props, bottom=bottom, label=label, color=colors[t % len(colors)])
        bottom += props

    ax.set_ylabel("Mean Proportion of Time")
    ax.set_title("Behavioral Strategy Vibe Distribution by Experimental Cohort")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved cohort strategy proportion plot to: {output_path}")

def plot_transition_matrix(
    transition_matrix: np.ndarray,
    output_path: str,
    title: str = "Strategy Transition Probability Matrix"
) -> None:
    """
    Plots a heatmap of strategy-to-strategy transition probabilities.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1)

    n_topics = transition_matrix.shape[0]
    ax.set_xticks(np.arange(n_topics))
    ax.set_yticks(np.arange(n_topics))
    ax.set_xlabel("Target Strategy S_t")
    ax.set_ylabel("Source Strategy S_{t-1}")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Transition Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved strategy transition matrix plot to: {output_path}")
