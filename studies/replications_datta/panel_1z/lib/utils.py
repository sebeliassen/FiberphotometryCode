import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("Panel1z.Utils")

def save_plot(fig, name, config):
    """
    Save figure to module's output directory.
    """
    out_dir = config['output_dir'] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"   Saved Plot: {path}")

def save_stats(df, name, config):
    """
    Save DataFrame to module's output directory as CSV.
    """
    out_dir = config['output_dir'] / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"   Saved Stats: {path}")

def bootstrap_ci(arr, n_boot=1000):
    """
    Bootstrap 95% CI for the mean of an array (axis 0).
    Returns (low, high).
    """
    means = []
    n = len(arr)
    rng = np.random.default_rng()

    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True, axis=0)
        means.append(np.mean(sample, axis=0))

    means = np.array(means)
    low = np.percentile(means, 2.5, axis=0)
    high = np.percentile(means, 97.5, axis=0)
    return low, high
