import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import config
from .signal_processing import compute_mean_ci


def get_time_axis():
    """
    Build a time axis from config.peak_interval_config and config.PLOTTING_CONFIG['fps'].
    Returns:
        xs: 1D numpy array of timepoints.
    """
    interval_start = config.peak_interval_config['interval_start']
    interval_end = config.peak_interval_config['interval_end']
    fps = config.PLOTTING_CONFIG['cpt']['fps']
    return np.arange(-interval_start, interval_end) / fps


def plot_matrix_ci(
    mat,
    xs=None,
    smoothing_len=1,
    color=None,
    label=None,
    xlabel='Time (s)',
    ylabel='z-score',
    title=None,
    save_path=None
):
    """
    Plot the mean trace ± 95% CI for a signal matrix.

    Parameters:
    - mat: 2D array (n_trials, n_time)
    - xs: optional 1D time axis; if None, derived from config
    - smoothing_len: int window length for moving-average smoothing of the mean
    - color: color spec for line and fill
    - label: legend label for the mean line
    - xlabel, ylabel: axis labels
    - title: optional figure title
    - save_path: if provided, saves the figure
    """
    # prepare time axis
    if xs is None:
        xs = get_time_axis()

    # compute and smooth mean
    mean_signal = np.mean(mat, axis=0)
    if smoothing_len and smoothing_len > 1:
        window = np.ones(smoothing_len) / smoothing_len
        mean_signal = np.convolve(mean_signal, window, mode='same')

    # compute CI on raw data
    ys_ci, lo, hi = compute_mean_ci(mat)

    # plotting
    plt.figure(dpi=150)
    plt.plot(xs, mean_signal, color=color, label=label or 'Mean')
    plt.fill_between(xs, lo, hi, color=color, alpha=0.2)
    plt.axvline(0, color='grey', linestyle='--')

    # labels and title
    if title:
        plt.title(title)
    if label:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # save or close
    if save_path:
        plt.savefig(save_path)
    plt.show()
    #plt.close()


def plot_individual_traces(
    mats,
    xs=None,
    color='C0',
    alpha=0.3,
    xlabel='Time (s)',
    ylabel='z-score',
    title=None,
    save_path=None
):
    """
    Overlay individual-session mean traces from a list of matrices.

    Parameters:
    - mats: iterable of 2D arrays (n_trials, n_time)
    - xs: optional 1D time axis; if None, derived from config
    - color: line color
    - alpha: transparency for each trace
    - xlabel, ylabel: axis labels
    - title: optional figure title
    - save_path: if provided, saves the figure
    """
    if xs is None:
        xs = get_time_axis()

    plt.figure(dpi=150)
    for m in mats:
        plt.plot(xs, np.mean(m, axis=0), color=color, alpha=alpha)
    plt.axvline(0, color='grey', linestyle='--')

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()
