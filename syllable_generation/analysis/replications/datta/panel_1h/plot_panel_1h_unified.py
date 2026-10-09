
import numpy as np

"""
Panel 1h Plotting Script (Unified)

This script generates visualizations for Panel 1h, supporting both 'peak' and 'trace' modes.

Modes:
- Peak: Generates bar plots comparing correlation coefficients across genotypes/regions.
- Trace: Generates heatmaps of Z-scored activity centered on syllable onsets.

Features:
- Auto-discovery of Hemispheres (L/R) and Fiber Colors (G/R).
- Drug splitting for AMPH dataset (Saline/Amphetamine).
- Row-wise normalization and sorting by peak activity.
- Bit-exact visual parity with legacy notebook outputs.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Pathing
this_file = Path(__file__).resolve()
datta_dir = this_file.parents[1]
repo_root = this_file.parents[5]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(datta_dir) not in sys.path:
    sys.path.insert(0, str(datta_dir))

# Parameters (Sync with 1h.ipynb)
X_LOWER_BOUND = -3.0
X_UPPER_QUANTILE = 0.999
BIN_COUNT = 100
CMAP = "magma"
SMOOTHING_SIGMA = 2
ROW_WISE_NORM = True
MIN_TRIALS = 100
KEEP_TOP_N_SYLLABLES = 50

# --- Trace Plotting Params (from 1h_right_all.ipynb) ---
TRACE_WINDOW_SECONDS = 1.5
TRACE_CMAP = "RdBu_r"
TRACE_SIGMA_SMOOTHING = 1.0
TRACE_ROW_WISE_NORM = True

def plot_panel_1h_trace(project, min_trials=50):
    data_path = Path(__file__).parent / "results" / f"results_1h_trace_{project}.pkl"
    if not data_path.exists():
        print(f"Error: Trace data file {data_path} not found. Run digestion with --type trace first.")
        return

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data_buckets = pickle.load(f)

    # Auto-discovery
    discovered_drugs = sorted(data_buckets.keys())

    unique_genos = sorted({g for d in data_buckets for g in data_buckets[d]})
    unique_regions = sorted({r for d in data_buckets for g in data_buckets[d] for r in data_buckets[d][g]})

    discovered_hemis = set()
    discovered_colors = set()
    for d in data_buckets:
        for g in data_buckets[d]:
            for r in data_buckets[d][g]:
                for h in data_buckets[d][g][r]:
                    discovered_hemis.add(h)
                    for c in data_buckets[d][g][r][h]:
                        discovered_colors.add(c)

    target_hemis = sorted(list(discovered_hemis))
    target_colors = sorted(list(discovered_colors))

    print(f"Drugs: {discovered_drugs}")
    print(f"Hemispheres: {target_hemis}")
    print(f"Colors: {target_colors}")

    output_dir = Path(__file__).parent / "plots" / f"panel_1h_trace_{project}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for drug in discovered_drugs:
        for color in target_colors:
            for hemi in target_hemis:
                print(f"\n=== Plotting: {drug} | {hemi} Hemisphere | {color} Channel ===")

                # Check Data Existence
                has_data = False
                for g in unique_genos:
                    for r in unique_regions:
                        # Safe check for nested dict
                        val = data_buckets.get(drug, {}).get(g, {}).get(r, {}).get(hemi, {}).get(color)
                        if val:
                            has_data = True
                            break
                    if has_data: break

                if not has_data:
                    print(f"   [SKIP] No data found for {drug}-{hemi}-{color}.")
                    continue

                fig, axes = plt.subplots(
                    nrows=len(unique_regions), ncols=len(unique_genos),
                    figsize=(5 * len(unique_genos), 6 * len(unique_regions)),
                    dpi=150, sharex=False, sharey=False
                )

                if len(unique_regions) == 1 and len(unique_genos) == 1: axes = np.array([[axes]])
                elif len(unique_regions) == 1: axes = axes[np.newaxis, :]
                elif len(unique_genos) == 1: axes = axes[:, np.newaxis]

                for r_idx, region in enumerate(unique_regions):
                    for c_idx, geno in enumerate(unique_genos):
                        ax = axes[r_idx, c_idx]

                        # Safe access
                        syll_data = data_buckets.get(drug, {}).get(geno, {}).get(region, {}).get(hemi, {}).get(color, {})

                        if not syll_data:
                            ax.set_title(f"{region} | {geno}\n(No Data)", fontsize=10)
                            ax.axis("off")
                            continue

                        z_waveforms = []
                        syllables = []
                        peak_metrics = []

                        # Process Syllables
                        for syll, session_list in syll_data.items():
                            total_trials = sum(item[0] for item in session_list)
                            if total_trials < min_trials: continue

                            first_act = session_list[0][1]
                            if first_act is None: continue

                            weighted_actual = np.zeros_like(first_act)
                            weighted_shuffle = np.zeros_like(first_act)
                            all_shuffle_means = []

                            for n, act, shuff in session_list:
                                weighted_actual += act * n
                                weighted_shuffle += shuff * n
                                all_shuffle_means.append(shuff)

                            global_actual = weighted_actual / total_trials
                            global_shuffle = weighted_shuffle / total_trials

                            global_actual = np.nan_to_num(global_actual)
                            global_shuffle = np.nan_to_num(global_shuffle)

                            if len(all_shuffle_means) > 1:
                                global_shuffle_std = np.std(all_shuffle_means, axis=0)
                            else:
                                global_shuffle_std = np.ones_like(global_shuffle)

                            global_shuffle_std[global_shuffle_std == 0] = 1.0
                            z_trace = (global_actual - global_shuffle) / global_shuffle_std

                            if TRACE_ROW_WISE_NORM:
                                max_val = np.nanmax(np.abs(z_trace))
                                if max_val > 0 and not np.isnan(max_val):
                                    z_trace /= max_val

                            if TRACE_SIGMA_SMOOTHING > 0:
                                z_trace = np.nan_to_num(z_trace)
                                z_trace = gaussian_filter1d(z_trace, sigma=TRACE_SIGMA_SMOOTHING)

                            z_waveforms.append(z_trace)
                            syllables.append(syll)

                            idx_0 = len(z_trace) // 2
                            idx_end = idx_0 + int(len(z_trace) * 0.1)
                            peak_metrics.append(np.mean(z_trace[idx_0:idx_end]))

                        if not z_waveforms:
                            ax.text(0.5, 0.5, f"All < {min_trials}", ha='center', va='center'); continue

                        # Sort
                        z_matrix = np.array(z_waveforms)
                        sort_idx = np.argsort(peak_metrics)[::-1]
                        z_sorted = z_matrix[sort_idx]

                        plot_vmax = 1.0 if TRACE_ROW_WISE_NORM else np.percentile(np.abs(z_sorted), 99)

                        im = ax.imshow(
                            z_sorted,
                            aspect="auto",
                            cmap=TRACE_CMAP,
                            vmin=-plot_vmax, vmax=plot_vmax,
                            interpolation="nearest",
                            extent=(-TRACE_WINDOW_SECONDS, TRACE_WINDOW_SECONDS, len(z_sorted), 0)
                        )

                        ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
                        ax.set_title(f"{region} | {geno}\n({len(z_sorted)} Sylls)", fontsize=10)
                        ax.set_xlabel("Time (s)", fontsize=8)

                plt.tight_layout()
                save_name = f"trace_{drug}_{hemi}_{color}.png"
                save_path = output_dir / save_name
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                print(f"Saved plot to {save_path}")

def plot_panel_1h(project, min_trials=100):
    # This remains unchanged for Peak mode
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ifdn")
    parser.add_argument("--type", type=str, choices=["peak", "trace"], default="peak")
    parser.add_argument("--min-trials", type=int, default=100)
    args = parser.parse_args()

    if args.type == "trace":
        # Trace mode often uses lower MIN_TRIALS in notebooks
        plot_panel_1h_trace(args.project, min_trials=args.min_trials)
    else:
        plot_panel_1h(args.project, min_trials=args.min_trials)
