import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from analysis.ulrik_kinematics.lib.neural_pnas import _get_top_syllables, extract_peths
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/pnas_replication")

def _compute_class_specific_saliency(clf, X_test, y_test):
    """
    Computes permutation importance manually for each class.
    Returns: Dictionary mapping syllable ID -> saliency array across time.
    """
    n_features = X_test.shape[1]
    unique_classes = np.unique(y_test)

    # 1. Base accuracy per class
    base_preds = clf.predict(X_test)
    base_acc_per_class = {}
    for c in unique_classes:
        mask = (y_test == c)
        base_acc_per_class[c] = np.mean(base_preds[mask] == c) if mask.sum() > 0 else 0

    # 2. Permute each feature and measure drop
    class_saliency = {c: np.zeros(n_features) for c in unique_classes}

    for f_idx in range(n_features):
        drops = {c: [] for c in unique_classes}
        for r in range(5):  # 5 repeats for stability
            X_shuf = X_test.copy()
            np.random.shuffle(X_shuf[:, f_idx])
            shuf_preds = clf.predict(X_shuf)

            for c in unique_classes:
                mask = (y_test == c)
                if mask.sum() > 0:
                    shuf_acc = np.mean(shuf_preds[mask] == c)
                    drops[c].append(base_acc_per_class[c] - shuf_acc)

        for c in unique_classes:
            if len(drops[c]) > 0:
                class_saliency[c][f_idx] = np.mean(drops[c])

    return class_saliency

def _find_true_peak_strict(curve, time_axis, sigma_smooth=4.0):
    """
    Finds a strictly interior local maximum peak in the smoothed curve.
    Returns (peak_time, peak_idx) or (None, None) if no valid interior peak
    having lower values to both its immediate left and right exists.
    """
    smoothed = gaussian_filter1d(curve, sigma=sigma_smooth)

    # Find peaks with some prominence to avoid tiny noise wiggles
    prominence_thresh = max(0.001, (np.max(smoothed) - np.min(smoothed)) * 0.1)
    peaks, properties = find_peaks(smoothed, prominence=prominence_thresh)

    if len(peaks) == 0:
        return None, None

    # Take the peak with the highest amplitude
    best_peak_idx = peaks[np.argmax(smoothed[peaks])]

    # Ensure it is strictly inside the bounds (interior only, having lower values left and right)
    if best_peak_idx == 0 or best_peak_idx == len(smoothed) - 1:
        return None, None

    return time_axis[best_peak_idx], best_peak_idx

def _find_true_peak_naive(curve, time_axis, sigma_smooth=4.0):
    """
    Finds the peak timing, falling back to boundaries/argmax if no interior peak is found.
    (This is our old naive perturbation peak detection strategy).
    """
    smoothed = gaussian_filter1d(curve, sigma=sigma_smooth)
    prominence_thresh = max(0.001, (np.max(smoothed) - np.min(smoothed)) * 0.1)
    peaks, properties = find_peaks(smoothed, prominence=prominence_thresh)

    if len(peaks) == 0:
        search_range = smoothed[5:-5]
        if len(search_range) > 0:
            peak_idx = np.argmax(search_range) + 5
            return time_axis[peak_idx], peak_idx
        idx = np.argmax(smoothed)
        return time_axis[idx], idx

    best_peak_idx = peaks[np.argmax(smoothed[peaks])]
    return time_axis[best_peak_idx], best_peak_idx

def _find_true_peak(curve, time_axis, sigma_smooth=4.0, mode="strict"):
    """
    Wrapper peak detector that switches between the strict interior local max
    and the old naive boundary fallback perturbation strategy.
    """
    if mode == "strict":
        return _find_true_peak_strict(curve, time_axis, sigma_smooth=sigma_smooth)
    else:
        return _find_true_peak_naive(curve, time_axis, sigma_smooth=sigma_smooth)

def calculate_peak_timing_stats(sessions, project_name, is_verification=False, mode="strict"):
    """
    Extracts PETHs, computes class-specific saliency, finds peak timing for VS and DS,
    and runs a paired t-test on the 20 syllables.
    Excludes syllables without valid interior local peaks for both regions and saves
    individual syllable debug plots to direct visual validation.
    """
    print(f"\n--- Running Regional Timing Statistics (DS vs VS) for {project_name} ---")
    if is_verification:
        print(">>> VERIFICATION MODE: Intentionally shuffling temporal relationships to test failure.")

    top_sylls = _get_top_syllables(sessions, k=20)
    window_s = 0.5
    traces = extract_peths(sessions, top_sylls, window_s=window_s, cache_key=f"{project_name}_rfc")

    if not traces:
        print("No traces extracted.")
        return

    df_traces = pd.DataFrame(traces)
    if "Region" not in df_traces.columns:
        print("No Region column. Cannot run.")
        return

    drugs = sorted(df_traces["Drug"].unique())
    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)

    time_axis = np.linspace(-window_s, window_s, len(df_traces.iloc[0]["Trace"]))

    # Establish debug plot folder inside the respective group subfolder to prevent parent clutter
    debug_dir = os.path.join(group_dir, "debug_syllable_peaks")
    os.makedirs(debug_dir, exist_ok=True)

    for drug in drugs:
        print(f"  Processing {drug}...")
        sub_df = df_traces[df_traces["Drug"] == drug]

        region_results = {}
        class_saliency_dict = {}

        for region in ["VS", "DS"]:
            reg_df = sub_df[sub_df["Region"] == region].copy()
            if reg_df.empty: continue

            X = np.vstack(reg_df["Trace"].values)
            y_syll = reg_df["Syllable"].values

            if is_verification and region == "DS":
                # Intentionally shift DS backward in time to break the VS-first relationship
                # We do this by rolling the traces so the peak happens earlier
                X = np.roll(X, shift=-20, axis=1) # shift 250ms left

            MAX_SAMPLES = 10000
            if len(X) > MAX_SAMPLES:
                X, y_syll = resample(X, y_syll, n_samples=MAX_SAMPLES, random_state=42, stratify=y_syll)

            if len(X) < 50: continue

            X_train, X_test, y_train, y_test = train_test_split(X, y_syll, test_size=0.2, random_state=42, stratify=y_syll)
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

            # Compute Saliency per class
            class_saliency = _compute_class_specific_saliency(clf, X_test, y_test)
            class_saliency_dict[region] = class_saliency

            # Find peaks
            peak_times = {}
            for c, curve in class_saliency.items():
                peak_time, peak_idx = _find_true_peak(curve, time_axis, sigma_smooth=4.0, mode=mode)
                peak_times[c] = (peak_time, peak_idx)

            region_results[region] = peak_times

        if "VS" not in region_results or "DS" not in region_results:
            print("  Missing VS or DS for this drug. Skipping.")
            continue

        # Align syllables and enforce strict interior peak filters
        common_sylls = set(region_results["VS"].keys()).intersection(set(region_results["DS"].keys()))
        vs_times = []
        ds_times = []
        syll_ids = []

        for c in common_sylls:
            vs_t, vs_idx = region_results["VS"][c]
            ds_t, ds_idx = region_results["DS"][c]

            # Exclude if either curve lacks a valid interior peak
            if vs_t is None or ds_t is None:
                print(f"    Syllable {c} excluded: VS peak = {vs_t}, DS peak = {ds_t} (Missing interior local max)")

                # Still write the debug plot so we can inspect why it failed
                plt.figure(figsize=(8, 5))
                vs_curve = class_saliency_dict["VS"][c]
                ds_curve = class_saliency_dict["DS"][c]
                plt.plot(time_axis * 1000, vs_curve, color="#d62728", alpha=0.3, linestyle=":")
                plt.plot(time_axis * 1000, gaussian_filter1d(vs_curve, 4.0), color="#d62728", lw=2, label="VS (smoothed)")
                plt.plot(time_axis * 1000, ds_curve, color="#1f77b4", alpha=0.3, linestyle=":")
                plt.plot(time_axis * 1000, gaussian_filter1d(ds_curve, 4.0), color="#1f77b4", lw=2, label="DS (smoothed)")
                plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
                plt.axvline(0, color="black", linestyle="-", alpha=0.4, label="Onset")
                plt.xlabel("Time Relative to Onset (ms)", fontsize=11)
                plt.ylabel("Saliency (Accuracy Drop)", fontsize=11)
                plt.title(f"Syllable {c} Timing Debug ({project_name} - {drug})\n[EXCLUDED: Missing Interior Local Peak]", fontsize=10, fontweight="bold", color="red")
                plt.legend(frameon=True, loc="upper right")
                sns.despine()
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, f"{project_name.lower()}_{drug.lower()}_syll_{c}_excluded.png"), dpi=120)
                plt.close()
                continue

            syll_ids.append(c)
            vs_times.append(vs_t)
            ds_times.append(ds_t)

            # Generate debug plots for valid paired syllables
            plt.figure(figsize=(8, 5))
            vs_curve = class_saliency_dict["VS"][c]
            ds_curve = class_saliency_dict["DS"][c]
            vs_smooth = gaussian_filter1d(vs_curve, 4.0)
            ds_smooth = gaussian_filter1d(ds_curve, 4.0)

            # Plot curves (raw dotted, smoothed solid)
            plt.plot(time_axis * 1000, vs_curve, color="#d62728", alpha=0.3, linestyle=":")
            plt.plot(time_axis * 1000, vs_smooth, color="#d62728", lw=2.5, label="VS (smoothed)")

            plt.plot(time_axis * 1000, ds_curve, color="#1f77b4", alpha=0.3, linestyle=":")
            plt.plot(time_axis * 1000, ds_smooth, color="#1f77b4", lw=2.5, label="DS (smoothed)")

            # Draw peak markers
            plt.axvline(vs_t * 1000, color="#d62728", linestyle="--", alpha=0.6)
            plt.scatter(vs_t * 1000, vs_smooth[vs_idx], color="#d62728", marker="*", s=180, zorder=5, label=f"VS Peak ({vs_t*1000:.1f}ms)")

            plt.axvline(ds_t * 1000, color="#1f77b4", linestyle="--", alpha=0.6)
            plt.scatter(ds_t * 1000, ds_smooth[ds_idx], color="#1f77b4", marker="*", s=180, zorder=5, label=f"DS Peak ({ds_t*1000:.1f}ms)")

            plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
            plt.axvline(0, color="black", linestyle="-", alpha=0.4, label="Onset")

            plt.xlabel("Time Relative to Onset (ms)", fontsize=11)
            plt.ylabel("Saliency (Accuracy Drop)", fontsize=11)
            plt.title(f"Syllable {c} Timing Debug ({project_name} - {drug})\nVS: {vs_t*1000:.1f}ms | DS: {ds_t*1000:.1f}ms | Shift: {(ds_t - vs_t)*1000:.1f}ms", fontsize=10, fontweight="bold")
            plt.legend(frameon=True, loc="upper right")
            sns.despine()
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, f"{project_name.lower()}_{drug.lower()}_syll_{c}_valid.png"), dpi=120)
            plt.close()

        if len(vs_times) < 3:
            print(f"  [{drug}] Skipping t-test: too few syllables with valid interior peaks (N={len(vs_times)}).")
            continue

        vs_times = np.array(vs_times)
        ds_times = np.array(ds_times)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(vs_times, ds_times)

        mean_vs = np.mean(vs_times)
        mean_ds = np.mean(ds_times)
        diff_ms = (mean_ds - mean_vs) * 1000

        print(f"  [{drug}] VS Mean Peak: {mean_vs*1000:.1f}ms | DS Mean Peak: {mean_ds*1000:.1f}ms | Diff: {diff_ms:.1f}ms")
        print(f"  [{drug}] Paired t-test (N={len(vs_times)}): t={t_stat:.3f}, p={p_val:.5f}")

        if is_verification:
            if diff_ms < 0:
                print("  [VERIFICATION SUCCESS] The test failed appropriately because we intentionally shifted DS!")
            else:
                print("  [VERIFICATION WARNING] The test still passed despite the perturbation.")

        # Export
        export_df = pd.DataFrame({
            "Syllable": syll_ids,
            "VS_Peak_Time_s": vs_times,
            "DS_Peak_Time_s": ds_times,
            "Drug": drug
        })
        export_df.to_csv(os.path.join(group_dir, f"{project_name.lower()}_regional_timing_stats_{str(drug).lower()}.csv"), index=False)

        # Plotting
        plt.figure(figsize=(6, 6))

        # Create paired plot
        for i in range(len(vs_times)):
            plt.plot([0, 1], [vs_times[i]*1000, ds_times[i]*1000], color='gray', alpha=0.5, zorder=1)

        plt.scatter(np.zeros(len(vs_times)), vs_times*1000, color="#d62728", s=60, alpha=0.8, zorder=2, label="Ventral Striatum")
        plt.scatter(np.ones(len(ds_times)), ds_times*1000, color="#1f77b4", s=60, alpha=0.8, zorder=2, label="Dorsal Striatum")

        plt.boxplot([vs_times*1000, ds_times*1000], positions=[0, 1], widths=0.4, showfliers=False,
                   boxprops=dict(alpha=0.3, color='black'), medianprops=dict(color='black', lw=2), zorder=0)

        plt.xticks([0, 1], ["Ventral\n(Planning)", "Dorsal\n(Execution)"], fontsize=12)
        plt.ylabel("Peak Saliency Time (ms from onset)", fontsize=12)

        title_suffix = "(Verification - Intentionally Broken)" if is_verification else ""
        sig_stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        plt.title(f"Syllable-by-Syllable Peak Timing ({project_name} - {drug})\nMean Shift: {diff_ms:.1f}ms | p-val: {p_val:.4f} {sig_stars}\n{title_suffix}", fontsize=11)

        # Add chance line
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)

        sns.despine()
        plt.tight_layout()

        suffix = "_verification" if is_verification else ""
        drug_suffix = str(drug).lower().replace(" ", "_").replace("/", "")
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_regional_timing_plot_{drug_suffix}{suffix}.png"), dpi=300)
        plt.close()

