import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import scipy.stats as stats

from analysis.ulrik_kinematics.lib.neural_pnas import extract_peths, _get_top_syllables
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

def _find_true_peak(curve, time_axis, sigma_smooth=4.0):
    """
    Finds the inflection point peak to avoid edge artifacts.
    """
    smoothed = gaussian_filter1d(curve, sigma=sigma_smooth)
    
    # Find peaks with some prominence to avoid tiny wiggles
    prominence_thresh = max(0.001, (np.max(smoothed) - np.min(smoothed)) * 0.1)
    peaks, properties = find_peaks(smoothed, prominence=prominence_thresh)
    
    if len(peaks) == 0:
        # Fallback to argmax if no true peaks found (e.g., flat or single slope)
        # but avoid the absolute edges
        search_range = smoothed[5:-5]
        if len(search_range) > 0:
            peak_idx = np.argmax(search_range) + 5
            return time_axis[peak_idx]
        return time_axis[np.argmax(smoothed)]
        
    # If multiple peaks, take the one with the highest prominence or highest amplitude
    best_peak_idx = peaks[np.argmax(smoothed[peaks])]
    return time_axis[best_peak_idx]

def calculate_peak_timing_stats(sessions, project_name, is_verification=False):
    """
    Extracts PETHs, computes class-specific saliency, finds peak timing for VS and DS,
    and runs a paired t-test on the 20 syllables.
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
    
    for drug in drugs:
        print(f"  Processing {drug}...")
        sub_df = df_traces[df_traces["Drug"] == drug]
        
        region_results = {}
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
            
            # Find peaks
            peak_times = {}
            for c, curve in class_saliency.items():
                peak_time = _find_true_peak(curve, time_axis, sigma_smooth=4.0)
                peak_times[c] = peak_time
                
            region_results[region] = peak_times
            
        if "VS" not in region_results or "DS" not in region_results:
            print("  Missing VS or DS for this drug. Skipping.")
            continue
            
        # Align syllables
        common_sylls = set(region_results["VS"].keys()).intersection(set(region_results["DS"].keys()))
        vs_times = []
        ds_times = []
        syll_ids = []
        
        for c in common_sylls:
            syll_ids.append(c)
            vs_times.append(region_results["VS"][c])
            ds_times.append(region_results["DS"][c])
            
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
