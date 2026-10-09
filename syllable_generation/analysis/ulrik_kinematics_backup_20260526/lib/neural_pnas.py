import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.ulrik_kinematics.lib import metrics
from analysis.ulrik_kinematics.lib.kinematic_similarity import get_kinematic_similarity
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/pnas_replication")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _get_top_syllables(sessions, k=20):
    from collections import Counter
    c = Counter()
    for s in sessions:
        b = metrics.get_syllable_instances(s.all_data_df)
        if not b.empty:
            c.update(b.to_dict())
    return [x[0] for x in c.most_common(k)]

def _get_latent_states(sessions, top_sylls):
    """Get mean latent state representation for each top syllable."""
    sample_s = sessions[0]
    syll_df_sample = sample_s.syllables
    if isinstance(syll_df_sample, dict): syll_df_sample = list(syll_df_sample.values())[0]

    latent_cols = [c for c in syll_df_sample.columns if 'latent_state' in c]
    if not latent_cols:
        latent_cols = [f"latent_state_{i}" for i in range(10)]
    else:
        latent_cols = sorted(latent_cols, key=lambda x: int(x.split('_')[-1]))

    syll_vectors = {syll: [] for syll in top_sylls}

    for s in sessions:
        df = s.all_data_df.copy()
        if not hasattr(s, 'syllables'): continue
        syll_df = s.syllables
        if isinstance(syll_df, dict): syll_df = list(syll_df.values())[0]

        merged = df.join(syll_df[latent_cols])

        for syll in top_sylls:
            mask = merged['syllable'] == syll
            if mask.sum() > 0:
                mean_vec = merged.loc[mask, latent_cols].mean(skipna=True).values
                if not np.isnan(mean_vec).any():
                    syll_vectors[syll].append(mean_vec)

    final_vectors = {}
    for syll, vecs in syll_vectors.items():
        if vecs:
            final_vectors[syll] = np.mean(vecs, axis=0)

    return final_vectors

def find_kinematic_twins_and_groups(sessions, project_name, top_k=20, n_groups=7, global_kinematics=None):
    """
    Identifies the most kinematically similar syllables (twins)
    using the EXACT same distance metric as the Fig 3H Heatmap (1 - Spearman Corr).
    """
    if global_kinematics is not None:
        dist_matrix, valid_sylls, Z, top_sylls = global_kinematics
    else:
        top_sylls = _get_top_syllables(sessions, k=top_k)
        # Use the shared helper from pnas_replication to ensure 100% parity with the heatmap
        dist_matrix, valid_sylls, Z = get_kinematic_similarity(sessions, project_name, top_sylls)

    if dist_matrix is None: return [], {}

    # 1. Kinematic Twins (top K unique pairs)
    sq_dist = dist_matrix.copy()
    np.fill_diagonal(sq_dist, np.inf)

    pairs = []
    used = set()

    # Sort all distances (ascending distance = descending similarity)
    flat_indices = np.argsort(sq_dist, axis=None)
    for idx in flat_indices:
        i, j = np.unravel_index(idx, sq_dist.shape)
        if i not in used and j not in used and i != j:
            pairs.append((valid_sylls[i], valid_sylls[j]))
            used.add(i)
            used.add(j)
        if len(pairs) >= 5: # Get top 5 pairs
            break

    # 2. Hierarchical Groups (for RFC)
    # We use the Z (linkage) already computed by get_kinematic_similarity
    labels = hierarchy.fcluster(Z, n_groups, criterion='maxclust')
    groups = {valid_sylls[i]: labels[i] for i in range(len(valid_sylls))}

    return pairs, groups

def extract_peths(sessions, target_syllables, window_s=1.0, cache_key=None):
    """
    Extracts PETHs for specific syllables from all valid functional fibers.
    Returns a dataframe of traces.
    """
    import hashlib
    import pickle

    if cache_key:
        cache_dir = os.path.join(os.path.dirname(__file__), "../cache")
        os.makedirs(cache_dir, exist_ok=True)
        syll_str = "_".join(map(str, sorted(target_syllables)))
        syll_hash = hashlib.md5(syll_str.encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{cache_key}_{syll_hash}_{window_s}s.pkl")

        if os.path.exists(cache_path):
            print(f"  [Cache] Loading extracted traces for {cache_key}...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    traces = []

    for s in sessions:
        fs = s.metadata.get("fs", 80.0)
        window_frames = int(window_s * fs)

        df = s.all_data_df
        if "onset" not in df.columns: continue

        # Identify functional signals
        sig_meta = s.metadata.get("signal_meta", {})
        functional_channels = []
        for key, info in sig_meta.items():
            if info.get("orig_letter") == "G":
                reg = info.get("brain_reg", "UNK").upper()
                # Classify into broad DS / VS categories
                if "DMS" in reg or "DLS" in reg or "DS" in reg: broad_reg = "DS"
                elif "VS" in reg or "NAC" in reg: broad_reg = "VS"
                else: broad_reg = "UNK"

                functional_channels.append({
                    "col": key, # usually signal_X
                    "zcol": f"{key}_zscore_datta" if f"{key}_zscore_datta" in df.columns else f"{key}_zscore",
                    "broad_reg": broad_reg,
                    "hemi": info.get("hemisphere", "UNK").upper()
                })

        if not functional_channels: continue

        df_indexed = df.set_index("frame_index")

        for syll in target_syllables:
            onsets = df[(df["onset"] > 0) & (df["syllable"] == syll)]["frame_index"].values

            for onset in onsets:
                start_idx = onset - window_frames
                end_idx = onset + window_frames

                # Check boundaries using the original index limits
                if start_idx < df_indexed.index.min() or end_idx > df_indexed.index.max(): continue

                # Fast O(log N) lookup instead of O(N) boolean mask
                window_df = df_indexed.loc[start_idx:end_idx]

                for ch in functional_channels:
                    zcol = ch["zcol"]
                    if zcol not in window_df.columns: continue

                    # Ensure trace is exactly the right length by matching to theoretical frame_indices
                    series = pd.Series(index=np.arange(start_idx, end_idx + 1), dtype=float)
                    # Assign values where frame_index matches
                    series.update(window_df[zcol])

                    trace = series.values

                    if np.isnan(trace).all(): continue

                    traces.append({
                        "Animal": s.metadata.get("animal_id"),
                        "Genotype": s.metadata.get("genotype"),
                        "Drug": "Baseline" if s.metadata.get("drug", "N/A") in ["N/A", None] else s.metadata.get("drug"),
                        "Syllable": syll,
                        "Region": ch["broad_reg"],
                        "Hemisphere": ch["hemi"],
                        "Trace": trace
                    })

    if cache_key:
        with open(cache_path, "wb") as f:
            pickle.dump(traces, f)

    return traces

def plot_kinematic_vs_neural(sessions, project_name, global_kinematics=None):
    """
    Plots the DA signatures of the kinematically similar syllables using Alphabet labels.
    """
    print("Plotting Kinematic vs Neural Paradox...")

    if global_kinematics is not None:
        _, _, _, top_sylls = global_kinematics
    else:
        # Get alphabet mapping to match Fig 3H Dendrogram exactly (based on bout frequency)
        top_sylls = _get_top_syllables(sessions, k=20)

    letters = 'abcdefghijklmnopqrst'
    alphabet_map = {syll: letters[i] for i, syll in enumerate(top_sylls)}

    twins_list, _ = find_kinematic_twins_and_groups(sessions, project_name, global_kinematics=global_kinematics)
    if not twins_list:
        print("Could not identify kinematic twins.")
        return

    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    twins_dir = os.path.join(group_dir, "paradox_twins")
    os.makedirs(twins_dir, exist_ok=True)

    for pair_idx, twins in enumerate(twins_list):
        twin_a, twin_b = twins
        label_a = alphabet_map.get(twin_a, f"Syll {twin_a}")
        label_b = alphabet_map.get(twin_b, f"Syll {twin_b}")

        print(f"Plotting Kinematic Twins Pair {pair_idx+1}: {label_a} and {label_b}")

        # 2. Extract PETHs
        window_s = 1.0
        traces = extract_peths(sessions, [twin_a, twin_b], window_s=window_s, cache_key=f"{project_name}_twins_{pair_idx}")
        if not traces:
            print(f"No valid traces extracted for pair {label_a} and {label_b}.")
            continue

        df_traces = pd.DataFrame(traces)
        drugs = df_traces["Drug"].unique()

        fs = sessions[0].metadata.get("fs", 80.0)
        time_axis = np.linspace(-window_s, window_s, int(2 * window_s * fs) + 1)
        regions = ["DS", "VS"]
        hemispheres = ["L", "R"]
        palette = {twin_a: "#4c72b0", twin_b: "#dd8452"}

        for drug in drugs:
            sub_df_traces = df_traces[df_traces["Drug"] == drug]
            if sub_df_traces.empty: continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

            for i, reg in enumerate(regions):
                for j, hemi in enumerate(hemispheres):
                    ax = axes[i, j]

                    sub = sub_df_traces[(sub_df_traces["Region"] == reg) & (sub_df_traces["Hemisphere"] == hemi)]
                    if sub.empty:
                        ax.set_title(f"{reg} - {hemi} (No Data)")
                        continue

                    for syll in [twin_a, twin_b]:
                        syll_sub = sub[sub["Syllable"] == syll]
                        if syll_sub.empty: continue

                        stacked = np.vstack(syll_sub["Trace"].values)
                        mean_trace = np.nanmean(stacked, axis=0)
                        sem_trace = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])

                        lbl = alphabet_map.get(syll, f"Syll {syll}")
                        ax.plot(time_axis, mean_trace, label=f"Syllable {lbl}", color=palette[syll], lw=2)
                        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=palette[syll], alpha=0.2)

                    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                    ax.set_title(f"{reg} - {hemi} Hemisphere")

                    if i == 1: ax.set_xlabel("Time from onset (s)")
                    if j == 0: ax.set_ylabel("DA Z-Score")

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12)

            plt.suptitle(f"Kinematic Twins Pair {pair_idx+1} ({project_name} - {drug})\n({label_a} and {label_b} are Kinematically identical, Neurally distinct)", y=1.08)
            sns.despine()
            plt.tight_layout()
            drug_suffix = str(drug).lower().replace(" ", "_").replace("/", "")
            plt.savefig(os.path.join(twins_dir, f"{project_name.lower()}_{drug_suffix}_pair_{pair_idx+1}_{label_a}_vs_{label_b}.png"), dpi=300, bbox_inches="tight")
            plt.close()

def run_random_forest(sessions, project_name, global_kinematics=None):
    """
    Fig 3I Replica: Random Forest predicting Syllable vs Kinematic Group.
    """
    print("Running Random Forest Classifier (Fig 3I Replica)...")

    if global_kinematics is not None:
        _, _, _, top_sylls = global_kinematics
    else:
        top_sylls = _get_top_syllables(sessions, k=20)

    _, groups = find_kinematic_twins_and_groups(sessions, project_name, top_k=20, n_groups=7, global_kinematics=global_kinematics)

    if not groups: return

    # We will use a smaller window for RFC to avoid overfitting, e.g. [-0.5, 0.5]
    window_s = 0.5
    traces = extract_peths(sessions, top_sylls, window_s=window_s, cache_key=f"{project_name}_rfc")
    if not traces: return

    df_traces = pd.DataFrame(traces)
    drugs = df_traces["Drug"].unique()

    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)

    def _evaluate_rfc(X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

        accs = []
        shufs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            accs.append(clf.score(X_test, y_test))

            y_shuf = np.random.permutation(y_train)
            clf.fit(X_train, y_shuf)
            shufs.append(clf.score(X_test, y_test))

        return np.mean(accs), np.mean(shufs)

    for drug in drugs:
        sub_df = df_traces[df_traces["Drug"] == drug]
        X = np.vstack(sub_df["Trace"].values)
        y_syll = sub_df["Syllable"].values
        y_group = np.array([groups.get(s, -1) for s in y_syll])

        valid_mask = y_group != -1
        X = X[valid_mask]
        y_syll = y_syll[valid_mask]
        y_group = y_group[valid_mask]

        if len(X) < 50:
            print(f"Not enough samples for RFC for {drug}.")
            continue

        print(f"Training RFC on {len(X)} instances for {drug}...")
        acc_syll, shuf_syll = _evaluate_rfc(X, y_syll)
        acc_group, shuf_group = _evaluate_rfc(X, y_group)

        fold_syll = acc_syll / shuf_syll if shuf_syll > 0 else 0
        fold_group = acc_group / shuf_group if shuf_group > 0 else 0

        plt.figure(figsize=(6, 5))
        bars = plt.bar(["Individual Syllable", "Kinematic Group"], [fold_syll, fold_group], color=["#dd8452", "#4c72b0"])

        plt.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
        plt.ylabel("Prediction Accuracy (Fold over shuffle)")
        plt.title(f"Fig 3I: Classifier Accuracy ({project_name} - {drug})\n(DA predicts Identity better than Kinematics)")

        sns.despine()
        plt.tight_layout()
        drug_suffix = str(drug).lower().replace(" ", "_").replace("/", "")
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig3i_rfc_classifier_{drug_suffix}.png"), dpi=300)
        plt.close()

        print(f"RFC Results ({drug}) -> Syllable: {fold_syll:.2f}x, Group: {fold_group:.2f}x")

def run_temporal_permutation_importance(sessions, project_name, global_kinematics=None):
    """
    Calculates the temporal permutation importance (Option B) for Syllable Identity decoding.
    Includes positive and negative verification tests.
    """
    print("Running Temporal Permutation Importance (Saliency)...")

    if global_kinematics is not None:
        _, _, _, top_sylls = global_kinematics
    else:
        top_sylls = _get_top_syllables(sessions, k=20)

    window_s = 0.5
    traces = extract_peths(sessions, top_sylls, window_s=window_s, cache_key=f"{project_name}_rfc")
    if not traces: return

    df_traces = pd.DataFrame(traces)
    drugs = sorted(df_traces["Drug"].unique())

    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)

    fs = sessions[0].metadata.get("fs", 80.0)
    time_axis = np.linspace(-window_s, window_s, len(df_traces.iloc[0]["Trace"]))

    SMOOTHING_SIGMA = 2.0

    plt.figure(figsize=(8, 5))
    palette = {"Saline": "#4c72b0", "Amphetamine": "#dd8452"}

    export_rows = []

    for drug in drugs:
        sub_df = df_traces[df_traces["Drug"] == drug]
        X = np.vstack(sub_df["Trace"].values)
        y_syll = sub_df["Syllable"].values

        MAX_SAMPLES = 10000
        if len(X) > MAX_SAMPLES:
            X, y_syll = resample(X, y_syll, n_samples=MAX_SAMPLES, random_state=42, stratify=y_syll)

        if len(X) < 50:
            print(f"Not enough samples for permutation importance for {drug}.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y_syll, test_size=0.2, random_state=42, stratify=y_syll)

        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        print(f"  Calculating Real Permutation Importance for {drug}...")
        result_real = permutation_importance(clf, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)

        print(f"  Calculating Null Permutation Importance for {drug}...")
        y_scrambled = np.random.permutation(y_test)
        result_null = permutation_importance(clf, X_test, y_scrambled, n_repeats=3, random_state=42, n_jobs=-1)

        mean_real = result_real.importances_mean
        std_real = result_real.importances_std
        mean_null = result_null.importances_mean
        std_null = result_null.importances_std

        corrected_mean = mean_real - mean_null
        corrected_std = np.sqrt(std_real**2 + std_null**2)

        if SMOOTHING_SIGMA > 0:
            from scipy.ndimage import gaussian_filter1d
            corrected_mean = gaussian_filter1d(corrected_mean, sigma=SMOOTHING_SIGMA)
            corrected_std = gaussian_filter1d(corrected_std, sigma=SMOOTHING_SIGMA)

        color = palette.get(drug, "#55a868")
        plt.plot(time_axis, corrected_mean, label=f"{drug} Saliency", color=color, lw=2)
        plt.fill_between(time_axis, corrected_mean - corrected_std, corrected_mean + corrected_std, color=color, alpha=0.3)

        for t, m, s in zip(time_axis, corrected_mean, corrected_std):
            export_rows.append({"Drug": drug, "Time_s": t, "Corrected_Saliency": m, "Corrected_Std": s})

    plt.axhline(0, color="grey", linestyle="--", alpha=0.5, label="Null Baseline")
    plt.axvline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(f"Temporal Action Identity Saliency ({project_name})\n(When does DA predict the syllable?)")
    plt.xlabel("Time from onset (s)")
    plt.ylabel("Corrected Accuracy Drop (Real - Shuffle)")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_temporal_saliency_comparison.png"), dpi=300)
    plt.close()

    if export_rows:
        export_df = pd.DataFrame(export_rows)
        export_df.to_csv(os.path.join(group_dir, f"{project_name.lower()}_temporal_saliency_comparison.csv"), index=False)

def run_regional_temporal_saliency(sessions, project_name, global_kinematics=None):
    """
    Calculates the temporal permutation importance split by Dorsal vs Ventral Striatum.
    This directly investigates whether VS drives the early 'Planning' peak and DS drives
    the late 'Execution' peak.
    """
    print("Running Regional Temporal Saliency (DS vs VS)...")

    if global_kinematics is not None:
        _, _, _, top_sylls = global_kinematics
    else:
        top_sylls = _get_top_syllables(sessions, k=20)

    window_s = 0.5
    # We use the same cache since extract_peths pulls all regions anyway
    traces = extract_peths(sessions, top_sylls, window_s=window_s, cache_key=f"{project_name}_rfc")
    if not traces: return

    df_traces = pd.DataFrame(traces)
    if "Region" not in df_traces.columns:
        print("No Region column found in extracted traces. Cannot run regional saliency.")
        return

    drugs = sorted(df_traces["Drug"].unique())
    regions = ["VS", "DS"] # Ventral first (planning), Dorsal second (execution)

    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)

    fs = sessions[0].metadata.get("fs", 80.0)
    time_axis = np.linspace(-window_s, window_s, len(df_traces.iloc[0]["Trace"]))

    SMOOTHING_SIGMA = 2.0

    # Standard colors for regions
    region_palette = {"VS": "#d62728", "DS": "#1f77b4"} # Red-ish for VS, Blue-ish for DS

    export_rows = []

    for drug in drugs:
        plt.figure(figsize=(8, 5))

        for region in regions:
            sub_df = df_traces[(df_traces["Drug"] == drug) & (df_traces["Region"] == region)]
            if sub_df.empty: continue

            X = np.vstack(sub_df["Trace"].values)
            y_syll = sub_df["Syllable"].values

            MAX_SAMPLES = 10000
            if len(X) > MAX_SAMPLES:
                X, y_syll = resample(X, y_syll, n_samples=MAX_SAMPLES, random_state=42, stratify=y_syll)

            if len(X) < 50:
                print(f"Not enough samples for regional permutation importance ({drug} - {region}).")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y_syll, test_size=0.2, random_state=42, stratify=y_syll)

            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

            print(f"  [{region}] Calculating Real Permutation Importance for {drug}...")
            result_real = permutation_importance(clf, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)

            print(f"  [{region}] Calculating Null Permutation Importance for {drug}...")
            y_scrambled = np.random.permutation(y_test)
            result_null = permutation_importance(clf, X_test, y_scrambled, n_repeats=3, random_state=42, n_jobs=-1)

            mean_real = result_real.importances_mean
            std_real = result_real.importances_std
            mean_null = result_null.importances_mean
            std_null = result_null.importances_std

            corrected_mean = mean_real - mean_null
            corrected_std = np.sqrt(std_real**2 + std_null**2)

            if SMOOTHING_SIGMA > 0:
                from scipy.ndimage import gaussian_filter1d
                corrected_mean = gaussian_filter1d(corrected_mean, sigma=SMOOTHING_SIGMA)
                corrected_std = gaussian_filter1d(corrected_std, sigma=SMOOTHING_SIGMA)

            color = region_palette.get(region, "#55a868")
            plt.plot(time_axis, corrected_mean, label=f"{region} Saliency", color=color, lw=2)
            plt.fill_between(time_axis, corrected_mean - corrected_std, corrected_mean + corrected_std, color=color, alpha=0.3)

            for t, m, s in zip(time_axis, corrected_mean, corrected_std):
                export_rows.append({"Drug": drug, "Region": region, "Time_s": t, "Corrected_Saliency": m, "Corrected_Std": s})

        plt.axhline(0, color="grey", linestyle="--", alpha=0.5, label="Null Baseline")
        plt.axvline(0, color="black", linestyle="--", alpha=0.5)
        plt.title(f"Regional Saliency: Ventral vs Dorsal Striatum ({drug})\n(Planning vs Execution Encoding)")
        plt.xlabel("Time from onset (s)")
        plt.ylabel("Corrected Accuracy Drop (Real - Shuffle)")
        plt.legend()
        sns.despine()
        plt.tight_layout()

        drug_suffix = str(drug).lower().replace(" ", "_").replace("/", "")
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig3i_regional_saliency_{drug_suffix}.png"), dpi=300)
        plt.close()

    if export_rows:
        export_df = pd.DataFrame(export_rows)
        export_df.to_csv(os.path.join(group_dir, f"{project_name.lower()}_regional_saliency.csv"), index=False)


def run_replication(sessions, project_name, global_kinematics=None):
    print(f"\n=== Starting Neural Replications ({project_name}) ===")
    plot_kinematic_vs_neural(sessions, project_name, global_kinematics=global_kinematics)
    run_random_forest(sessions, project_name, global_kinematics=global_kinematics)
    run_temporal_permutation_importance(sessions, project_name, global_kinematics=global_kinematics)
    run_regional_temporal_saliency(sessions, project_name, global_kinematics=global_kinematics)
    print(f"Completed Neural Replications for {project_name}.")
