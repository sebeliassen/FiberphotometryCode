import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
from analysis.language_models.lib.preprocessing.segmentation import assign_frame_strategies, generate_sliding_windows
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def get_group_label(session) -> str:
    """
    Returns a unified experimental group label for cohort comparisons.
    """
    m = session.metadata
    genotype = str(m.get("genotype", "WT")).upper()
    drug = m.get("drug")
    group = m.get("group")

    if drug is not None and drug != "N/A" and str(drug).strip() != "":
        return f"{genotype}_{drug}"
    if group is not None and group != "N/A" and str(group).strip() != "":
        return f"{genotype}_{group}"
    return genotype

def get_primary_video_path(session) -> str:
    """
    Retrieves the primary video path from session metadata.
    """
    if hasattr(session, "video_refs_df") and session.video_refs_df is not None and not session.video_refs_df.empty:
        primary_mask = session.video_refs_df["role"] == "primary"
        if primary_mask.any():
            return str(session.video_refs_df.loc[primary_mask, "path"].values[0])
        return str(session.video_refs_df.iloc[0]["path"])
    return "unknown_video"

def subset_moseq_data(data, metadata, target_keys):
    """
    Subsets MoSeq data and metadata dicts to only target_keys.
    """
    keys, bounds = metadata
    target_set = set(target_keys)

    indices = []
    for idx, key in enumerate(keys):
        matched = [t for t in target_set if t in key]
        if matched:
            indices.append(idx)

    if not indices:
        raise ValueError("None of the target keys matched the checkpoint metadata.")

    indices = np.array(indices)

    # Subset data arrays (first dimension is chunk index)
    new_data = {
        "Y": data["Y"][indices],
        "mask": data["mask"][indices]
    }
    if "conf" in data:
        new_data["conf"] = data["conf"][indices]

    new_keys = keys[indices]
    new_bounds = bounds[indices].copy()

    # Reconstruct bounds to be contiguous starting at 0
    lengths = new_bounds[:, 1] - new_bounds[:, 0]
    cumulative_ends = np.cumsum(lengths)
    cumulative_starts = np.insert(cumulative_ends[:-1], 0, 0)
    new_bounds = np.column_stack((cumulative_starts, cumulative_ends))

    return new_data, (new_keys, new_bounds), indices

def load_or_compute_moseq_marginals(sessions, output_dir):
    """
    Loads estimated MoSeq syllable marginals from cache or computes them from checkpoint.
    """
    cache_path = os.path.join(output_dir, "moseq_marginals.pkl")
    marginals = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                marginals = pickle.load(f)
            print(f"Loaded {len(marginals)} cached MoSeq marginals from: {cache_path}")
        except Exception as read_err:
            print(f"Error reading cache, starting fresh: {read_err}")
            marginals = {}

    print("Computing MoSeq marginals from checkpoint...")
    # Prevent JAX VRAM preallocation
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Force CPU for JAX as GPU ordinal 0 is occupied by DeepLabCut
    # os.environ["JAX_PLATFORMS"] = "cpu"

    import keypoint_moseq as kpms

    checkpoint_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_full_run/checkpoint.h5"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MoSeq checkpoint not found at: {checkpoint_path}")

    model, data, metadata, iteration = kpms.load_checkpoint(path=checkpoint_path)
    project_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    conf = kpms.load_config(project_dir)

    # Extract target base session IDs
    target_sessions = []
    for s in sessions:
        sid = s.session_id
        # Remove pre/post suffix
        if sid.endswith("_pre"): sid = sid[:-4]
        elif sid.endswith("_post"): sid = sid[:-5]
        target_sessions.append(sid)

    conf.pop("error_estimator", None)
    conf.pop("verbose", None)
    error_est = model.get("error_estimator") or conf.get("error_estimator")

    # Filter target sessions to only those not in cached marginals keys
    uncomputed_sessions = [ts for ts in target_sessions if not any(ts in k for k in marginals.keys())]

    if len(uncomputed_sessions) == 0:
        print("All target sessions are already computed and loaded from cache.")
        return marginals

    batch_size = 5
    batches = [uncomputed_sessions[i : i + batch_size] for i in range(0, len(uncomputed_sessions), batch_size)]

    total_batches = len(batches)
    print(f"Divided {len(uncomputed_sessions)} uncomputed sessions into {total_batches} batches of size {batch_size}.")

    for idx, batch_sessions in enumerate(batches):
        print(f"Estimating marginals for batch {idx+1}/{total_batches} ({len(batch_sessions)} sessions)...")
        try:
            single_data, single_metadata, sub_idx = subset_moseq_data(data, metadata, batch_sessions)
            single_noise_prior = model["noise_prior"][sub_idx]

            # Gibbs sampling for marginal probabilities (running 50 burn-in + 50 sample iterations for speed)
            batch_marginal = kpms.fitting.estimate_syllable_marginals(
                model, single_data, single_metadata,
                noise_prior=single_noise_prior,
                error_estimator=error_est,
                burn_in_iters=50,
                num_samples=50,
                steps_per_sample=2,
                verbose=True,
                parallel_message_passing=False,
                **conf
            )
            for k, v in batch_marginal.items():
                marginals[k] = v

            # Save incrementally
            os.makedirs(output_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(marginals, f)
        except Exception as err:
            print(f"Error processing batch {idx+1}: {err}")

    return marginals

def extract_window_inputs(sessions: List, clusterer, vectorizer_type: str) -> Tuple[List, List]:
    """
    Slices frames into sliding windows and formats window representations based on vectorizer_type.
    """
    global _EXTRACTED_WINDOWS_CACHE
    if '_EXTRACTED_WINDOWS_CACHE' not in globals():
        globals()['_EXTRACTED_WINDOWS_CACHE'] = {}

    cache_key = (id(clusterer), vectorizer_type)
    if cache_key in globals()['_EXTRACTED_WINDOWS_CACHE']:
        return globals()['_EXTRACTED_WINDOWS_CACHE'][cache_key]

    all_window_inputs = []
    session_window_meta = []

    # Load marginals if using moseq_soft_count
    moseq_marginals = None
    if vectorizer_type == "moseq_soft_count":
        lib_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_dir = os.path.join(lib_dir, "output/strategies")
        moseq_marginals = load_or_compute_moseq_marginals(sessions, output_dir)

    for s in sessions:
        fs = s.metadata.get("fs", 80.0)
        group_label = get_group_label(s)
        video_path = get_primary_video_path(s)

        if vectorizer_type == "moseq_soft_count":
            # Match session ID to marginals key using substring
            matching_keys = [k for k in moseq_marginals.keys() if s.session_id.replace("_pre", "").replace("_post", "") in k]
            if not matching_keys:
                # Fallback: fill with zeros
                num_syllables = next(iter(moseq_marginals.values())).shape[1]
                soft_probs = np.zeros((len(s.all_data_df), num_syllables), dtype=np.float32)
            else:
                soft_probs = moseq_marginals[matching_keys[0]]

            n = len(s.all_data_df)
            w_size = int(128.0 * fs)
            stride = int(10.0 * fs)

            # Align lengths
            if len(soft_probs) != n:
                if len(soft_probs) > n:
                    soft_probs = soft_probs[:n]
                else:
                    padded = np.zeros((n, soft_probs.shape[1]), dtype=np.float32)
                    padded[:len(soft_probs)] = soft_probs
                    soft_probs = padded

            windows = []
            if n < w_size:
                windows.append({"start_frame": 0, "end_frame": n})
                win_probs = soft_probs.sum(axis=0)
                all_window_inputs.append(win_probs)
            else:
                start = 0
                while start + w_size <= n:
                    end = start + w_size
                    windows.append({"start_frame": start, "end_frame": end})
                    win_probs = soft_probs[start:end].sum(axis=0)
                    all_window_inputs.append(win_probs)
                    start += stride

        elif vectorizer_type == "soft_count" and hasattr(clusterer, "transform_soft"):
            # Transform to soft probabilities matrix of shape (num_frames, max_token_id + 1)
            soft_probs = clusterer.transform_soft(s)
            # Slice into sliding windows
            n = len(soft_probs)
            w_size = int(128.0 * fs)
            stride = int(10.0 * fs)

            windows = []
            if n < w_size:
                windows.append({"start_frame": 0, "end_frame": n})
                # sum soft probs over frames
                win_probs = soft_probs.sum(axis=0)
                all_window_inputs.append(win_probs)
            else:
                start = 0
                while start + w_size <= n:
                    end = start + w_size
                    windows.append({"start_frame": start, "end_frame": end})
                    win_probs = soft_probs[start:end].sum(axis=0)
                    all_window_inputs.append(win_probs)
                    start += stride
        elif vectorizer_type == "transition_base":
            base_sylls_filled = s.all_data_df["syllable"].fillna(-1).to_numpy().astype(int)
            windows = generate_sliding_windows(base_sylls_filled, fs=fs, window_size_sec=128.0, stride_sec=10.0)

            for win in windows:
                tokens = [t for t in win["tokens"] if t != -1]
                all_window_inputs.append(tokens)
        else:
            rich_tokens = clusterer.transform(s)
            windows = generate_sliding_windows(rich_tokens, fs=fs, window_size_sec=128.0, stride_sec=10.0)

            for win in windows:
                if vectorizer_type == "transition":
                    # Pass raw list of token IDs
                    all_window_inputs.append(win["tokens"])
                else:
                    # Pass space-separated string
                    doc_str = " ".join([clusterer.get_rich_token_str(tid) for tid in win["tokens"]])
                    all_window_inputs.append(doc_str)

        session_window_meta.append((len(windows), windows, fs, len(s.all_data_df), video_path, group_label, s.session_id))

    res = (all_window_inputs, session_window_meta)
    globals()['_EXTRACTED_WINDOWS_CACHE'][cache_key] = res
    return res

def extract_session_strategy_intervals(sessions: List, clusterer, strategy_model, output_csv_path: str):
    """
    Extracts contiguous intervals of each Strategy ID per session and writes to CSV.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    all_window_inputs, session_window_meta = extract_window_inputs(sessions, clusterer, strategy_model.vectorizer_type)

    if len(all_window_inputs) > 0:
        all_strat_ids, _ = strategy_model.transform(all_window_inputs)
    else:
        all_strat_ids = np.array([])

    rows = []
    strat_idx = 0
    for idx, s in enumerate(sessions):
        n_win, windows, fs, num_frames, video_path, group_label, session_id = session_window_meta[idx]
        if n_win == 0:
            continue

        strat_ids = all_strat_ids[strat_idx : strat_idx + n_win]
        strat_idx += n_win

        win_df = pd.DataFrame({
            "start_frame": [w["start_frame"] for w in windows],
            "end_frame": [w["end_frame"] for w in windows],
            "strategy_id": strat_ids
        })

        frame_strats = assign_frame_strategies(win_df, num_frames)

        i = 0
        n_frames = len(frame_strats)
        while i < n_frames:
            curr_strat = frame_strats[i]
            start_frame = i
            while i < n_frames and frame_strats[i] == curr_strat:
                i += 1
            end_frame = i

            rows.append({
                "session_id": session_id,
                "video_path": video_path,
                "strategy_id": int(curr_strat),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": float(start_frame / fs),
                "end_time_sec": float(end_frame / fs)
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Exported {len(df_out)} strategy intervals to: {output_csv_path}")

def evaluate_group_similarity(sessions: List, clusterer, strategy_model, output_dir: str):
    """
    Computes session-level strategy proportions and generates comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_topics = strategy_model.n_topics

    all_window_inputs, session_window_meta = extract_window_inputs(sessions, clusterer, strategy_model.vectorizer_type)

    if len(all_window_inputs) > 0:
        all_strat_ids, _ = strategy_model.transform(all_window_inputs)
    else:
        all_strat_ids = np.array([])

    session_data = []
    strat_idx = 0
    for idx, s in enumerate(sessions):
        n_win, windows, fs, num_frames, video_path, group_label, session_id = session_window_meta[idx]
        if n_win == 0:
            continue

        strat_ids = all_strat_ids[strat_idx : strat_idx + n_win]
        strat_idx += n_win

        win_df = pd.DataFrame({
            "start_frame": [w["start_frame"] for w in windows],
            "end_frame": [w["end_frame"] for w in windows],
            "strategy_id": strat_ids
        })

        frame_strats = assign_frame_strategies(win_df, num_frames)

        counts = np.zeros(n_topics)
        for t in range(n_topics):
            counts[t] = np.sum(frame_strats == t)

        total = np.sum(counts)
        props = counts / total if total > 0 else counts

        session_data.append({
            "session_id": session_id,
            "group_label": group_label,
            "proportions": props
        })

    if len(session_data) == 0:
        print("No valid session data for group similarity evaluation.")
        return

    df_props = pd.DataFrame([s["proportions"] for s in session_data])
    df_props.columns = [f"Strategy_{i}" for i in range(n_topics)]
    df_props["group_label"] = [s["group_label"] for s in session_data]
    df_props["session_id"] = [s["session_id"] for s in session_data]

    # Plot 1: Mean Strategy Proportions by Group
    mean_props = df_props.groupby("group_label").mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    mean_props.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Average Frame Proportion", fontsize=12)
    ax.set_xlabel("Experimental Group", fontsize=12)
    ax.set_title("Behavioral Strategy Composition by Cohort", fontsize=14)
    ax.legend(title="Strategies", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    bar_path = os.path.join(output_dir, "strategy_distribution_by_group.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cohort distribution bar plot to: {bar_path}")

    # Plot 2: Session Strategy Space (PCA on proportions)
    if len(session_data) >= 3:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df_props.iloc[:, :n_topics].to_numpy())

        fig, ax = plt.subplots(figsize=(10, 8))
        groups = df_props["group_label"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

        for g, color in zip(groups, colors):
            mask = df_props["group_label"] == g
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                label=g, s=60, alpha=0.85, edgecolors='black', color=color
            )

        ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title("Session Space Projected from Macro-Strategy Proportions", fontsize=14)
        ax.legend(title="Cohort / Genotype_Drug", fontsize=10, title_fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        pca_path = os.path.join(output_dir, "session_strategy_spaces.png")
        fig.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PCA session space plot to: {pca_path}")

def predict_experimental_groups(sessions: List, clusterer, strategy_model) -> str:
    """
    Compares the accuracy of predicting cohort groups using base syllables, 
    rich tokens, and macro strategy proportions.
    """
    n_topics = strategy_model.n_topics

    labels = []
    feat_syllables = []
    feat_rich_tokens = []
    feat_strategies = []

    # Build global vocabulary sets to align dimensions
    all_syllables_set = set()
    for s in sessions:
        df = s.all_data_df
        if "syllable" in df.columns:
            all_syllables_set.update(df["syllable"].dropna().unique())
    all_syllables = sorted(list(all_syllables_set - {-1}))
    syll_map = {syll: idx for idx, syll in enumerate(all_syllables)}

    max_rich_token = clusterer.next_token_id - 1

    # Extract window representations for all sessions
    all_window_inputs, session_window_meta = extract_window_inputs(sessions, clusterer, strategy_model.vectorizer_type)

    # Predict strategy IDs for all windows
    if len(all_window_inputs) > 0:
        all_strat_ids, _ = strategy_model.transform(all_window_inputs)
    else:
        all_strat_ids = np.array([])

    strat_idx = 0
    for idx, s in enumerate(sessions):
        df = s.all_data_df
        group_label = get_group_label(s)
        labels.append(group_label)

        # 1. Base syllable proportions
        if '_BASE_SYLL_VEC_CACHE' not in globals():
            globals()['_BASE_SYLL_VEC_CACHE'] = {}
        if s.session_id in globals()['_BASE_SYLL_VEC_CACHE']:
            syll_vec = globals()['_BASE_SYLL_VEC_CACHE'][s.session_id]
        else:
            syll_vec = np.zeros(len(all_syllables))
            if "syllable" in df.columns:
                syll_series = df["syllable"].dropna().to_numpy().astype(int)
                # Filter to valid syllables only
                valid_mask = np.isin(syll_series, list(syll_map.keys()))
                valid_sylls = syll_series[valid_mask]
                if len(valid_sylls) > 0:
                    unique_s, counts_s = np.unique(valid_sylls, return_counts=True)
                    for s_val, c_val in zip(unique_s, counts_s):
                        syll_vec[syll_map[s_val]] = c_val
                total_syll = np.sum(syll_vec)
                if total_syll > 0:
                    syll_vec /= total_syll
            globals()['_BASE_SYLL_VEC_CACHE'][s.session_id] = syll_vec
        feat_syllables.append(syll_vec)

        # 2. Rich tokens proportions
        clusterer_key = (getattr(clusterer, "feature_mode", "default"), getattr(clusterer, "cluster_model", "default"))
        cache_key = (clusterer_key, s.session_id)
        if '_RICH_VEC_CACHE' not in globals():
            globals()['_RICH_VEC_CACHE'] = {}
        if cache_key in globals()['_RICH_VEC_CACHE']:
            rich_vec = globals()['_RICH_VEC_CACHE'][cache_key]
        else:
            rich_tokens = clusterer.transform(s)
            rich_vec = np.zeros(max_rich_token)
            valid_mask = (rich_tokens >= 1) & (rich_tokens <= max_rich_token)
            valid_tokens = rich_tokens[valid_mask]
            if len(valid_tokens) > 0:
                unique_t, counts_t = np.unique(valid_tokens, return_counts=True)
                rich_vec[unique_t - 1] = counts_t
            total_rich = np.sum(rich_vec)
            if total_rich > 0:
                rich_vec /= total_rich
            globals()['_RICH_VEC_CACHE'][cache_key] = rich_vec
        feat_rich_tokens.append(rich_vec)

        # 3. Strategy proportions
        n_win, windows, fs, num_frames, video_path, _, _ = session_window_meta[idx]
        if n_win == 0:
            feat_strategies.append(np.zeros(n_topics))
            continue

        strat_ids = all_strat_ids[strat_idx : strat_idx + n_win]
        strat_idx += n_win

        win_df = pd.DataFrame({
            "start_frame": [w["start_frame"] for w in windows],
            "end_frame": [w["end_frame"] for w in windows],
            "strategy_id": strat_ids
        })

        frame_strats = assign_frame_strategies(win_df, num_frames)

        strat_vec = np.zeros(n_topics)
        for t in range(n_topics):
            strat_vec[t] = np.sum(frame_strats == t)
        total_strat = np.sum(strat_vec)
        if total_strat > 0:
            strat_vec /= total_strat
        feat_strategies.append(strat_vec)

    X_syll = np.array(feat_syllables)
    X_rich = np.array(feat_rich_tokens)
    X_strat = np.array(feat_strategies)
    y = np.array(labels)

    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or np.min(counts) < 2:
        return "Cannot run classification comparison: too few sessions or too few sessions per cohort group in the loaded dataset."

    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Helper to run cross validation with parallel jobs
    def run_cv(X_data, y_data):
        accs, f1s, precs, recs = [], [], [], []
        for train_idx, test_idx in skf.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            n_jobs = 1 if os.environ.get("PARALLEL_CPU") == "1" else -1
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recs.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))

        return {
            "Accuracy": np.mean(accs),
            "F1-Score": np.mean(f1s),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "Num Features": X_data.shape[1]
        }

    # Caching cross-validation to avoid redundant RF fittings
    if '_BASE_SYLLABLE_CACHE' not in globals():
        globals()['_BASE_SYLLABLE_CACHE'] = None
    if '_RICH_TOKEN_CACHE' not in globals():
        globals()['_RICH_TOKEN_CACHE'] = {}

    results = {}

    # 1. Base Syllables (static global features)
    if globals()['_BASE_SYLLABLE_CACHE'] is not None:
        results["Base Syllables"] = globals()['_BASE_SYLLABLE_CACHE']
    else:
        results["Base Syllables"] = run_cv(X_syll, y)
        globals()['_BASE_SYLLABLE_CACHE'] = results["Base Syllables"]

    # 2. Rich Tokens (static per clusterer)
    clusterer_key = (getattr(clusterer, "feature_mode", "default"), getattr(clusterer, "cluster_model", "default"))
    if clusterer_key in globals()['_RICH_TOKEN_CACHE']:
        results["Rich Tokens"] = globals()['_RICH_TOKEN_CACHE'][clusterer_key]
    else:
        results["Rich Tokens"] = run_cv(X_rich, y)
        globals()['_RICH_TOKEN_CACHE'][clusterer_key] = results["Rich Tokens"]

    # 3. BERTopic Strategies (varies per config)
    results["BERTopic Strategies"] = run_cv(X_strat, y)

    table_rows = []
    table_rows.append("| Feature Space | Num Features | Mean Accuracy | Weighted F1-Score | Weighted Precision | Weighted Recall |")
    table_rows.append("|---|---|---|---|---|---|")
    for name, metrics in results.items():
        table_rows.append(
            f"| **{name}** | {metrics['Num Features']} | {metrics['Accuracy']:.4f} | {metrics['F1-Score']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} |"
        )

    return "\n".join(table_rows)

def compute_stereotypy_dispersion(sessions: List, clusterer, strategy_model, output_dir: str):
    """
    Computes UMAP coordinate dispersion within strategy clusters for each session
    and summarizes by experimental cohort. Saves a bar plot.
    """
    all_window_inputs, session_window_meta = extract_window_inputs(sessions, clusterer, strategy_model.vectorizer_type)
    if len(all_window_inputs) == 0:
        return

    all_strat_ids, umap_embeddings = strategy_model.transform(all_window_inputs)

    # Calculate global centroids of each strategy in UMAP space
    unique_strats = np.unique(all_strat_ids)
    global_centroids = {}
    for s_id in unique_strats:
        if s_id == -1:
            continue
        mask = (all_strat_ids == s_id)
        if mask.any():
            global_centroids[s_id] = umap_embeddings[mask].mean(axis=0)

    session_dispersions = []
    strat_idx = 0
    for idx, s in enumerate(sessions):
        n_win, windows, fs, num_frames, video_path, group_label, session_id = session_window_meta[idx]
        if n_win == 0:
            continue

        s_strat_ids = all_strat_ids[strat_idx : strat_idx + n_win]
        s_umap = umap_embeddings[strat_idx : strat_idx + n_win]
        strat_idx += n_win

        # Calculate dispersion per strategy for this session
        s_disps = []
        for s_id in unique_strats:
            if s_id == -1 or s_id not in global_centroids:
                continue
            mask = (s_strat_ids == s_id)
            if mask.sum() >= 2: # Need at least 2 points to measure dispersion
                coords = s_umap[mask]
                centroid = global_centroids[s_id]
                # Mean Euclidean distance to centroid
                dist_mean = np.mean(np.linalg.norm(coords - centroid, axis=1))
                s_disps.append(dist_mean)

        if len(s_disps) > 0:
            session_dispersions.append({
                "session_id": session_id,
                "group_label": group_label,
                "dispersion": np.mean(s_disps)
            })

    if len(session_dispersions) == 0:
        print("No valid dispersion data computed.")
        return

    df_disp = pd.DataFrame(session_dispersions)
    mean_disp = df_disp.groupby("group_label")["dispersion"].agg(["mean", "std", "count"])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = mean_disp.index.tolist()
    means = mean_disp["mean"].to_numpy()
    stds = mean_disp["std"].fillna(0).to_numpy()

    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    ax.bar(groups, means, yerr=stds, align='center', alpha=0.8, color=colors, edgecolor='black', capsize=10)
    ax.set_ylabel("UMAP Space Dispersion Index (Lower = More Stereotyped)", fontsize=12)
    ax.set_xlabel("Experimental Group", fontsize=12)
    ax.set_title("Behavioral Stereotypy (UMAP Dispersion) by Cohort", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    disp_plot_path = os.path.join(output_dir, "strategy_stereotypy_by_group.png")
    fig.savefig(disp_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved stereotypy dispersion bar plot to: {disp_plot_path}")

    disp_summary_path = os.path.join(output_dir, "strategy_stereotypy_summary.txt")
    with open(disp_summary_path, "w") as f:
        f.write("UMAP Strategy Space Dispersion Summary (Lower values = higher stereotypy):\n\n")
        f.write(mean_disp.to_markdown())

    print(f"Saved stereotypy summary to: {disp_summary_path}")

def plot_orbits(sessions: List, clusterer, strategy_model, output_dir: str):
    """
    Plots the continuous UMAP strategy space trajectory for a Saline and Amphetamine session.
    """
    all_window_inputs, session_window_meta = extract_window_inputs(sessions, clusterer, strategy_model.vectorizer_type)
    if len(all_window_inputs) == 0:
        return

    all_strat_ids, umap_embeddings = strategy_model.transform(all_window_inputs)

    # Locate representative sessions: WT_Saline vs WT_Amphetamine
    saline_idx = None
    amph_idx = None

    for idx, meta in enumerate(session_window_meta):
        group_label = meta[5]
        n_win = meta[0]
        if n_win > 20: # Require a decent trajectory length
            if "Saline" in group_label and saline_idx is None:
                saline_idx = idx
            elif "Amphetamine" in group_label and amph_idx is None:
                amph_idx = idx

    if saline_idx is None or amph_idx is None:
        # Fallback to any two distinct groups if WT_Saline/WT_Amphetamine not found
        for idx, meta in enumerate(session_window_meta):
            n_win = meta[0]
            if n_win > 10:
                if saline_idx is None:
                    saline_idx = idx
                elif amph_idx is None:
                    amph_idx = idx
                    break

    if saline_idx is None or amph_idx is None:
        print("Not enough sessions with long trajectories to plot orbits.")
        return

    # Helper to reconstruct where in the flat embeddings list each session's windows are
    strat_idx_map = []
    curr_idx = 0
    for meta in session_window_meta:
        n_win = meta[0]
        strat_idx_map.append((curr_idx, curr_idx + n_win))
        curr_idx += n_win

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    for ax, s_idx, title_prefix in zip(axes, [saline_idx, amph_idx], ["Saline Control", "Amphetamine Treated"]):
        n_win, windows, fs, num_frames, video_path, group_label, session_id = session_window_meta[s_idx]
        start_idx, end_idx = strat_idx_map[s_idx]

        s_umap = umap_embeddings[start_idx:end_idx]
        s_strat_ids = all_strat_ids[start_idx:end_idx]

        # Plot background: all windows as small grey dots
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c='lightgrey', s=5, alpha=0.3, label="Global Manifold")

        # Plot the trajectory as a line
        ax.plot(s_umap[:, 0], s_umap[:, 1], c='blue', linewidth=1.5, alpha=0.6, zorder=2)

        # Scatter the trajectory points colored by strategy ID
        scatter = ax.scatter(s_umap[:, 0], s_umap[:, 1], c=s_strat_ids, cmap='tab20', s=35, edgecolor='black', alpha=0.9, zorder=3, vmin=0, vmax=20)

        # Mark start and end points
        ax.scatter(s_umap[0, 0], s_umap[0, 1], c='green', marker='^', s=150, label='Start', zorder=4)
        ax.scatter(s_umap[-1, 0], s_umap[-1, 1], c='red', marker='s', s=120, label='End', zorder=4)

        ax.set_title(f"{title_prefix}\nSession: {session_id} ({group_label})", fontsize=12)
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    orbit_path = os.path.join(output_dir, "session_strategy_orbits.png")
    fig.savefig(orbit_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved continuous strategy orbits plot to: {orbit_path}")
