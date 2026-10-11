from typing import List, Tuple

import numpy as np


def extract_bouts_from_session(session) -> List[dict]:
    """
    Extracts contiguous runs of the same syllable (excluding -1) and aggregates 
    their kinematics and postural states to create bout features.
    
    Args:
        session: Session object with s.all_data_df and s.syllables.
        
    Returns:
        List[dict]: List of bouts with aggregated features.
    """
    df = session.all_data_df
    if "syllable" not in df.columns:
        return []

    syllables = df["syllable"].fillna(-1).to_numpy().astype(int)
    n = len(syllables)
    if n == 0:
        return []

    # Identify postural latent state columns if present in session.syllables
    syll_df = getattr(session, "syllables", None)
    latent_cols = []
    if syll_df is not None:
        latent_cols = [c for c in syll_df.columns if "latent_state" in c]
        latent_cols = sorted(latent_cols, key=lambda x: int(x.split("_")[-1]))

    bouts = []
    i = 0
    while i < n:
        syll_id = syllables[i]
        if syll_id == -1:
            i += 1
            continue

        start_idx = i
        while i < n and syllables[i] == syll_id:
            i += 1
        end_idx = i

        duration = end_idx - start_idx
        bout_df = df.iloc[start_idx:end_idx]

        mean_vel = float(bout_df["velocity_px_s"].mean()) if "velocity_px_s" in df.columns else 0.0
        mean_ang = float(bout_df["angular_velocity"].abs().mean()) if "angular_velocity" in df.columns else 0.0
        mean_acc = float(bout_df["acceleration"].abs().mean()) if "acceleration" in df.columns else 0.0
        mean_dist = float(bout_df["dist_from_center"].mean()) if "dist_from_center" in df.columns else 0.0

        # Replace NaNs
        mean_vel = 0.0 if np.isnan(mean_vel) else mean_vel
        mean_ang = 0.0 if np.isnan(mean_ang) else mean_ang
        mean_acc = 0.0 if np.isnan(mean_acc) else mean_acc
        mean_dist = 0.0 if np.isnan(mean_dist) else mean_dist

        bout_dict = {
            "syllable": syll_id,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "duration_frames": duration,
            "mean_velocity": mean_vel,
            "mean_angular_velocity": mean_ang,
            "mean_acceleration": mean_acc,
            "mean_dist_from_center": mean_dist,
        }

        if syll_df is not None and len(latent_cols) > 0:
            bout_sylls = syll_df.iloc[start_idx:end_idx]
            for col in latent_cols:
                m_val = float(bout_sylls[col].mean())
                s_val = float(bout_sylls[col].std())
                bout_dict[f"{col}_mean"] = 0.0 if np.isnan(m_val) else m_val
                bout_dict[f"{col}_std"] = 0.0 if np.isnan(s_val) else s_val

        bouts.append(bout_dict)
    return bouts


# Feature indices that are heavily right-skewed. Applying log1p before StandardScaler
# makes the distributions near-Gaussian so that K-Means centroids are not pulled towards outliers.
_LOG1P_COLS = [0, 1, 2, 3]  # duration_frames, mean_velocity, mean_angular_velocity, mean_acceleration


class KinematicSyllableClustered:
    """
    Kinematic & Postural Sub-State Clustering (Baseline Micro-Fusion).
    Clusters bouts of each individual syllable using local K-Means or GMM.
    """
    def __init__(self, n_clusters: int = 3, min_bouts: int = 10, use_gpu: bool = True, random_state: int = 42,
                 feature_mode: str = "kinematic", cluster_model: str = "kmeans"):
        """
        Args:
            n_clusters: Target number of substates to cluster each syllable into.
            min_bouts: Minimum occurrences of a syllable to perform clustering (else k=1).
            use_gpu: If True, uses RAPIDS cuml when available (only for KMeans).
            random_state: Random seed for reproducibility.
            feature_mode: 'kinematic', 'postural', or 'hybrid'.
            cluster_model: 'kmeans' or 'gmm'.
        """
        self.n_clusters = n_clusters
        self.min_bouts = min_bouts
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.feature_mode = feature_mode
        self.cluster_model = cluster_model

        self.syllable_models = {}  # syllable_id -> (scaler, kmeans_or_gmm, rank_mapping)
        self.token_mapping = {}    # (syllable_id, substate_idx) -> rich_token_id
        self.token_id_to_str = {}  # rich_token_id -> string (e.g. "4_slow")
        self.next_token_id = 0
        self.latent_cols = []      # Cached list of latent columns

    def _get_feature_keys(self, latent_cols: List[str]) -> Tuple[List[str], List[int]]:
        """
        Returns feature keys and the indices that should be log1p-transformed.
        """
        if self.feature_mode == "postural" and len(latent_cols) > 0:
            feat_keys = ["duration_frames"]
            for col in latent_cols:
                feat_keys.append(f"{col}_mean")
                feat_keys.append(f"{col}_std")
            log1p_idxs = [0]
        elif self.feature_mode == "hybrid" and len(latent_cols) > 0:
            feat_keys = [
                "duration_frames",
                "mean_velocity",
                "mean_angular_velocity",
                "mean_acceleration",
                "mean_dist_from_center"
            ]
            for col in latent_cols:
                feat_keys.append(f"{col}_mean")
                feat_keys.append(f"{col}_std")
            log1p_idxs = [0, 1, 2, 3]
        else:
            # Fallback to pure kinematic features
            feat_keys = [
                "duration_frames",
                "mean_velocity",
                "mean_angular_velocity",
                "mean_acceleration",
                "mean_dist_from_center"
            ]
            log1p_idxs = [0, 1, 2, 3]
        return feat_keys, log1p_idxs

    def fit(self, sessions: List):
        """
        Extracts bouts across all provided sessions and fits a local K-Means or GMM 
        model for each unique syllable.
        """
        # Discover latent state columns from sessions
        self.latent_cols = []
        for s in sessions:
            syll_df = getattr(s, "syllables", None)
            if syll_df is not None:
                cols = [c for c in syll_df.columns if "latent_state" in c]
                if len(cols) > 0:
                    self.latent_cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
                    break

        all_bouts = []
        for s in sessions:
            all_bouts.extend(extract_bouts_from_session(s))

        # Group bouts by syllable ID
        from collections import defaultdict
        syllable_bouts = defaultdict(list)
        for b in all_bouts:
            syllable_bouts[b["syllable"]].append(b)

        # Determine clustering library
        cuml_available = False
        if self.use_gpu and self.cluster_model == "kmeans":
            try:
                import cuml
                from cuml.cluster import KMeans as GPUKMeans
                from cuml.preprocessing import StandardScaler as GPUStandardScaler
                cuml_available = True
                print("Using GPU-accelerated RAPIDS cuML for clustering.")
            except ImportError:
                print("RAPIDS cuML not found. Falling back to CPU scikit-learn.")

        if not cuml_available:
            from sklearn.cluster import KMeans as CPUKMeans
            from sklearn.preprocessing import StandardScaler as CPUStandardScaler

        feature_keys, log1p_idxs = self._get_feature_keys(self.latent_cols)

        self.syllable_models = {}
        self.token_mapping = {}
        self.token_id_to_str = {}
        self.next_token_id = 1  # Rich tokens start at index 1 to distinguish from -1

        for syll_id, bouts in sorted(syllable_bouts.items()):
            n_bouts = len(bouts)
            X = np.array([[b[k] for k in feature_keys] for b in bouts], dtype=np.float32)

            if n_bouts < self.min_bouts:
                # Too few occurrences, keep as single state
                self.syllable_models[syll_id] = (None, None, None)
                self.token_mapping[(syll_id, 0)] = self.next_token_id
                self.token_id_to_str[self.next_token_id] = f"{syll_id}_slow"
                self.next_token_id += 1
                continue

            k = min(self.n_clusters, n_bouts)

            # Apply log1p transform to designated columns
            X[:, log1p_idxs] = np.log1p(X[:, log1p_idxs])

            if cuml_available:
                scaler = GPUStandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = GPUKMeans(n_clusters=k, random_state=self.random_state)
                model.fit(X_scaled)
                preds = model.predict(X_scaled)
                if hasattr(preds, "get"):
                    preds = preds.get()
            else:
                scaler = CPUStandardScaler()
                X_scaled = scaler.fit_transform(X)
                if self.cluster_model == "gmm":
                    from sklearn.mixture import GaussianMixture
                    model = GaussianMixture(n_components=k, random_state=self.random_state, covariance_type='diag', reg_covar=1e-4)
                    model.fit(X_scaled)
                    preds = model.predict(X_scaled)
                else:
                    model = CPUKMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    model.fit(X_scaled)
                    preds = model.labels_

            # Compute average velocity of bouts in each cluster to sort them
            cluster_velocities = []
            for cluster_idx in range(k):
                bout_indices = np.where(preds == cluster_idx)[0]
                if len(bout_indices) > 0:
                    avg_v = np.mean([bouts[bi]["mean_velocity"] for bi in bout_indices])
                else:
                    avg_v = 0.0
                cluster_velocities.append(avg_v)
            sorted_idxs = np.argsort(cluster_velocities)

            if k == 2:
                labels = ["slow", "fast"]
            elif k == 3:
                labels = ["slow", "med", "fast"]
            elif k == 4:
                labels = ["slow", "med_low", "med_high", "fast"]
            else:
                labels = [f"substate_{r}" for r in range(k)]

            rank_mapping = {orig_idx: rank for rank, orig_idx in enumerate(sorted_idxs)}
            self.syllable_models[syll_id] = (scaler, model, rank_mapping)

            for rank in range(k):
                label_str = labels[rank]
                self.token_mapping[(syll_id, rank)] = self.next_token_id
                self.token_id_to_str[self.next_token_id] = f"{syll_id}_{label_str}"
                self.next_token_id += 1

        return self

    def get_rich_token_str(self, token_id: int) -> str:
        """
        Returns the string representation of a rich token ID.
        """
        return self.token_id_to_str.get(token_id, "unk")

    def transform_session_to_bouts(self, session) -> List[dict]:
        """
        Translates a session's raw dataframe into a list of bout dictionaries, 
        annotating each with its 'substate_idx' and mapped 'rich_token_id'.
        """
        bouts = extract_bouts_from_session(session)

        feature_keys, log1p_idxs = self._get_feature_keys(self.latent_cols)

        for b in bouts:
            syll_id = b["syllable"]
            if syll_id not in self.syllable_models:
                b["substate_idx"] = 0
                b["rich_token_id"] = -1
                continue

            model_info = self.syllable_models[syll_id]
            scaler, model, rank_mapping = model_info

            if scaler is None or model is None:
                b["substate_idx"] = 0
                b["rich_token_id"] = self.token_mapping.get((syll_id, 0), -1)
                continue

            X_val = np.array([[b[k] for k in feature_keys]], dtype=np.float32)
            X_val[:, log1p_idxs] = np.log1p(X_val[:, log1p_idxs])
            X_scaled = scaler.transform(X_val)

            if hasattr(model, "predict"):
                pred_idx = model.predict(X_scaled)
                if hasattr(pred_idx, "get"):
                    pred_idx = pred_idx.get()
                pred_idx = int(pred_idx[0])
            else:
                pred_idx = 0

            rank = rank_mapping[pred_idx] if rank_mapping is not None else 0

            b["substate_idx"] = rank
            b["rich_token_id"] = self.token_mapping.get((syll_id, rank), -1)

        return bouts

    def transform(self, session) -> np.ndarray:
        """
        Transforms the session dataframe into a frame-by-frame array of Rich Token IDs.
        """
        df = session.all_data_df
        n = len(df)
        rich_token_seq = np.full(n, -1, dtype=int)

        bouts = self.transform_session_to_bouts(session)
        for b in bouts:
            start = b["start_frame"]
            end = b["end_frame"]
            rich_token_seq[start:end] = b["rich_token_id"]

        return rich_token_seq

    def transform_soft(self, session) -> np.ndarray:
        """
        For GMM models, returns a frame-by-frame probability distribution over the 
        rich token vocabulary. Shape: (num_frames, max_token_id + 1).
        """
        df = session.all_data_df
        n = len(df)
        max_token_id = max(self.token_id_to_str.keys()) if self.token_id_to_str else 0

        probs_matrix = np.zeros((n, max_token_id + 1), dtype=np.float32)

        bouts = extract_bouts_from_session(session)
        feature_keys, log1p_idxs = self._get_feature_keys(self.latent_cols)

        for b in bouts:
            syll_id = b["syllable"]
            start = b["start_frame"]
            end = b["end_frame"]

            if syll_id not in self.syllable_models:
                continue

            scaler, model, rank_mapping = self.syllable_models[syll_id]
            if scaler is None or model is None:
                token_id = self.token_mapping.get((syll_id, 0), -1)
                if token_id != -1:
                    probs_matrix[start:end, token_id] = 1.0
                continue

            X_val = np.array([[b[k] for k in feature_keys]], dtype=np.float32)
            X_val[:, log1p_idxs] = np.log1p(X_val[:, log1p_idxs])
            X_scaled = scaler.transform(X_val)

            if self.cluster_model == "gmm" and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[0]
                for pred_idx, prob in enumerate(probs):
                    rank = rank_mapping[pred_idx] if rank_mapping is not None else 0
                    token_id = self.token_mapping.get((syll_id, rank), -1)
                    if token_id != -1:
                        probs_matrix[start:end, token_id] = prob
            else:
                pred_idx = model.predict(X_scaled)
                if hasattr(pred_idx, "get"):
                    pred_idx = pred_idx.get()
                pred_idx = int(pred_idx[0])
                rank = rank_mapping[pred_idx] if rank_mapping is not None else 0
                token_id = self.token_mapping.get((syll_id, rank), -1)
                if token_id != -1:
                    probs_matrix[start:end, token_id] = 1.0

        return probs_matrix
