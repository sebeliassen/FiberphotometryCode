from typing import Any, Dict, List, Optional  # noqa: F401 - retain historical module attributes

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# GPU acceleration if available
try:
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
    from cuml.cluster import KMeans as GPU_KMeans
    from cuml.manifold import UMAP as GPU_UMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class BaseStrategyClusterer:
    """ Base interface for strategy clustering models. """
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class GMMStrategyClusterer(BaseStrategyClusterer):
    """
    Gaussian Mixture Model on continuous SVD window embeddings.
    Supports 'diag' and 'full' covariance matrices.
    """
    def __init__(self, n_components: int = 9, covariance_type: str = "diag", svd_dim: int = 30, random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.svd_dim = svd_dim
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.model: Optional[GaussianMixture] = None
        self.bic_: float = 0.0
        self.aic_: float = 0.0

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > self.svd_dim:
            self.svd = TruncatedSVD(n_components=min(self.svd_dim, X.shape[1] - 1), random_state=self.random_state)
            X_proc = self.svd.fit_transform(X)
        else:
            X_proc = X

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100,
            n_init=3
        )
        labels = self.model.fit_predict(X_proc)
        self.bic_ = float(self.model.bic(X_proc))
        self.aic_ = float(self.model.aic(X_proc))
        return labels

class KMeansStrategyClusterer(BaseStrategyClusterer):
    """
    Euclidean or Spherical (Cosine-normalized) K-Means.
    """
    def __init__(self, n_clusters: int = 9, spherical: bool = False, svd_dim: int = 30, random_state: int = 42):
        self.n_clusters = n_clusters
        self.spherical = spherical
        self.svd_dim = svd_dim
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > self.svd_dim:
            self.svd = TruncatedSVD(n_components=min(self.svd_dim, X.shape[1] - 1), random_state=self.random_state)
            X_proc = self.svd.fit_transform(X)
        else:
            X_proc = X

        if self.spherical:
            X_proc = normalize(X_proc, norm='l2')

        if HAS_GPU:
            model = GPU_KMeans(n_clusters=self.n_clusters, random_state=self.random_state, max_iter=300)
            labels = model.fit_predict(X_proc)
            if hasattr(labels, 'get'):
                labels = labels.get()
        else:
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=5, max_iter=300)
            labels = model.fit_predict(X_proc)

        return np.asarray(labels, dtype=int)

class SpectralStrategyClusterer(BaseStrategyClusterer):
    """
    Spectral Clustering using nearest-neighbors affinity graph.
    """
    def __init__(self, n_clusters: int = 9, n_neighbors: int = 30, svd_dim: int = 30, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.svd_dim = svd_dim
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > self.svd_dim:
            self.svd = TruncatedSVD(n_components=min(self.svd_dim, X.shape[1] - 1), random_state=self.random_state)
            X_proc = self.svd.fit_transform(X)
        else:
            X_proc = X

        model = SpectralClustering(
            n_clusters=self.n_clusters,
            n_neighbors=self.n_neighbors,
            affinity="nearest_neighbors",
            random_state=self.random_state,
            n_init=5,
            assign_labels="kmeans"
        )
        labels = model.fit_predict(X_proc)
        return np.asarray(labels, dtype=int)

class AgglomerativeStrategyClusterer(BaseStrategyClusterer):
    """
    Hierarchical bottom-up clustering with Ward linkage.
    """
    def __init__(self, n_clusters: int = 9, svd_dim: int = 30):
        self.n_clusters = n_clusters
        self.svd_dim = svd_dim
        self.svd: Optional[TruncatedSVD] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > self.svd_dim:
            self.svd = TruncatedSVD(n_components=min(self.svd_dim, X.shape[1] - 1), random_state=42)
            X_proc = self.svd.fit_transform(X)
        else:
            X_proc = X

        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        labels = model.fit_predict(X_proc)
        return np.asarray(labels, dtype=int)

class BERTopicStrategyClusterer(BaseStrategyClusterer):
    """
    UMAP + HDBSCAN with Centroid Outlier Remap.
    """
    def __init__(self, n_neighbors: int = 15, min_cluster_size: int = 150, min_samples: int = 10, n_components: int = 10, random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_components = n_components
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > 50:
            svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=self.random_state)
            X_proc = svd.fit_transform(X)
        else:
            X_proc = X

        if HAS_GPU:
            u_model = GPU_UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=0.0, metric='euclidean', random_state=self.random_state)
            umap_emb = u_model.fit_transform(X_proc)
            if hasattr(umap_emb, 'get'):
                umap_emb = umap_emb.get()
            h_model = GPU_HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
            raw_labels = h_model.fit_predict(umap_emb)
            if hasattr(raw_labels, 'get'):
                raw_labels = raw_labels.get()
        else:
            import umap as cpu_umap
            from sklearn.cluster import HDBSCAN as CPU_HDBSCAN
            u_model = cpu_umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=0.0, metric='euclidean', random_state=self.random_state)
            umap_emb = u_model.fit_transform(X_proc)
            h_model = CPU_HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, cluster_selection_method='eom')
            raw_labels = h_model.fit_predict(umap_emb)

        raw_labels = np.asarray(raw_labels, dtype=int)
        unique_clusters = [c for c in set(raw_labels) if c != -1]
        if len(unique_clusters) == 0:
            return np.zeros(len(X), dtype=int)

        centroids = np.array([umap_emb[raw_labels == c].mean(axis=0) for c in unique_clusters])
        resolved_ids = raw_labels.copy()
        outlier_mask = (raw_labels == -1)
        if np.any(outlier_mask):
            outlier_coords = umap_emb[outlier_mask]
            dists = np.linalg.norm(outlier_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            closest_indices = np.argmin(dists, axis=1)
            resolved_ids[outlier_mask] = np.array(unique_clusters)[closest_indices]

        # Remap to continuous 0..K-1
        label_map = {old_c: new_idx for new_idx, old_c in enumerate(sorted(set(resolved_ids)))}
        mapped_ids = np.array([label_map[c] for c in resolved_ids], dtype=int)
        return mapped_ids
