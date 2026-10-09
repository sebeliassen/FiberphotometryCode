import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class Behavior2VecEmbedder:
    """
    Computes Pointwise Mutual Information (PMI) matrix over syllable transitions
    and performs SVD to obtain smooth contextual semantic embeddings.
    """
    def __init__(self, n_components: int = 16, window_size: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.window_size = window_size
        self.random_state = random_state
        self.embeddings: Optional[np.ndarray] = None
        self.syll_to_idx: Dict[int, int] = {}
        self.idx_to_syll: Dict[int, int] = {}
        self.vocab_size: int = 0

    def fit(self, syllable_sequences: List[np.ndarray]):
        all_sylls = set()
        for seq in syllable_sequences:
            valid = seq[~pd.isna(seq)]
            all_sylls.update(valid.astype(int).tolist())
        
        unique_sylls = sorted(list(all_sylls))
        self.vocab_size = len(unique_sylls)
        self.syll_to_idx = {s: i for i, s in enumerate(unique_sylls)}
        self.idx_to_syll = {i: s for i, s in enumerate(unique_sylls)}
        
        # Build co-occurrence matrix within context window
        co_occ = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        total_pairs = 0
        
        for seq in syllable_sequences:
            valid = seq[~pd.isna(seq)].astype(int)
            L = len(valid)
            for i in range(L):
                target = self.syll_to_idx[valid[i]]
                start_w = max(0, i - self.window_size)
                end_w = min(L, i + self.window_size + 1)
                for j in range(start_w, end_w):
                    if i != j:
                        ctx = self.syll_to_idx[valid[j]]
                        co_occ[target, ctx] += 1.0
                        total_pairs += 1.0
                        
        if total_pairs == 0:
            self.embeddings = np.zeros((self.vocab_size, self.n_components))
            return self

        # Compute Positive Pointwise Mutual Information (PPMI)
        p_ij = (co_occ + 1e-8) / total_pairs
        p_i = np.sum(p_ij, axis=1, keepdims=True)
        p_j = np.sum(p_ij, axis=0, keepdims=True)
        
        pmi = np.log(p_ij / (p_i @ p_j + 1e-12))
        ppmi = np.maximum(pmi, 0.0)
        
        # Truncated SVD for dense semantic vectors
        k_dim = min(self.n_components, self.vocab_size - 1)
        svd = TruncatedSVD(n_components=k_dim, random_state=self.random_state)
        emb = svd.fit_transform(ppmi)
        
        if emb.shape[1] < self.n_components:
            pad = np.zeros((self.vocab_size, self.n_components - emb.shape[1]))
            emb = np.hstack([emb, pad])
            
        # L2 normalize embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        self.embeddings = emb / norms
        return self

    def get_embedding(self, syllable_id: int) -> np.ndarray:
        if syllable_id in self.syll_to_idx and self.embeddings is not None:
            return self.embeddings[self.syll_to_idx[syllable_id]]
        return np.zeros(self.n_components, dtype=np.float32)


def extract_session_bouts(session, b2v: Optional[Behavior2VecEmbedder] = None) -> List[Dict]:
    """
    Extracts continuous 30-D feature vectors for all active syllable bouts in a single session.
    Features per bout:
      - 10D Postural PCA Centroid (latent_state_0 .. latent_state_9)
      - 3D Kinematics: log1p(velocity), log1p(|angular_velocity|), log1p(|acceleration|)
      - 1D Duration: log1p(duration_frames)
      - 16D Contextual Semantic Embedding (Behavior2Vec)
    """
    df = session.all_data_df
    syll_df = session.syllables
    
    first_valid = df['centroid_x'].first_valid_index()
    last_valid = df['centroid_x'].last_valid_index()
    if first_valid is None or last_valid is None:
        return []
        
    sub_df = df.loc[first_valid:last_valid].copy()
    sub_syll = syll_df.loc[first_valid:last_valid].copy()
    
    # Drop interleaved NA rows to get contiguous tracking samples
    valid_mask = ~pd.isna(sub_syll['syllable'])
    if not np.any(valid_mask):
        return []
        
    sub_df = sub_df[valid_mask].copy()
    sub_syll = sub_syll[valid_mask].copy()
    
    # Compute instantaneous acceleration
    vel = sub_df['velocity_px_s'].values
    dt = 1.0 / 40.0  # active sampling rate
    acc = np.gradient(vel, dt)
    sub_df['acceleration'] = acc
    
    pose_cols = [f'latent_state_{i}' for i in range(10)]
    pose_data = sub_syll[pose_cols].values
    syll_arr = sub_syll['syllable'].values.astype(int)
    frame_indices = sub_df.index.values
    
    # Run Length Encoding on genuine continuous bouts
    bouts = []
    curr_syll = syll_arr[0]
    start_idx = 0
    
    n_samples = len(syll_arr)
    for i in range(1, n_samples):
        if syll_arr[i] != curr_syll:
            end_idx = i
            dur_frames = end_idx - start_idx
            # Postural PCA Centroid (10D)
            mean_pose = np.nanmean(pose_data[start_idx:end_idx], axis=0)
            if np.isnan(mean_pose).any():
                mean_pose = np.nan_to_num(mean_pose, nan=0.0)
                
            # Kinematics
            mean_v = np.nanmean(sub_df['velocity_px_s'].values[start_idx:end_idx])
            mean_w = np.nanmean(np.abs(sub_df['angular_velocity'].values[start_idx:end_idx]))
            mean_a = np.nanmean(np.abs(sub_df['acceleration'].values[start_idx:end_idx]))
            
            # Contextual Semantic Embedding (16D)
            syll_int = int(curr_syll)
            emb = b2v.get_embedding(syll_int) if b2v is not None else np.zeros(16, dtype=np.float32)
                
            # 30-D vector
            feat_vec = np.hstack([
                mean_pose,                               # 10D
                np.log1p(max(0.0, mean_v)),              # 1D
                np.log1p(max(0.0, mean_w)),              # 1D
                np.log1p(max(0.0, mean_a)),              # 1D
                np.log1p(dur_frames),                    # 1D
                emb                                      # 16D
            ]).astype(np.float32)
            
            bouts.append({
                "syllable": syll_int,
                "start_frame": frame_indices[start_idx],
                "end_frame": frame_indices[end_idx - 1],
                "duration_frames": dur_frames,
                "mid_frame": (frame_indices[start_idx] + frame_indices[end_idx - 1]) // 2,
                "vector": feat_vec
            })
            
            curr_syll = syll_arr[i]
            start_idx = i
            
    # Tail bout
    if start_idx < n_samples:
        dur_frames = n_samples - start_idx
        mean_pose = np.nanmean(pose_data[start_idx:], axis=0)
        mean_pose = np.nan_to_num(mean_pose, nan=0.0)
        mean_v = np.nanmean(sub_df['velocity_px_s'].values[start_idx:])
        mean_w = np.nanmean(np.abs(sub_df['angular_velocity'].values[start_idx:]))
        mean_a = np.nanmean(np.abs(sub_df['acceleration'].values[start_idx:]))
        syll_int = int(curr_syll)
        emb = b2v.get_embedding(syll_int) if b2v is not None else np.zeros(16, dtype=np.float32)
        feat_vec = np.hstack([
            mean_pose,
            np.log1p(max(0.0, mean_v)),
            np.log1p(max(0.0, mean_w)),
            np.log1p(max(0.0, mean_a)),
            np.log1p(dur_frames),
            emb
        ]).astype(np.float32)
        bouts.append({
            "syllable": syll_int,
            "start_frame": frame_indices[start_idx],
            "end_frame": frame_indices[n_samples - 1],
            "duration_frames": dur_frames,
            "mid_frame": (frame_indices[start_idx] + frame_indices[n_samples - 1]) // 2,
            "vector": feat_vec
        })
        
    return bouts


def build_and_cache_dataset_bouts(
    sessions: List, 
    cache_path: str = "syllable_generation/analysis/language_models/output/continuous_bout_cache.pkl",
    force_recompute: bool = False
) -> Tuple[List[List[Dict]], Behavior2VecEmbedder]:
    """
    Builds and caches the 30-D continuous bout representation for all sessions.
    """
    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            print(f"[CACHE HIT] Loaded {len(data['session_bouts'])} sessions from {cache_path}")
            return data["session_bouts"], data["b2v"]
            
    print("[BUILDING] Fitting Behavior2Vec syllable contextual embeddings across all sessions...")
    syll_seqs = []
    for s in sessions:
        first_valid = s.all_data_df['centroid_x'].first_valid_index()
        last_valid = s.all_data_df['centroid_x'].last_valid_index()
        if first_valid is not None and last_valid is not None:
            syll_seqs.append(s.syllables.loc[first_valid:last_valid, 'syllable'].values)
            
    b2v = Behavior2VecEmbedder(n_components=16, window_size=2)
    b2v.fit(syll_seqs)
    
    print("[BUILDING] Extracting continuous 30-D bout features per session...")
    all_session_bouts = []
    for s in sessions:
        bouts = extract_session_bouts(s, b2v)
        all_session_bouts.append(bouts)
        
    total_bouts = sum(len(b) for b in all_session_bouts)
    print(f"[DONE] Extracted {total_bouts} total bouts across {len(all_session_bouts)} sessions.")
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"session_bouts": all_session_bouts, "b2v": b2v}, f)
    print(f"[CACHED] Saved to {cache_path}")
    
    return all_session_bouts, b2v
