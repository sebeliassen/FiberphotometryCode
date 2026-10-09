"""
Kinematic Similarity Engine.

Pure math module — no plotting. Computes the (1 - Spearman Corr) distance matrix
and hierarchical linkage between syllable kinematic trajectories.
Used by both pnas_replication.py (for the dendrogram) and neural_pnas.py (for twins/groups).
"""
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/pnas_replication")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_or_build_kinematic_cache(sessions, project_name, top_sylls, max_pre=40, max_post=40):
    import pandas as pd
    cache_dir = os.path.join(OUTPUT_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{project_name}_kinematic_max_cache.parquet")
    
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
        
    print(f"  [Cache Miss] Building maximal [-{max_pre}, {max_post}] kinematic cache for {project_name}...")
    
    sample_s = sessions[0]
    syll_df_sample = sample_s.syllables
    if isinstance(syll_df_sample, dict): syll_df_sample = list(syll_df_sample.values())[0]
    feature_cols = [c for c in syll_df_sample.columns if 'latent_state' in c]
    feature_cols = sorted(feature_cols, key=lambda x: int(x.split('_')[-1]))
    if not feature_cols:
        feature_cols = [f"latent_state_{i}" for i in range(10)]

    all_rows = []
    
    for s in sessions:
        df = s.all_data_df
        if not hasattr(s, 'syllables'): continue
        syll_df = s.syllables
        if isinstance(syll_df, dict): syll_df = list(syll_df.values())[0]

        fi = df['frame_index'].values
        if len(fi) == 0: continue
        
        syll_slice = syll_df.loc[syll_df.index.isin(fi), feature_cols] if fi[0] in syll_df.index \
                     else syll_df.iloc[fi[0]:fi[-1]+1][feature_cols]
        df_latent = df.reset_index(drop=True).join(syll_slice.reset_index(drop=True))
        
        df_latent[feature_cols] = df_latent[feature_cols].interpolate(method='linear', limit=3)
        df_latent[feature_cols] = df_latent[feature_cols].ffill(limit=1).bfill(limit=1)

        df_latent_np = df_latent[feature_cols].to_numpy()
        sylls_np = df_latent['syllable'].to_numpy()
        onsets_np = df_latent['onset'].fillna(0).astype(bool).to_numpy()
        
        onset_positions = np.where(onsets_np)[0]
        valid_mask = (onset_positions - max_pre >= 0) & (onset_positions + max_post < len(df_latent_np))
        valid_positions = onset_positions[valid_mask]
        
        if len(valid_positions) == 0: continue
        pos_sylls = sylls_np[valid_positions]
        top_sylls_set = set(top_sylls)
        top_mask = np.array([s in top_sylls_set for s in pos_sylls])
        valid_positions = valid_positions[top_mask]
        pos_sylls = pos_sylls[top_mask]
        
        for pos, syll in zip(valid_positions, pos_sylls):
            start, end = pos - max_pre, pos + max_post
            chunk = df_latent_np[start:end]
            if not np.isnan(chunk[:, 0]).any():
                flat = chunk.flatten(order='F')
                row = [syll] + flat.tolist()
                all_rows.append(row)
                
    if not all_rows:
        return None
        
    col_names = ["syllable"]
    for feat in feature_cols:
        for offset in range(-max_pre, max_post):
            col_names.append(f"{feat}_offset_{offset}")
            
    df_cache = pd.DataFrame(all_rows, columns=col_names)
    df_cache.to_parquet(cache_path)
    return df_cache

def get_kinematic_similarity(sessions, project_name, top_sylls, pre=5, post=15):
    """
    Core helper: compute (1 - Spearman Corr) distance matrix between syllable kinematic
    trajectories. pre/post are in FRAMES.
    Utilizes intelligent parquet caching for instant [-40, 40] querying.
    Returns: (dist_matrix [N x N], valid_sylls [list], Z [linkage array])
    """
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import squareform
    from scipy.cluster import hierarchy
    import pandas as pd
    
    max_pre, max_post = 40, 40
    if pre > max_pre or post > max_post or pre < -max_pre or post < -max_post:
        print(f"Requested window {-pre}:{post} exceeds max cache limit {-max_pre}:{max_post}. Defaulting to limits.")
        pre = min(max_pre, max(pre, -max_pre))
        post = min(max_post, max(post, -max_post))

    df_cache = _get_or_build_kinematic_cache(sessions, project_name, top_sylls, max_pre=max_pre, max_post=max_post)
    if df_cache is None or df_cache.empty:
        return None, [], None

    feature_cols = [c for c in df_cache.columns if 'offset' in c]
    base_feats = list(dict.fromkeys([c.rsplit('_offset_', 1)[0] for c in feature_cols]))

    typical_matrices = []
    valid_sylls = []
    
    for syll in top_sylls:
        syll_df = df_cache[df_cache["syllable"] == syll]
        if len(syll_df) < 2: continue
        
        meds = []
        for feat in base_feats:
            feat_cols_in_window = [f"{feat}_offset_{offset}" for offset in range(-pre, post)]
            arr = syll_df[feat_cols_in_window].to_numpy()
            med = np.median(arr, axis=0)
            std = np.std(med)
            if std > 0: med = (med - np.mean(med)) / std
            meds.append(med)
            
        typical_matrices.append(np.concatenate(meds))
        valid_sylls.append(syll)

    print(f"  [get_kinematic_similarity] {len(valid_sylls)}/{len(top_sylls)} syllables had enough bouts.")
    if len(typical_matrices) < 2: return None, [], None

    Xs = np.stack(typical_matrices)
    spearman_corr, _ = scipy_stats.spearmanr(Xs.T)
    dist_matrix = 1.0 - spearman_corr
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    dist_condensed = squareform(dist_matrix)
    Z = hierarchy.linkage(dist_condensed, method="complete")

    return dist_matrix, valid_sylls, Z
