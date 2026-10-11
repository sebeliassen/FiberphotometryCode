import os
import pickle
import sys

import numpy as np
from sklearn.decomposition import NMF

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")

with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

past_counts = []
spatial_features = []

for s_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1

    frame_sylls = np.zeros(N, dtype=np.int32)
    frame_spatial = np.zeros((N, 2), dtype=np.float32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 2:
            frame_spatial[sf : ef + 1, 0] = vec[0]
            frame_spatial[sf : ef + 1, 1] = vec[1]

    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0

    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    cum_spatial = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])

    for t in range(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        sp_mean = (cum_spatial[t] - cum_spatial[t - W_FRAMES]) / float(W_FRAMES)
        past_counts.append(w_b)
        spatial_features.append(sp_mean)

X_counts = np.array(past_counts, dtype=np.float32)
Spatial = np.array(spatial_features, dtype=np.float32)

print(f"Loaded X_counts: {X_counts.shape}, Spatial: {Spatial.shape}")

K = 4
nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
W = nmf.fit_transform(X_counts)
H = nmf.components_
W_probs = W / np.clip(np.sum(W, axis=1, keepdims=True), 1e-8, None)

mean_pop = np.mean(X_counts, axis=0) + 1e-4

for k in range(K):
    enrichment = H[k] / mean_pop
    top_sylls = np.argsort(enrichment)[::-1][:5]
    print(f"\nComponent {k+1}:")
    for s in top_sylls:
        print(f"  Syllable {s:2d}: fold-enrichment = {enrichment[s]:.2f}x (H loading = {H[k, s]:.3f})")

    # Spatial center/periphery check
    weights_k = W_probs[:, k]
    high_mask = weights_k > np.percentile(weights_k, 75)
    sp_k = Spatial[high_mask]
    print(f"  High expression windows: {np.sum(high_mask)}, mean x={np.mean(sp_k[:,0]):.2f}, mean y={np.mean(sp_k[:,1]):.2f}")
