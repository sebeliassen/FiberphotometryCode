import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

# 1. Load bout cache
cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

# 2. Load session metadata to filter WT Saline
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
if not os.path.exists(pkl_path):
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")

with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

wt_saline_indices = []
for i, s in enumerate(raw_sessions):
    meta = getattr(s, "metadata", {})
    gt = meta.get("genotype", "WT")
    drug = meta.get("drug", "Saline")
    if gt == "WT" and drug.lower() == "saline":
        entry_f = meta.get("entry_frame", 0)
        inj_f = meta.get("injection_frame", 0)
        wt_saline_indices.append((i, entry_f, inj_f, getattr(s, "session_id", f"sess_{i}")))

print(f"Found {len(wt_saline_indices)} WT Saline sessions:")
for idx, ent, inj, sid in wt_saline_indices:
    print(f"  Session {idx:2d}: ID={sid}, Entry={ent}, Inj={inj}")

# 3. Fit NMF K=4 on all evaluation windows to maintain consistent component definitions
FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

all_past_counts = []
for s_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
            
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    
    for t in range(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        all_past_counts.append(w_b)

X_all = np.array(all_past_counts, dtype=np.float32)
K = 4
nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
nmf.fit(X_all)

# 4. Extract time-aligned trajectory for each WT Saline session from entry_frame
timecourses = {}
max_time_min = 45.0  # inspect up to 45 min
t_grid = np.arange(0, max_time_min * 60, STRIDE_SEC)

for s_idx, entry_f, inj_f, sid in wt_saline_indices:
    bouts = session_bouts[s_idx]
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
            
    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    
    # Start tracking from entry_f + W_FRAMES
    sess_w_counts = []
    sess_times = []
    
    start_t = entry_f + W_FRAMES
    end_t = N
    
    for t in range(start_t, end_t, STRIDE_FRAMES):
        w_b = cum_bout[t] - cum_bout[t - W_FRAMES]
        sess_w_counts.append(w_b)
        sess_times.append((t - entry_f) / FPS / 60.0)  # minutes from entry
        
    if len(sess_w_counts) > 20:
        W_sess = nmf.transform(np.array(sess_w_counts, dtype=np.float32))
        W_sess_probs = W_sess / np.clip(np.sum(W_sess, axis=1, keepdims=True), 1e-8, None)
        timecourses[sid] = {
            "times_min": np.array(sess_times),
            "probs": W_sess_probs,
            "inj_min": (inj_f - entry_f) / FPS / 60.0 if inj_f > entry_f else 0.0
        }
        print(f"Session {sid[:30]}: {len(sess_times)} windows ({sess_times[0]:.1f} to {sess_times[-1]:.1f} min)")

print(f"\nSuccessfully extracted {len(timecourses)} WT Saline timecourses.")
