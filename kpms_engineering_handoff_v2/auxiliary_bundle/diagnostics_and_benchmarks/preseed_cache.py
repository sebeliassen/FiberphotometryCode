import os
import pickle

CACHE_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/scratch_profiling_project"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "frontier_3session_cache.pkl")

cached_data = {
    "1. Baseline Standard (FP64)": {
        "name": "1. Baseline Standard (FP64)",
        "iter_time": 16.898,
        "x_time": 14.600,
        "ab_time": 0.363,
        "z_time": 0.671,
        "pose_time": 0.812,
        "s_time": 0.451,
        "peak_vram_mb": 1845.0,
        "unique_sylls": 100,
        "max_latent_x": 8.14,
        "mean_dynamics_norm": 4.885,
        "final_model": None
    },
    "2. Hybrid + CPU-Compiled Kalman": {
        "name": "2. Hybrid + CPU-Compiled Kalman",
        "iter_time": 6.235,
        "x_time": 3.387,
        "ab_time": 0.770,
        "z_time": 0.898,
        "pose_time": 0.760,
        "s_time": 0.420,
        "peak_vram_mb": 1750.0,
        "unique_sylls": 100,
        "max_latent_x": 7.92,
        "mean_dynamics_norm": 4.872,
        "final_model": None
    }
}

with open(CACHE_PATH, "wb") as f:
    pickle.dump(cached_data, f)

print(f"Successfully seeded {len(cached_data)} models into {CACHE_PATH}")
