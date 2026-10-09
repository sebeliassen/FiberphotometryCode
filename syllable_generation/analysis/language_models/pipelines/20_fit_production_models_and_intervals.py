"""
Fit & Export Clean Production Strategy Models (KMeans & GMM on K=8, K=16)
File: syllable_generation/analysis/language_models/pipelines/20_fit_production_models_and_intervals.py

- Loads 14,092 clean 128s transition windows.
- Fits the 4 Bayesian Champion configurations:
    1. KMeans (K=8)
    2. KMeans (K=16)
    3. GMM (K=8)
    4. GMM (K=16)
- Exports trained models to `output/strategies_extended/production_models/`
- Computes frame-level and bout-level strategy intervals for all 30 sessions:
    - `session_strategy_intervals_kmeans_k8.csv`
    - `session_strategy_intervals_kmeans_k16.csv`
    - `session_strategy_intervals_gmm_k8.csv`
    - `session_strategy_intervals_gmm_k16.csv`
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
MODEL_DIR = os.path.join(OUT_DIR, "production_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data
print("Loading continuous bout cache and window scalar cache...")
with open(os.path.join(OUT_DIR, "continuous_bout_cache.pkl"), "rb") as f:
    bout_data = pickle.load(f)
session_bouts = bout_data["session_bouts"]

with open(os.path.join(OUT_DIR, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
raw_trans_inputs = wdata["trans_inputs"] # (14092, 10000)
win_meta = wdata["meta"]

pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

# 2. Define Best Bayesian Model Configurations
CHAMPIONS = {
    "kmeans_k8": {
        "family": "KMeans",
        "k": 8,
        "svd_dim": 96,
        "norm_type": "l1",
        "laplace_smooth": 0.0357,
        "spherical": False,
        "n_init": 20
    },
    "kmeans_k16": {
        "family": "KMeans",
        "k": 16,
        "svd_dim": 24,
        "norm_type": "l1",
        "laplace_smooth": 0.0085,
        "spherical": False,
        "n_init": 20
    },
    "gmm_k8": {
        "family": "GMM",
        "k": 8,
        "svd_dim": 24,
        "norm_type": "log1p",
        "laplace_smooth": 0.002,
        "cov_type": "spherical",
        "reg_covar": 0.01
    },
    "gmm_k16": {
        "family": "GMM",
        "k": 16,
        "svd_dim": 32,
        "norm_type": "none",
        "laplace_smooth": 0.0867,
        "cov_type": "spherical",
        "reg_covar": 0.0115
    }
}

win_sids = np.array([w["session_id"] for w in win_meta])
win_mids = np.array([w["mid_frame"] for w in win_meta])

# 3. Fit Each Champion and Generate Intervals
for model_name, cfg in CHAMPIONS.items():
    print(f"\n=======================================================")
    print(f"Fitting Champion Model: {model_name} (K={cfg['k']})")
    print(f"Config: {cfg}")
    print(f"=======================================================")
    
    # Preprocessing
    mat = raw_trans_inputs + cfg["laplace_smooth"]
    if cfg["norm_type"] == "log1p":
        mat = np.log1p(mat)
    elif cfg["norm_type"] == "l1":
        mat = normalize(mat, norm="l1")
    elif cfg["norm_type"] == "l2":
        mat = normalize(mat, norm="l2")
        
    svd = TruncatedSVD(n_components=cfg["svd_dim"], random_state=42)
    features = svd.fit_transform(mat)
    
    if cfg["family"] == "KMeans":
        clf = KMeans(n_clusters=cfg["k"], init="k-means++", n_init=cfg["n_init"], random_state=42)
        labels = clf.fit_predict(features)
        probs = np.eye(cfg["k"])[labels]
    else:
        clf = GaussianMixture(
            n_components=cfg["k"], covariance_type=cfg["cov_type"],
            reg_covar=cfg["reg_covar"], init_params="kmeans", random_state=42
        )
        clf.fit(features)
        probs = clf.predict_proba(features)
        labels = np.argmax(probs, axis=1)
        
    # Save Model Artifact
    model_save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": clf, "svd": svd, "config": cfg, "window_labels": labels, "window_probs": probs}, f)
    print(f"Saved model artifact to: {model_save_path}")
    
    # 4. Generate Continuous Intervals for Every Session
    all_intervals = []
    
    for s_idx, session in enumerate(raw_sessions):
        sid = session.session_id
        df = session.all_data_df
        total_frames = len(df)
        
        # Get bouts for this session
        bouts = session_bouts[s_idx]
        if len(bouts) == 0:
            continue
            
        cand_mask = (win_sids == sid)
        if not np.any(cand_mask):
            continue
            
        cand_indices = np.where(cand_mask)[0]
        cand_win_mids = win_mids[cand_indices]
        
        # Map each bout to closest window
        bout_strats = []
        bout_confs = []
        for b in bouts:
            mid = b["mid_frame"]
            closest_idx = cand_indices[np.argmin(np.abs(cand_win_mids - mid))]
            st = labels[closest_idx]
            conf = probs[closest_idx, st]
            bout_strats.append(st)
            bout_confs.append(conf)
            
        # Merge adjacent identical strategy bouts into continuous intervals
        FPS_TRACKING = 80.025
        curr_strat = bout_strats[0]
        curr_start_frame = int(bouts[0]["start_frame"])
        curr_start_time = float(curr_start_frame / FPS_TRACKING)
        curr_confs = [bout_confs[0]]
        curr_bout_count = 1
        
        for b_i in range(1, len(bouts)):
            b = bouts[b_i]
            st = bout_strats[b_i]
            conf = bout_confs[b_i]
            
            if st == curr_strat:
                curr_confs.append(conf)
                curr_bout_count += 1
            else:
                end_frame = int(bouts[b_i - 1]["end_frame"])
                end_time = float(end_frame / FPS_TRACKING)
                all_intervals.append({
                    "session_id": sid,
                    "strategy_id": curr_strat,
                    "start_frame": curr_start_frame,
                    "end_frame": end_frame,
                    "start_time_sec": curr_start_time,
                    "end_time_sec": end_time,
                    "duration_sec": end_time - curr_start_time,
                    "num_bouts": curr_bout_count,
                    "mean_confidence": float(np.mean(curr_confs))
                })
                curr_strat = st
                curr_start_frame = int(b["start_frame"])
                curr_start_time = float(curr_start_frame / FPS_TRACKING)
                curr_confs = [conf]
                curr_bout_count = 1
                
        # Final interval
        end_frame = int(bouts[-1]["end_frame"])
        end_time = float(end_frame / FPS_TRACKING)
        all_intervals.append({
            "session_id": sid,
            "strategy_id": curr_strat,
            "start_frame": curr_start_frame,
            "end_frame": end_frame,
            "start_time_sec": curr_start_time,
            "end_time_sec": end_time,
            "duration_sec": end_time - curr_start_time,
            "num_bouts": curr_bout_count,
            "mean_confidence": float(np.mean(curr_confs))
        })
        
    df_intervals = pd.DataFrame(all_intervals)
    csv_path = os.path.join(OUT_DIR, f"session_strategy_intervals_{model_name}.csv")
    df_intervals.to_csv(csv_path, index=False)
    print(f"Generated {len(df_intervals)} continuous strategy epochs across 30 sessions.")
    print(f"Saved intervals to: {csv_path}")

print("\n✅ Phase 1 Complete: All 4 production models & intervals successfully exported.")
