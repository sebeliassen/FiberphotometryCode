import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from scipy.spatial.distance import jensenshannon

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")

print("Loading cache from:", cache_path)
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

print(f"Total sessions: {len(session_bouts)}")

# Build window dataset
X_list = []
Y_list = []
sess_ids = []

for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1
    
    frame_sylls = np.zeros(N, dtype=np.int32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf:ef+1] = s_id
        
    start_f = W_FRAMES
    end_f = N - H_FRAMES
    for center_f in range(start_f, end_f, STRIDE_FRAMES):
        w_sylls = frame_sylls[center_f - W_FRAMES : center_f]
        h_sylls = frame_sylls[center_f : center_f + H_FRAMES]
        
        # Past counts
        w_counts = np.bincount(w_sylls, minlength=NUM_SYLLS)[:NUM_SYLLS].astype(np.float32)
        # normalize to frequencies
        w_freq = w_counts / (np.sum(w_counts) + 1e-8)
        
        # Future distribution
        h_counts = np.bincount(h_sylls, minlength=NUM_SYLLS)[:NUM_SYLLS].astype(np.float32)
        h_dist = h_counts / (np.sum(h_counts) + 1e-8)
        
        X_list.append(w_freq)
        Y_list.append(h_dist)
        sess_ids.append(sess_idx)

X = np.array(X_list)
Y = np.array(Y_list)
sess_ids = np.array(sess_ids)

print(f"Dataset shape: X={X.shape}, Y={Y.shape}, Unique sessions={len(np.unique(sess_ids))}")

# Global null baseline for JS divergence
p_global = np.mean(Y, axis=0)
p_global = np.clip(p_global, 1e-8, None)
p_global /= np.sum(p_global)

def compute_js_div(y_true, y_pred):
    eps = 1e-8
    y_pred_clipped = np.clip(y_pred, eps, None)
    y_pred_norm = y_pred_clipped / np.sum(y_pred_clipped, axis=1, keepdims=True)
    
    js_vals = []
    for i in range(len(y_true)):
        m = 0.5 * (y_true[i] + y_pred_norm[i])
        # JS divergence using base e or natural log
        kl1 = np.sum(np.where(y_true[i] > 0, y_true[i] * np.log(y_true[i] / m), 0.0))
        kl2 = np.sum(np.where(y_pred_norm[i] > 0, y_pred_norm[i] * np.log(y_pred_norm[i] / m), 0.0))
        js_vals.append(0.5 * (kl1 + kl2))
    return np.mean(js_vals)

# Null error
null_preds = np.tile(p_global, (len(Y), 1))
d_null = compute_js_div(Y, null_preds)
print(f"Baseline Null JS Divergence: {d_null:.6f}")

# Full 100-D Bag-of-Words Repertoire Baseline (5-fold CV by session)
unique_sessions = np.unique(sess_ids)
np.random.seed(42)
shuffled_sess = np.random.permutation(unique_sessions)
folds = np.array_split(shuffled_sess, 5)

def evaluate_features_cv(F):
    js_gains = []
    for fold_idx, test_sess in enumerate(folds):
        train_mask = ~np.isin(sess_ids, test_sess)
        test_mask = np.isin(sess_ids, test_sess)
        
        reg = Ridge(alpha=100.0)
        reg.fit(F[train_mask], Y[train_mask])
        y_pred = reg.predict(F[test_mask])
        
        d_pred = compute_js_div(Y[test_mask], y_pred)
        d_null_test = compute_js_div(Y[test_mask], np.tile(p_global, (np.sum(test_mask), 1)))
        gain = (d_null_test - d_pred) / d_null_test * 100.0
        js_gains.append(gain)
    return np.mean(js_gains), np.std(js_gains)

# Full 100-D Baseline
gain_full, std_full = evaluate_features_cv(X)
print(f"Full 100-D Repertoire Predictive Gain: {gain_full:.2f}% ± {std_full:.2f}%")

# Sweep NMF components K
K_sweep = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
results = []

# Total variance of X
X_centered = X - np.mean(X, axis=0)
total_var = np.sum(X_centered ** 2)
total_frob = np.sum(X ** 2)

for K in K_sweep:
    print(f"\nFitting NMF with K = {K}...")
    nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    X_recon = W @ H
    
    # 1. Reconstruction Metrics
    recon_err_sq = np.sum((X - X_recon) ** 2)
    # Variance Explained (R^2 on matrix)
    var_explained = 1.0 - (recon_err_sq / total_var)
    # Relative Frobenius Error
    frob_retention = 1.0 - np.sqrt(recon_err_sq) / np.sqrt(total_frob)
    
    # 2. Predictive Retention (using W as features to predict future Y)
    # Normalize W as mixture distribution
    W_norm = W / np.clip(np.sum(W, axis=1, keepdims=True), 1e-8, None)
    pred_gain, pred_std = evaluate_features_cv(W_norm)
    retention_pct = (pred_gain / gain_full) * 100.0
    
    print(f"K = {K:2d} | Variance Explained: {var_explained*100:5.1f}% | Frob Retention: {frob_retention*100:5.1f}% | Pred Gain: {pred_gain:5.2f}% (Retention of 100-D: {retention_pct:5.1f}%)")
    results.append({
        "K": K,
        "Var_Explained": var_explained * 100.0,
        "Frob_Retention": frob_retention * 100.0,
        "Pred_Gain": pred_gain,
        "Pred_Gain_Std": pred_std,
        "Retention_of_100D": retention_pct
    })

res_df = pd.DataFrame(results)
print("\n" + "="*80)
print("SUMMARY TABLE: NMF COMPONENT NUMBER VS. RETENTION")
print("="*80)
print(res_df.to_string(index=False))
