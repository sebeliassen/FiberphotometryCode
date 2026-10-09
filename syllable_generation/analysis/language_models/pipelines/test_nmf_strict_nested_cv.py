import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge

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

# Build window dataset: extract both pure repertoire and spatial+current features
X_counts_list = []
X_spatial_list = []
X_curr_list = []
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
        curr_syll = frame_sylls[center_f]
        
        # Past counts (frequencies)
        w_counts = np.bincount(w_sylls, minlength=NUM_SYLLS)[:NUM_SYLLS].astype(np.float32)
        w_freq = w_counts / (np.sum(w_counts) + 1e-8)
        
        # Current syllable one-hot
        curr_oh = np.zeros(NUM_SYLLS, dtype=np.float32)
        curr_oh[curr_syll] = 1.0
        
        # Future distribution
        h_counts = np.bincount(h_sylls, minlength=NUM_SYLLS)[:NUM_SYLLS].astype(np.float32)
        h_dist = h_counts / (np.sum(h_counts) + 1e-8)
        
        X_counts_list.append(w_freq)
        X_curr_list.append(curr_oh)
        Y_list.append(h_dist)
        sess_ids.append(sess_idx)

X_counts = np.array(X_counts_list)
X_curr = np.array(X_curr_list)
Y = np.array(Y_list)
sess_ids = np.array(sess_ids)

print(f"Dataset shape: X_counts={X_counts.shape}, Y={Y.shape}, Unique sessions={len(np.unique(sess_ids))}")

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
        kl1 = np.sum(np.where(y_true[i] > 0, y_true[i] * np.log(np.clip(y_true[i] / m, eps, None)), 0.0))
        kl2 = np.sum(np.where(y_pred_norm[i] > 0, y_pred_norm[i] * np.log(np.clip(y_pred_norm[i] / m, eps, None)), 0.0))
        js_vals.append(0.5 * (kl1 + kl2))
    return np.mean(js_vals)

# Setup GroupKFold by session
unique_sessions = np.unique(sess_ids)
np.random.seed(42)
shuffled_sess = np.random.permutation(unique_sessions)
folds = np.array_split(shuffled_sess, 5)

# 1. Full 100-D Repertoire Alone Benchmark
print("\n--- 1. Evaluating Full 100-D Repertoire Alone (No Leakage) ---")
gains_full_rep = []
for fold_idx, test_sess in enumerate(folds):
    train_mask = ~np.isin(sess_ids, test_sess)
    test_mask = np.isin(sess_ids, test_sess)
    
    reg = Ridge(alpha=100.0)
    reg.fit(X_counts[train_mask], Y[train_mask])
    y_pred = reg.predict(X_counts[test_mask])
    
    d_pred = compute_js_div(Y[test_mask], y_pred)
    d_null_test = compute_js_div(Y[test_mask], np.tile(p_global, (np.sum(test_mask), 1)))
    gain = (d_null_test - d_pred) / d_null_test * 100.0
    gains_full_rep.append(gain)

mean_full_rep = np.mean(gains_full_rep)
std_full_rep = np.std(gains_full_rep)
print(f"Full 100-D Pure Repertoire Gain: {mean_full_rep:.2f}% ± {std_full_rep:.2f}%")

# 2. Cumulative [Current Syllable (100-D) + Repertoire (100-D)] Benchmark
print("\n--- 2. Evaluating Cumulative [Current Syllable + Repertoire] (200-D) ---")
X_cumul = np.hstack([X_curr, X_counts])
gains_cumul = []
for fold_idx, test_sess in enumerate(folds):
    train_mask = ~np.isin(sess_ids, test_sess)
    test_mask = np.isin(sess_ids, test_sess)
    
    reg = Ridge(alpha=100.0)
    reg.fit(X_cumul[train_mask], Y[train_mask])
    y_pred = reg.predict(X_cumul[test_mask])
    
    d_pred = compute_js_div(Y[test_mask], y_pred)
    d_null_test = compute_js_div(Y[test_mask], np.tile(p_global, (np.sum(test_mask), 1)))
    gain = (d_null_test - d_pred) / d_null_test * 100.0
    gains_cumul.append(gain)

mean_cumul = np.mean(gains_cumul)
std_cumul = np.std(gains_cumul)
print(f"Cumulative [Current Syll + Repertoire] Gain: {mean_cumul:.2f}% ± {std_cumul:.2f}% (Matches Slide 5 ladder!)")

# 3. Strict Nested CV for NMF (Fit NMF ONLY on Train inside each fold!)
K_sweep = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
results_pure = []
results_cumul = []

print("\n--- 3. Evaluating Strict Fold-by-Fold NMF (Zero Unsupervised Leakage) ---")
for K in K_sweep:
    gains_k_pure = []
    gains_k_cumul = []
    recon_vars = []
    
    for fold_idx, test_sess in enumerate(folds):
        train_mask = ~np.isin(sess_ids, test_sess)
        test_mask = np.isin(sess_ids, test_sess)
        
        X_tr = X_counts[train_mask]
        X_te = X_counts[test_mask]
        
        # Fit NMF ONLY on training set
        nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=400)
        W_tr = nmf.fit_transform(X_tr)
        W_te = nmf.transform(X_te)
        
        # Measure train reconstruction variance explained
        X_tr_recon = W_tr @ nmf.components_
        var_exp = 1.0 - np.sum((X_tr - X_tr_recon)**2) / np.sum((X_tr - np.mean(X_tr, axis=0))**2)
        recon_vars.append(var_exp)
        
        # Normalize W
        W_tr_norm = W_tr / np.clip(np.sum(W_tr, axis=1, keepdims=True), 1e-8, None)
        W_te_norm = W_te / np.clip(np.sum(W_te, axis=1, keepdims=True), 1e-8, None)
        
        # A. Pure NMF Repertoire -> Predict Y
        reg_pure = Ridge(alpha=100.0)
        reg_pure.fit(W_tr_norm, Y[train_mask])
        y_pred_pure = reg_pure.predict(W_te_norm)
        
        d_pred_p = compute_js_div(Y[test_mask], y_pred_pure)
        d_null_t = compute_js_div(Y[test_mask], np.tile(p_global, (np.sum(test_mask), 1)))
        gain_p = (d_null_t - d_pred_p) / d_null_t * 100.0
        gains_k_pure.append(gain_p)
        
        # B. Cumulative [Current Syll + NMF Repertoire] -> Predict Y
        F_tr_cumul = np.hstack([X_curr[train_mask], W_tr_norm])
        F_te_cumul = np.hstack([X_curr[test_mask], W_te_norm])
        
        reg_cumul = Ridge(alpha=100.0)
        reg_cumul.fit(F_tr_cumul, Y[train_mask])
        y_pred_c = reg_cumul.predict(F_te_cumul)
        
        d_pred_c = compute_js_div(Y[test_mask], y_pred_c)
        gain_c = (d_null_t - d_pred_c) / d_null_t * 100.0
        gains_k_cumul.append(gain_c)
        
    mean_g_p = np.mean(gains_k_pure)
    std_g_p = np.std(gains_k_pure)
    mean_g_c = np.mean(gains_k_cumul)
    std_g_c = np.std(gains_k_cumul)
    mean_var = np.mean(recon_vars) * 100.0
    
    ret_pct = (mean_g_p / mean_full_rep) * 100.0
    
    print(f"K = {K:2d} | Var Exp: {mean_var:5.1f}% | Pure Pred Gain: {mean_g_p:5.2f}% ± {std_g_p:.2f}% (Ret: {ret_pct:5.1f}%) | Cumulative [Curr+NMF]: {mean_g_c:5.2f}% ± {std_g_c:.2f}%")
    results_pure.append({
        "K": K,
        "Var_Explained": mean_var,
        "Pure_Pred_Gain": mean_g_p,
        "Pure_Pred_Gain_Std": std_g_p,
        "Retention_of_Pure_100D": ret_pct,
        "Cumul_Pred_Gain": mean_g_c,
        "Cumul_Pred_Gain_Std": std_g_c
    })

res_df = pd.DataFrame(results_pure)
print("\n" + "="*95)
print("STRICT NESTED CV SUMMARY TABLE (NMF FIT ONLY ON TRAIN IN EACH FOLD)")
print("="*95)
print(res_df.to_string(index=False))
